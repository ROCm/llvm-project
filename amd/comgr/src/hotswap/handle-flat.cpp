//===- handle-flat.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flat-addr.h"
#include "handlers.h"

#include "canonical-op.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>

#define DEBUG_TYPE "transpiler"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Shared helper for the `_D16_HI` half-register-store lift shape.
//
// Both `GLOBAL_STORE_SHORT_D16_HI` and `FLAT_STORE_SHORT_D16_HI` have
// the same half-register selector baked into the opcode: store bits
// [31:16] of the source VGPR (the "high half") rather than [15:0].
// The lowered shape is `lshr i32 %src32, 16` followed by `trunc i32
// to i16` -- InstCombine folds this pair to the backend-preferred
// sub-dword-extraction shape on every AMDGPU target.
//
// Kept namespace-local (rather than as a public helper) because the
// semantics are tied 1:1 to the FLAT family's sub-dword store lift
// inside this file.  DS family has its own structurally-identical
// emission in handle-ds.cpp (DS_WRITE_B16_D16_HI under
// `ds_st_d16_hi` / `ds_st_hi16_shr` breadcrumbs); MUBUF family has
// the load-side companion in handle-mubuf.cpp (via `d16Half=2` in
// `mubufClassify`).  Three addrspace-specific handlers, one
// conceptual operation; they intentionally do not share a helper
// because each resolves a different addressing / EXEC-gating
// context before this final step.
//
// Value-name breadcrumbs (`d16hi_shift` on the lshr, `d16hi_trunc`
// on the trunc) match the `ds_store_b16_d16_hi` fixture's naming
// convention in spirit (prefix-on-shift, prefix-on-trunc), adapted
// to the FLAT addrspace to keep lit patterns family-local.
Value *emitD16HiHalfTruncI16(RaiseContext &Ctx, Value *Src32) {
  Value *Shifted = Ctx.B.CreateLShr(Src32,
                                     ConstantInt::get(Ctx.I32Ty, 16),
                                     "d16hi_shift");
  return Ctx.B.CreateTrunc(Shifted, Type::getInt16Ty(Ctx.C),
                            "d16hi_trunc");
}

// Byte-store sibling of `emitD16HiHalfTruncI16`. Both `_D16_HI` store
// forms (b16 / b8) drop the low 16 bits of the source VGPR via `lshr
// 16`; the byte form additionally truncates to i8, surfacing bits
// [23:16] (the low byte of the high 16-bit half). Used by
// `GLOBAL_STORE_BYTE_D16_HI` -- the byte-store counterpart to the
// existing `GLOBAL_STORE_SHORT_D16_HI` path. Value-name breadcrumbs
// (`d16hi_shift` / `d16hi_trunc`) match the b16 helper so the lit
// family stays uniform.
Value *emitD16HiHalfTruncI8(RaiseContext &Ctx, Value *Src32) {
  Value *Shifted = Ctx.B.CreateLShr(Src32,
                                     ConstantInt::get(Ctx.I32Ty, 16),
                                     "d16hi_shift");
  return Ctx.B.CreateTrunc(Shifted, Type::getInt8Ty(Ctx.C),
                            "d16hi_trunc");
}

int64_t firstScratchImm(const DecodedInst &Di, OpResolver &Op,
                        unsigned ImmStart) {
  for (unsigned K = ImmStart; K < Op.nSrcs(); ++K) {
    if (Di.isImm(Op.srcIdx(K)))
      return Di.getImm(Op.srcIdx(K));
  }
  return 0;
}

std::string formatScratchAbiDetail(RaiseContext &Ctx, const Twine &Why) {
  using namespace llvm::amdhsa;
  std::string Detail;
  raw_string_ostream Os(Detail);
  const bool SourceEnablePrivate =
      (Ctx.SourceComputePgmRsrc2 &
       (1u << COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT)) != 0;
  const bool SourceFlatScratchInit =
      (Ctx.SourceKernelCodeProperties &
       KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) != 0;
  const bool SourcePrivateSegmentSize =
      (Ctx.SourceKernelCodeProperties &
       KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) != 0;
  Why.print(Os);
  Os << " source_scratch_kd={private_segment_fixed_size="
     << Ctx.SourcePrivateSegmentFixedSize
     << ", compute_pgm_rsrc2=0x" << utohexstr(Ctx.SourceComputePgmRsrc2)
     << ", enable_private_segment=" << (SourceEnablePrivate ? 1 : 0)
     << ", kernel_code_properties=0x"
     << utohexstr(static_cast<unsigned>(Ctx.SourceKernelCodeProperties))
     << ", enable_sgpr_flat_scratch_init="
     << (SourceFlatScratchInit ? 1 : 0)
     << ", enable_sgpr_private_segment_size="
     << (SourcePrivateSegmentSize ? 1 : 0) << "}.";
  Os.flush();
  return Detail;
}

AllocaInst *getOrCreateSourcePrivateSegment(RaiseContext &Ctx,
                                            const DecodedInst &Di,
                                            HandlerResult &Hr) {
  if (Ctx.SourcePrivateSegmentFixedSize == 0) {
    std::string Detail = formatScratchAbiDetail(
        Ctx,
        "scratch_* requires source KD private-segment allocation, but "
        "the parsed source KD reports zero private_segment_fixed_size; "
        "refusing rather than inventing scratch backing.");
    errs() << "transpiler: FLAT scratch refused: " << Di.Mnemonic
           << " -- " << Detail << "\n";
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "FLAT", Detail);
    return nullptr;
  }

  if (Ctx.ScratchPrivateSegmentAlloca)
    return Ctx.ScratchPrivateSegmentAlloca;

  BasicBlock &Entry = Ctx.Kernel->getEntryBlock();
  IRBuilder<> EntryB(&*Entry.getFirstInsertionPt());
  auto *Size = ConstantInt::get(Ctx.I32Ty, Ctx.SourcePrivateSegmentFixedSize);
  auto *Alloca =
      EntryB.CreateAlloca(Ctx.I8Ty, /*AddrSpace=*/5, Size,
                          "source_private_segment");
  Alloca->setAlignment(Align(4));
  Ctx.ScratchPrivateSegmentAlloca = Alloca;
  Ctx.UsesScratchPrivateSegment = true;
  LLVM_DEBUG(dbgs() << "transpiler: FLAT scratch ABI: allocated source "
                    << "private segment model for '" << Ctx.Kernel->getName()
                    << "' size=" << Ctx.SourcePrivateSegmentFixedSize
                    << " compute_pgm_rsrc2=0x"
                    << utohexstr(Ctx.SourceComputePgmRsrc2)
                    << " kernel_code_properties=0x"
                    << utohexstr(static_cast<unsigned>(
                           Ctx.SourceKernelCodeProperties))
                    << "\n");
  return Alloca;
}

Value *decodeScratchOffset(RaiseContext &Ctx, const DecodedInst &Di,
                           OpResolver &Op, unsigned AddrStart,
                           unsigned ElemBytes, StringRef Label,
                           HandlerResult &Hr) {
  Value *Offset = ConstantInt::get(Ctx.I32Ty, 0);
  unsigned Idx = AddrStart;

  if (Idx < Op.nSrcs() && Op.isSrcReg(Idx) &&
      Op.srcReg(Idx).RegKind == ParsedReg::VGPR) {
    ParsedReg VaddrPr = Op.srcReg(Idx++);
    Value *Vaddr = Ctx.Regs.readReg32(Ctx.B, VaddrPr);
    if (Di.HasScaleOffset)
      Vaddr = Ctx.B.CreateMul(Vaddr, ConstantInt::get(Ctx.I32Ty, ElemBytes),
                              "scratch_scaled_voff");
    Offset = Ctx.B.CreateAdd(Offset, Vaddr, "scratch_voff");
  } else if (Idx < Op.nSrcs() && Op.isSrcReg(Idx) &&
             Op.srcReg(Idx).RegKind != ParsedReg::SGPR &&
             Op.srcReg(Idx).RegKind != ParsedReg::NOREG) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "FLAT",
        (Twine(Label) + ": expected VGPR or off/null for VADDR"));
    return nullptr;
  }

  if (Idx < Op.nSrcs() && Op.isSrcReg(Idx) &&
      Op.srcReg(Idx).RegKind == ParsedReg::SGPR) {
    ParsedReg SaddrPr = Op.srcReg(Idx++);
    Offset = Ctx.B.CreateAdd(Offset, Ctx.Regs.readReg32(Ctx.B, SaddrPr),
                             "scratch_soff");
  } else if (Idx < Op.nSrcs() && Op.isSrcReg(Idx) &&
             Op.srcReg(Idx).RegKind != ParsedReg::NOREG) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "FLAT",
        (Twine(Label) + ": expected SGPR or off/null for SADDR"));
    return nullptr;
  }

  int64_t Imm = firstScratchImm(Di, Op, Idx);
  if (Imm != 0)
    Offset = Ctx.B.CreateAdd(
        Offset, ConstantInt::get(Ctx.I32Ty, static_cast<uint32_t>(Imm)),
        "scratch_iadd");

  AllocaInst *Frame = getOrCreateSourcePrivateSegment(Ctx, Di, Hr);
  if (!Frame)
    return nullptr;
  return Ctx.B.CreateGEP(Ctx.I8Ty, Frame, Offset, "scratch_ptr");
}

} // namespace

HandlerResult handleFLAT(RaiseContext &Ctx, const DecodedInst &Di,
                        OpResolver &Op) {
  HandlerResult Hr;
  StringRef Mn(Di.Mnemonic);
  CanonicalOp Sop = Di.CanonOp;

  // ---------------------------------------------------------------------
  // FLAT scratch family (`scratch_load_*`, `scratch_store_*`).
  //
  // `SIInstrFlags::FlatScratch` is the authoritative discriminator for the
  // scratch-addressing sub-family. We model it as source private-segment
  // memory, not as a global/flat pointer. The first scratch use creates a
  // single addrspace(5) private frame sized from the source KD's
  // `private_segment_fixed_size`; every translated scratch offset is a GEP
  // within that frame. That is the ABI extension: the target AMDGPU backend
  // sees real private memory, lays it out together with any target spills, and
  // emits a target KD with `enable_private_segment` /
  // `private_segment_fixed_size` instead of relying on a hand-written scratch
  // backing patch.
  //
  // If the source KD does not request private memory, scratch instructions are
  // structurally inconsistent with the source launch ABI and still refuse with
  // KD fields in the diagnostic.
  if (Di.TsFlags & SIInstrFlags::FlatScratch) {
    auto ScratchAccessBytes = [&]() -> unsigned {
      switch (Sop) {
      case CanonicalOp::SCRATCH_LOAD_DWORD:
      case CanonicalOp::SCRATCH_STORE_DWORD:
        return 4;
      case CanonicalOp::SCRATCH_LOAD_DWORDX2:
      case CanonicalOp::SCRATCH_STORE_DWORDX2:
        return 8;
      case CanonicalOp::SCRATCH_LOAD_DWORDX3:
      case CanonicalOp::SCRATCH_STORE_DWORDX3:
        return 12;
      case CanonicalOp::SCRATCH_LOAD_DWORDX4:
      case CanonicalOp::SCRATCH_STORE_DWORDX4:
        return 16;
      default:
        return 0;
      }
    };
    auto ScratchAlign = [](unsigned Bytes) -> Align {
      // dwordx3 is a 12-byte access; LLVM Align requires a power of two.
      // The ISA only promises dword granularity for scratch offsets, so use a
      // conservative 4-byte alignment for that aggregate width.
      return Bytes == 12 ? Align(4) : Align(Bytes);
    };

    unsigned AccessBytes = ScratchAccessBytes();
    if (AccessBytes == 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "FLAT",
          formatScratchAbiDetail(
              Ctx,
              Twine("scratch_* opcode is not a load/store shape Hotswap "
                    "models yet: ") +
                  Di.Mnemonic));
      return Hr;
    }

    if (Sop == CanonicalOp::SCRATCH_LOAD_DWORD ||
        Sop == CanonicalOp::SCRATCH_LOAD_DWORDX2 ||
        Sop == CanonicalOp::SCRATCH_LOAD_DWORDX3 ||
        Sop == CanonicalOp::SCRATCH_LOAD_DWORDX4) {
      if (Op.nSrcs() < 2) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "FLAT", "scratch_load_* expected address/cpol operands");
        return Hr;
      }

      Value *Addr = decodeScratchOffset(Ctx, Di, Op, /*addrStart=*/0,
                                        AccessBytes, "scratch_load", Hr);
      if (Hr.Failure.hasFailed())
        return Hr;

      ParsedReg Dest = Op.dst();
      Type *LoadTy = nullptr;
      switch (Sop) {
      case CanonicalOp::SCRATCH_LOAD_DWORD:
        LoadTy = Ctx.I32Ty;
        break;
      case CanonicalOp::SCRATCH_LOAD_DWORDX2:
        LoadTy = FixedVectorType::get(Ctx.I32Ty, 2);
        break;
      case CanonicalOp::SCRATCH_LOAD_DWORDX3:
        LoadTy = FixedVectorType::get(Ctx.I32Ty, 3);
        break;
      case CanonicalOp::SCRATCH_LOAD_DWORDX4:
        LoadTy = FixedVectorType::get(Ctx.I32Ty, 4);
        break;
      default:
        llvm_unreachable("scratch load dispatch drifted");
      }

      Ctx.emitUnderExec([&] {
        Value *Loaded = Ctx.B.CreateAlignedLoad(
            LoadTy, Addr, ScratchAlign(AccessBytes), "scratch_load");
        if (AccessBytes == 4) {
          Ctx.Regs.writeReg32(Ctx.B, Dest, Loaded);
        } else {
          unsigned Dwords = AccessBytes / 4;
          for (unsigned D = 0; D < Dwords; ++D) {
            ParsedReg Sub = Dest;
            Sub.BaseIdx = Dest.BaseIdx + static_cast<int>(D);
            Sub.Width = 1;
            Ctx.Regs.writeReg32(
                Ctx.B, Sub,
                Ctx.B.CreateExtractElement(Loaded, Ctx.B.getInt32(D)));
          }
        }
      });
      Hr.Handled = true;
      return Hr;
    }

    if (Sop == CanonicalOp::SCRATCH_STORE_DWORD ||
        Sop == CanonicalOp::SCRATCH_STORE_DWORDX2 ||
        Sop == CanonicalOp::SCRATCH_STORE_DWORDX3 ||
        Sop == CanonicalOp::SCRATCH_STORE_DWORDX4) {
      if (Op.nSrcs() < 3) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "FLAT",
            "scratch_store_* expected data plus address/cpol operands");
        return Hr;
      }

      Value *Addr = decodeScratchOffset(Ctx, Di, Op, /*addrStart=*/1,
                                        AccessBytes, "scratch_store", Hr);
      if (Hr.Failure.hasFailed())
        return Hr;

      ParsedReg StData = Op.srcReg(0);
      if (StData.RegKind != ParsedReg::VGPR) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "FLAT", "scratch_store_* expected VGPR data operand");
        return Hr;
      }

      Ctx.emitUnderExec([&] {
        if (AccessBytes == 4) {
          Ctx.B.CreateAlignedStore(Ctx.Regs.readReg32(Ctx.B, StData), Addr,
                                   Align(4));
        } else {
          auto *VecTy = FixedVectorType::get(Ctx.I32Ty, AccessBytes / 4);
          Ctx.B.CreateAlignedStore(Ctx.Regs.readRegVec(Ctx.B, StData, VecTy),
                                   Addr, ScratchAlign(AccessBytes));
        }
      });
      Hr.Handled = true;
      return Hr;
    }

    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "FLAT",
        formatScratchAbiDetail(
            Ctx,
            Twine("scratch_* shape reached unreachable dispatch: ") +
                Di.Mnemonic));
    return Hr;
  }

  if (Sop == CanonicalOp::GLOBAL_LOAD_USHORT || Sop == CanonicalOp::GLOBAL_LOAD_SHORT_D16_HI ||
      Sop == CanonicalOp::GLOBAL_LOAD_SSHORT || Sop == CanonicalOp::GLOBAL_LOAD_UBYTE ||
      Sop == CanonicalOp::GLOBAL_LOAD_SBYTE) {
    ParsedReg Dest = Op.dst();
    bool IsByte = Sop == CanonicalOp::GLOBAL_LOAD_UBYTE || Sop == CanonicalOp::GLOBAL_LOAD_SBYTE;
    Type *LoadTy = IsByte ? Ctx.I8Ty : Type::getInt16Ty(Ctx.C);
    // The ISA only guarantees natural alignment for each sub-dword access:
    // 1 byte for byte loads, 2 bytes for short loads. Using the ABI default
    // alignment here was over-promising for buffers legitimately aligned
    // to the element size.
    Align LoadAlign = Align(IsByte ? 1 : 2);

    FlatAddr Fa = decodeGlobalLoadAddr(Ctx, Di, Op, IsByte ? 1 : 2,
                                        "GLOBAL_LOAD sub-dword");
    Value *Addr = Fa.Ptr;
    // SPE-gate the memory access itself, not just the VGPR write-back.
    // The store counterparts (GLOBAL_STORE_*, ~line 196 below) are
    // already wrapped in `emitUnderExec`; the asymmetric pre-2026-04-22
    // handling where loads fired on every target lane -- including
    // WaveNative "phantom" lanes whose source-wave had no workitem at
    // this position -- caused HIP error 700 "illegal memory access"
    // when phantom-lane pointer-arithmetic VGPRs held `undef` /
    // stale-VGPR-slot values that pointed outside any allocated
    // region. See `handle-flat.cpp::FLAT_LOAD_*` block comment below
    // (and the matching audit of sub-dword, dword, and FLAT_LOAD
    // variants) for the full rationale. `ctx.Regs.writeReg32` (the
    // low-level alloca path) is called inside the body rather than
    // `ctx.writeReg32` (which would wrap the write in a nested
    // `emitUnderExec` -- harmless but redundant IR).
    Ctx.emitUnderExec([&] {
      bool IsUnsigned = Sop == CanonicalOp::GLOBAL_LOAD_UBYTE ||
                        Sop == CanonicalOp::GLOBAL_LOAD_USHORT;
      Value *Loaded = Ctx.B.CreateAlignedLoad(LoadTy, Addr, LoadAlign,
                                              "gload_sub");
      Value *Ext = IsUnsigned ? Ctx.B.CreateZExt(Loaded, Ctx.I32Ty)
                              : Ctx.B.CreateSExt(Loaded, Ctx.I32Ty);
      if (Sop == CanonicalOp::GLOBAL_LOAD_SHORT_D16_HI) {
        Value *Prev = Ctx.Regs.readReg32(Ctx.B, Dest);
        Ext = Ctx.B.CreateOr(
            Ctx.B.CreateAnd(Prev, ConstantInt::get(Ctx.I32Ty, 0xFFFF)),
            Ctx.B.CreateShl(Ext, 16), "d16hi");
      }
      Ctx.Regs.writeReg32(Ctx.B, Dest, Ext);
    });
    Hr.Handled = true;
    return Hr;
  }

  if (Sop == CanonicalOp::GLOBAL_LOAD_DWORD || Sop == CanonicalOp::GLOBAL_LOAD_DWORDX2 ||
      Sop == CanonicalOp::GLOBAL_LOAD_DWORDX3 || Sop == CanonicalOp::GLOBAL_LOAD_DWORDX4) {
    int LoadDwords = 1;
    if (Sop == CanonicalOp::GLOBAL_LOAD_DWORDX2) LoadDwords = 2;
    else if (Sop == CanonicalOp::GLOBAL_LOAD_DWORDX3) LoadDwords = 3;
    else if (Sop == CanonicalOp::GLOBAL_LOAD_DWORDX4) LoadDwords = 4;

    ParsedReg Dest = Op.dst();

    FlatAddr Fa = decodeGlobalLoadAddr(Ctx, Di, Op, LoadDwords * 4,
                                        "GLOBAL_LOAD dword");
    Value *Addr = Fa.Ptr;

    // Same SPE-gating rationale as the GLOBAL_LOAD sub-dword block
    // above: without `emitUnderExec` wrapping the `CreateLoad`, every
    // target lane -- including WaveNative "phantom" lanes whose
    // pointer-arithmetic VGPRs hold `undef` / stale slot data --
    // dereferences the addr VGPR pair and faults at runtime (HIP
    // error 700).  For the vector-load case (`DWORDX{2,3,4}`), the
    // single load + N extract-write pairs all go inside one
    // emitUnderExec block so inactive lanes skip the whole sequence.
    Ctx.emitUnderExec([&] {
      if (LoadDwords == 1) {
        Ctx.Regs.writeReg32(
            Ctx.B, Dest,
            Ctx.B.CreateBitCast(
                Ctx.B.CreateLoad(Ctx.F32Ty, Addr, "gload"),
                Ctx.I32Ty));
      } else {
        Type *VecTy = FixedVectorType::get(Ctx.I32Ty, LoadDwords);
        Value *Loaded = Ctx.B.CreateLoad(VecTy, Addr, "gload");
        for (int D = 0; D < LoadDwords; D++) {
          ParsedReg Sub = Dest;
          Sub.BaseIdx = Dest.BaseIdx + D;
          Sub.Width = 1;
          Ctx.Regs.writeReg32(
              Ctx.B, Sub,
              Ctx.B.CreateExtractElement(Loaded, Ctx.B.getInt32(D)));
        }
      }
    });
    Hr.Handled = true;
    return Hr;
  }

  if (Sop == CanonicalOp::GLOBAL_STORE_BYTE || Sop == CanonicalOp::GLOBAL_STORE_BYTE_D16_HI ||
      Sop == CanonicalOp::GLOBAL_STORE_SHORT ||
      Sop == CanonicalOp::GLOBAL_STORE_SHORT_D16_HI || Sop == CanonicalOp::GLOBAL_STORE_DWORD ||
      Sop == CanonicalOp::GLOBAL_STORE_DWORDX2 || Sop == CanonicalOp::GLOBAL_STORE_DWORDX3 ||
      Sop == CanonicalOp::GLOBAL_STORE_DWORDX4) {
    int StoreDwords = 1;
    int StoreBits = 32;
    // `_D16_HI` variants store bits [31:16] of the source VGPR rather
    // than [15:0] -- a half-register selector baked into the opcode
    // (AMDGPU ISA; see `global_store_d16_hi_b16` in
    // FLATInstructions.td and handle-ds.cpp's DS_WRITE_B16_D16_HI for
    // the existing DS-family precedent).  The compiler emits this
    // form to write the upper-16-bits half of a 32-bit value without
    // an explicit `v_lshrrev_b32` shift -- idiomatic in the fp32->bf16
    // round-to-nearest-even epilogue (`v_add3_u32 v, bits, odd_bit,
    // 0x7fff` produces the RNE-biased sum in a 32-bit VGPR, and
    // `global_store_d16_hi_b16` writes its upper 16 bits = the bf16
    // result).  Pre-fix, `storeHiHalf=false` for
    // `GLOBAL_STORE_SHORT_D16_HI` was silently wrong: every bf16-cast-
    // store kernel (Triton's `.to(tl.bfloat16) + tl.store` shape, which
    // the observed-production `topk_forward_bisect_m_laneprobe` recipe
    // exercises with no cross-lane ops) stored the LOW 16 bits of the
    // biased sum, reading as NaN-ish (`0x7FFF`) for typical values.
    bool StoreHiHalf = false;
    if (Sop == CanonicalOp::GLOBAL_STORE_DWORDX4) StoreDwords = 4;
    else if (Sop == CanonicalOp::GLOBAL_STORE_DWORDX3) StoreDwords = 3;
    else if (Sop == CanonicalOp::GLOBAL_STORE_DWORDX2) StoreDwords = 2;
    else if (Sop == CanonicalOp::GLOBAL_STORE_DWORD) StoreDwords = 1;
    else if (Sop == CanonicalOp::GLOBAL_STORE_SHORT ||
             Sop == CanonicalOp::GLOBAL_STORE_SHORT_D16_HI) {
      StoreBits = 16;
      StoreDwords = 0;
      StoreHiHalf = (Sop == CanonicalOp::GLOBAL_STORE_SHORT_D16_HI);
    } else if (Sop == CanonicalOp::GLOBAL_STORE_BYTE ||
               Sop == CanonicalOp::GLOBAL_STORE_BYTE_D16_HI) {
      StoreBits = 8;
      StoreDwords = 0;
      StoreHiHalf = (Sop == CanonicalOp::GLOBAL_STORE_BYTE_D16_HI);
    }

    // scale_offset on stores scales the per-lane vaddr by the access
    // element size. For sub-dword stores (byte/short) the element size
    // is 1 or 2 bytes; for dword/dwordx{2,3,4} it is 4, 8, 12, or 16
    // bytes -- the compiler emits `global_store_dwordx4 … scale_offset`
    // with a lane-index vaddr to lower `out[tid] = vec4` patterns.
    int ElemBytes = StoreBits < 32 ? (StoreBits / 8)
                                    : std::max(StoreDwords, 1) * 4;
    FlatAddr Fa =
        decodeGlobalStoreAddr(Ctx, Di, Op, ElemBytes, "GLOBAL_STORE");
    Value *Addr = Fa.Ptr;
    ParsedReg StData = Fa.StData;

    if (StoreDwords == 0) {
      Value *Src32 = Ctx.Regs.readReg32(Ctx.B, StData);
      // `_D16_HI` variants route through the shared half-register
      // helpers that emit `lshr 16 + trunc to iN` (N = 16 for the
      // short form, 8 for the byte form -- the b8 form surfaces
      // bits [23:16], the low byte of the high half).  The non-
      // `_D16_HI` short / byte path takes a plain trunc to `memTy`.
      Value *Val;
      if (StoreHiHalf) {
        Val = (StoreBits == 8) ? emitD16HiHalfTruncI8(Ctx, Src32)
                                : emitD16HiHalfTruncI16(Ctx, Src32);
      } else {
        Val = Ctx.B.CreateTrunc(Src32, Type::getIntNTy(Ctx.C, StoreBits));
      }
      Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Addr); });
    } else if (StoreDwords == 1) {
      Value *Val = Ctx.Regs.readReg32(Ctx.B, StData);
      Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Addr); });
    } else {
      auto *VecTy = FixedVectorType::get(Ctx.I32Ty, StoreDwords);
      Value *Val = Ctx.Regs.readRegVec(Ctx.B, StData, VecTy);
      Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Addr); });
    }
    Hr.Handled = true;
    return Hr;
  }

  // ---------------------------------------------------------------------
  // gfx1250 async global -> LDS load (FLAT VFLAT 0x5f-0x62 -- b8 / b32 /
  // b64 / b128).
  //
  // Operand layout from `FLAT_Global_Load_LDS_Pseudo<…, IsAsync=1>`
  // (FLATInstructions.td:391-417) is identical across all four widths
  // (only the data byte size -- and so the intrinsic ID / per-lane load
  // type -- varies):
  //
  //   plain (4 srcs): vdst:VGPR_32, vaddr:VGPR_64,            offset, cpol
  //   SADDR (5 srcs): vdst:VGPR_32, saddr:SReg_64, vaddr:VGPR_32, offset, cpol
  //
  // `vdst` is in the *input* list (because `has_vdst = IsAsync = 1`)
  // and carries the per-lane LDS i32 base offset. The same-target
  // arm lowers to
  // `int_amdgcn_global_load_async_to_lds_b{8,32,64,128}`
  // (IntrinsicsAMDGPU.td:3939-3946, all sharing the
  // `AMDGPUAsyncGlobalLoadToLDS` signature on line 3904), which
  // consumes the LDS pointer as `local_ptr_ty` -- materialised via
  // `inttoptr i32 -> ptr addrspace(3)`. The cross-target arm emits a
  // synchronous per-lane `load <T> + store <T>` pair against the
  // same decoded operands; see the `GLOBAL_LOAD_ASYNC_TO_LDS_B*`
  // CanonicalOp doc block in `canonical-op.h` for the correctness argument and
  // the documented semantic trade-off (loss of pipelining overlap,
  // NOT of per-lane LDS state).
  //
  // Operand parsing is shared between the two arms so any shape
  // that lifts on gfx1250 lifts identically on gfx942, modulo the
  // emission tail. The ISA source-of-truth pragma
  // (`instruction_manual.pdf §13.6.9-12`, verbatim -- identical
  // across all four widths except the LDS-store width):
  //
  //   pragma "vector" do
  //     dsaddr  = LDS_BASE.b32 + VGPR[laneId][VDST.u32] + INST_OFFSET.b32;
  //     memaddr = ADDR;  // CalcGlobalAddr(VADDR, SADDR, IOFFSET)
  //     LDS[dsaddr].bN = MEM[memaddr].bN   // (N = 8/32/64/128)
  //   endpragma
  //
  // Key observations the emulation relies on:
  //   * `INST_OFFSET` is applied to BOTH the LDS address and the
  //     global address. The same-target intrinsic takes `offset` as
  //     a single immarg and the backend re-folds it onto both the
  //     computed dsaddr AND the memaddr during isel; the
  //     cross-target emulation reaches the same effect explicitly
  //     by GEP'ing the offset onto both pointers before the
  //     load/store pair.
  //   * `pragma "vector" do` runs the body per-active-lane. Both
  //     arms wrap in `emitUnderExec` so inactive lanes skip the
  //     entire memory round-trip (matches hardware EXEC gating).
  if (Sop == CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B8 ||
      Sop == CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B32 ||
      Sop == CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B64 ||
      Sop == CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B128) {
    // ---- Width -> (access type, bytes) dispatch (used by both arms) ----
    //
    // The ISA pragma stores per lane:
    //   b8   ->  1 byte                  (i8)
    //   b32  ->  4 bytes                 (i32)
    //   b64  ->  8 bytes  = 2  x i32     (<2 x i32>)
    //   b128 ->  16 bytes = 4  x i32     (<4 x i32>)
    //
    // `accessBytes` is consumed on the SADDR path for
    // `scale_offset` (cpol bit 0x400) -- the ISA specifies that the
    // scaled-offset mode multiplies the per-lane VGPR vaddr by the
    // access element size before adding it to the SGPR base.  The
    // same-target arm passes cpol to the intrinsic and lets the
    // gfx12 hardware encoding do the multiply on isel; the
    // cross-target arm has to materialise the multiply explicitly
    // because gfx942 has no equivalent encoding bit.  `accessTy`
    // is only used by the cross-target arm.
    //
    // Larger widths are lifted as vectors of i32 (mirroring the
    // GLOBAL_LOAD_DWORDX{2,3,4} path's handling of the same
    // aggregate shape) rather than as a single `iN` whose natural
    // alignment (16 B for b128) the source buffer would not
    // necessarily satisfy -- the vector type lets the backend
    // pick a `global_load_b{64,128}` opcode with the aligned-load
    // attribute below.
    Type *AccessTy = nullptr;
    unsigned AccessBytes = 0;
    switch (Sop) {
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B8:
      AccessTy = Ctx.I8Ty; AccessBytes = 1; break;
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B32:
      AccessTy = Ctx.I32Ty; AccessBytes = 4; break;
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B64:
      AccessTy = FixedVectorType::get(Ctx.I32Ty, 2); AccessBytes = 8; break;
    case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B128:
      AccessTy = FixedVectorType::get(Ctx.I32Ty, 4); AccessBytes = 16; break;
    default:
      llvm_unreachable("dispatch matched async-to-LDS family but width "
                       "CanonicalOp fell through the access-type switch");
    }

    // ---- Shape validation (identical on both arms) ----
    bool IsSaddr = false;
    if (Op.nSrcs() == 5) {
      IsSaddr = true;
    } else if (Op.nSrcs() != 4) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "FLAT",
          "global_load_async_to_lds_b*: expected 4 srcs (plain) or "
          "5 srcs (SADDR) per FLAT_Global_Load_LDS_Pseudo<IsAsync=1>");
      return Hr;
    }

    ParsedReg VdstPr = Op.srcReg(0);
    if (VdstPr.RegKind != ParsedReg::VGPR) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "FLAT",
          "global_load_async_to_lds_b*: vdst (LDS-base operand) is "
          "not a VGPR");
      return Hr;
    }
    Value *LdsOff = Ctx.Regs.readReg32(Ctx.B, VdstPr);
    Type *PtrLdsTy = PointerType::get(Ctx.C, /*addrspace=*/3);
    Value *LdsPtr = Ctx.B.CreateIntToPtr(LdsOff, PtrLdsTy, "lds_ptr");

    Value *GlobalAddr = nullptr;
    unsigned ImmStart = 0;
    if (IsSaddr) {
      ParsedReg SaddrPr = Op.srcReg(1);
      ParsedReg VaddrPr = Op.srcReg(2);
      if (SaddrPr.RegKind != ParsedReg::SGPR ||
          VaddrPr.RegKind != ParsedReg::VGPR) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "FLAT",
            "global_load_async_to_lds_b* SADDR: expected "
            "(SGPR_64, VGPR_32) for (saddr, vaddr)");
        return Hr;
      }
      Value *Saddr = Ctx.Regs.readReg64(Ctx.B, SaddrPr);
      // `zext` (not `sext`): the ISA programming manual §4.9.9
      // ("Instruction Fields") specifies that in the SADDR form
      // the VGPR vaddr is an "unsigned byte offset" added to the
      // SGPR base. The address-space decoder in `flat-addr.cpp`
      // uses `sext` for the general GLOBAL_LOAD SADDR family -- an
      // orthogonal pre-existing inconsistency (not addressed
      // here). We mirror the same-target arm's historical choice
      // so same-target and cross-target emit the same effective
      // per-lane global address; any correction to the signed/
      // unsigned semantics should be made in one place across the
      // whole GLOBAL_LOAD surface.
      Value *Voff = Ctx.B.CreateZExt(
          Ctx.Regs.readReg32(Ctx.B, VaddrPr), Ctx.I64Ty, "voff_zext");
      // The SADDR form's address is computed as `saddr + voff` at
      // this point, WITHOUT the `scale_offset` multiplier applied.
      // Reasoning, one clause per arm:
      //
      //   Same-target (gfx1250 -> gfx1250, `hasTensorOps == true`):
      //     the intrinsic consumes cpol as an immarg that includes
      //     the `SCAL` bit (`AMDGPU::CPol::SCAL = 0x400`); on isel
      //     the backend matches the `saddr + voff` pattern on the
      //     intrinsic's `%gaddr` operand AND the cpol's SCAL bit,
      //     and emits the `global_load_async_to_lds_*_SADDR ...
      //     scale_offset` real -- the HARDWARE applies the scale on
      //     execution.  Pre-multiplying in IR would produce
      //     `saddr + voff * N` as the `%gaddr`, which the backend
      //     would either (a) no longer recognise as a SADDR
      //     pattern, falling back to the plain form with unscaled
      //     hardware behaviour (correct but bypasses the SADDR
      //     optimisation) or (b) match the pattern anyway and
      //     apply SCAL on top, producing `saddr + voff * N * N`
      //     (silent 4x / 16x / 64x miscompile).  Neither outcome
      //     is desirable.  The same-target lit fixture
      //     (`global_load_async_to_lds_same_target.ll`) pins the
      //     `inttoptr i64` shape with no intervening multiply --
      //     dropping the gate below would immediately fail that
      //     fixture with a double-scale in IR.
      //
      //   Cross-target (gfx1250 -> gfx942, `hasTensorOps == false`):
      //     we emit plain `load` / `store` without an intrinsic, so
      //     there is no backend folding to rely on; the multiply
      //     MUST be materialised explicitly here for the lane
      //     address to match the source's per-lane semantics.
      //     When `hasScaleOffset` is false the multiply is elided
      //     (no observable change in either arm), so the same
      //     code path covers the HIP-builtin fixture's
      //     `scale_offset` form and the corpus matmul_ogs form.
      //
      // Reading `di.hasScaleOffset` (the decoded typed boolean)
      // rather than `cpolImm & AMDGPU::CPol::SCAL` keeps the
      // classification in one place -- see
      // `decode.cpp::decodeScaleOffset` for the authoritative
      // SCAL-bit extraction.
      if (Di.HasScaleOffset && !Ctx.TargetIsa.HasTensorOps)
        Voff = Ctx.B.CreateMul(
            Voff, ConstantInt::get(Ctx.I64Ty, AccessBytes),
            "scaled_voff");
      GlobalAddr = Ctx.B.CreateAdd(Saddr, Voff, "saddr_vaddr");
      ImmStart = 3;
    } else {
      ParsedReg VaddrPr = Op.srcReg(1);
      if (VaddrPr.RegKind != ParsedReg::VGPR) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "FLAT",
            "global_load_async_to_lds_b* plain: expected VGPR_64 "
            "for vaddr");
        return Hr;
      }
      GlobalAddr = Ctx.Regs.readReg64(Ctx.B, VaddrPr);
      ImmStart = 2;
    }

    // Trailing (offset, cpol) imm pair -- first imm is `flat_offset`
    // (signed 13-bit in the encoding, already sign-extended by MC),
    // second is `cpol` (gfx12+ cachepolicy bitfield: th, scope,
    // scale_offset).
    int64_t FlatOffset = 0;
    int64_t CpolImm = 0;
    bool SawOffset = false;
    for (unsigned K = ImmStart; K < Op.nSrcs(); ++K) {
      if (!Di.isImm(Op.srcIdx(K))) continue;
      int64_t V = Di.getImm(Op.srcIdx(K));
      if (!SawOffset) {
        FlatOffset = V;
        SawOffset = true;
      } else {
        CpolImm = V;
      }
    }

    Value *GlobalPtr = GlobalAddr;
    if (GlobalPtr->getType() != Ctx.PtrGlobalTy)
      GlobalPtr = Ctx.B.CreateIntToPtr(GlobalPtr, Ctx.PtrGlobalTy);

    // ---- Per-arm emission ----
    if (Ctx.TargetIsa.HasTensorOps) {
      // Same-target (gfx1250 -> gfx1250): emit the native intrinsic
      // directly; `IntrInaccessibleMemOrArgMemOnly` on the
      // intrinsic prevents downstream passes from CSEing or
      // reordering the asynchronous fetch across companion
      // `s_wait_asynccnt` / `s_wait_tensorcnt` barriers.
      Intrinsic::ID Iid;
      switch (Sop) {
      case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B8:
        Iid = Intrinsic::amdgcn_global_load_async_to_lds_b8; break;
      case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B32:
        Iid = Intrinsic::amdgcn_global_load_async_to_lds_b32; break;
      case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B64:
        Iid = Intrinsic::amdgcn_global_load_async_to_lds_b64; break;
      case CanonicalOp::GLOBAL_LOAD_ASYNC_TO_LDS_B128:
        Iid = Intrinsic::amdgcn_global_load_async_to_lds_b128; break;
      default:
        llvm_unreachable("dispatch matched async-to-LDS family but width CanonicalOp "
                         "fell through the switch");
      }
      Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Iid);
      Value *OffsetArg = ConstantInt::get(Ctx.I32Ty, FlatOffset);
      Value *CpolArg = ConstantInt::get(Ctx.I32Ty, CpolImm);
      Ctx.emitUnderExec([&] {
        Ctx.B.CreateCall(Fn, {GlobalPtr, LdsPtr, OffsetArg, CpolArg});
      });
      Hr.Handled = true;
      return Hr;
    }

    // Cross-target (gfx942 and earlier): synchronous per-lane
    // emulation.  See the CanonicalOp doc block in `canonical-op.h` for the
    // trade-off argument (throughput regression via loss of async
    // pipelining overlap; per-lane final LDS state identical to
    // the source's state observed after `s_wait_asynccnt 0`).
    //
    // Apply `flat_offset` to BOTH pointers via i8-GEP before the
    // load/store (matching the ISA pragma's
    // `dsaddr = ... + INST_OFFSET` AND `memaddr = CalcGlobalAddr(
    // ..., IOFFSET)` -- the same-target intrinsic folds both onto
    // the operand bank via its `offset` immarg; the cross-target
    // emulation materialises the GEP chains instead).
    //
    // Non-inbounds GEP because the ISA's signed 13-bit
    // `flat_offset` can legitimately leave the nominal allocation
    // (negative strides, compiler-scheduled prefetches), same
    // convention as `flat-addr.cpp::toGlobalPtr`.  Zero offsets
    // are elided to keep the IR shape compact when the immediate
    // is unused -- which is the common case in the observed
    // corpus.
    Value *EmuGlobalPtr = GlobalPtr;
    Value *EmuLdsPtr = LdsPtr;
    if (FlatOffset != 0) {
      EmuGlobalPtr = Ctx.B.CreateGEP(Ctx.I8Ty, EmuGlobalPtr,
                                      Ctx.B.getInt64(FlatOffset),
                                      "async_gptr_off");
      EmuLdsPtr = Ctx.B.CreateGEP(Ctx.I8Ty, EmuLdsPtr,
                                   Ctx.B.getInt64(FlatOffset),
                                   "async_lptr_off");
    }

    // cpol (gfx12+ cachepolicy bits: th, scope, scale_offset) has
    // no direct gfx942 equivalent.  The `scale_offset` bit has
    // already been materialised into the `saddr_vaddr` address
    // above via the `di.hasScaleOffset` branch (the gate is
    // `!hasTensorOps` so the same-target arm's intrinsic-driven
    // base is untouched).  The remaining bits (`th` temporal
    // hint, `scope` CU/DEV/SYS) are target-level tuning hints
    // without a gfx942 encoding -- the backend re-derives cache
    // behaviour from the IR's `load` / `store` ordering (default
    // semantics = unordered).  Dropping them is the same posture
    // that `sync-translation.md §5.3` takes for atomic scope
    // recovery on pre-gfx12 targets when the decoded modifiers
    // have no representable mapping.  The value is read only to
    // avoid an unused-variable warning in release builds.
    (void)CpolImm;

    // Natural alignment for the access. The ISA pragma requires
    // N-byte aligned accesses (no sub-element strides), so each
    // access is aligned to `accessBytes`. The backend's
    // alignment-derived codegen pass picks the appropriate
    // `global_load_b{8,32,64,128}` (CDNA3 naming:
    // `global_load_{ubyte,dword,dwordx2,dwordx4}`) and
    // `ds_store_b{8,32,64,128}` opcodes from that alignment.
    Align AccessAlign(AccessBytes);

    // gfx12 `global_load_async_to_lds` drops a lane whose LDS-destination
    // offset is out of range: its LDS write does not happen (gfx12 programming
    // manual, "Async LDS Load/Store" -- out of range is past the LDS allocated
    // to the workgroup, and unconditionally past the physical LDS size).
    // Triton predicates the masked (padding / K-edge) rows of a GEMM tile load
    // this way, parking their LDS destination at the INT_MAX sentinel; those
    // rows' global tile addresses are intentionally out of bounds and their
    // LDS slots already hold the `other` value.
    //
    // The synchronous emulation must skip such a lane entirely: its loaded
    // value is unused (the LDS write is dropped), and a real load of the masked
    // lane's OOB global address faults when that address is unmapped. That
    // fault is allocation-dependent -- benign when a loose allocator leaves the
    // page mapped, fatal under a packed runtime heap -- so it surfaces in a
    // full runtime but not in isolated single-kernel replay. Gate on the
    // target's physical LDS capacity: a conservative, allocation-independent
    // out-of-range bound that keeps every real destination and rejects the
    // sentinel (far past it).
    Value *LdsInBounds = Ctx.B.CreateICmpULT(
        LdsOff, ConstantInt::get(Ctx.I32Ty, Ctx.TargetIsa.LdsByteCapacity),
        "async_lds_inb");
    Ctx.emitUnderExec([&] {
      BasicBlock *PredBb = Ctx.B.GetInsertBlock();
      Function *Fn = PredBb->getParent();
      BasicBlock *DoBb = BasicBlock::Create(Ctx.C, "async_lds_do", Fn);
      BasicBlock *ContBb = BasicBlock::Create(Ctx.C, "async_lds_cont", Fn);
      Ctx.B.CreateCondBr(LdsInBounds, DoBb, ContBb);
      Ctx.B.SetInsertPoint(DoBb);
      Value *Loaded = Ctx.B.CreateAlignedLoad(AccessTy, EmuGlobalPtr,
                                              AccessAlign, "async_gload");
      Ctx.B.CreateAlignedStore(Loaded, EmuLdsPtr, AccessAlign);
      Ctx.B.CreateBr(ContBb);
      Ctx.B.SetInsertPoint(ContBb);
    });

    Hr.Handled = true;
    return Hr;
  }

  // ---------------------------------------------------------------------
  // gfx1250 FLAT VMEM prefetch (VFLAT 0x05D -- flat_prefetch_b8 /
  // global_prefetch_b8).
  //
  // Operand layout from `FLAT_Prefetch_Pseudo` (FLATInstructions.td
  // :525-553) -- note `has_vdst = 0`, so there is no dst slot:
  //
  //   plain (3 srcs): vaddr:VGPR_64,            offset, cpol
  //   SADDR (4 srcs): saddr:SReg_64, vaddr:VGPR_32, offset, cpol
  //
  // Lifts to `int_amdgcn_{flat,global}_prefetch(ptr, cpol)`; the
  // FLAT `flat_offset` is folded onto the pointer via a non-inbounds
  // GEP before the call (the intrinsic itself takes no offset
  // operand). The call sits OUTSIDE `emitUnderExec` because the
  // intrinsic carries the EXEC mask implicitly through
  // `IntrInaccessibleMemOrArgMemOnly` -- a hint with no observable
  // side effect on inactive lanes, so an extra `if-spe-active` guard
  // would gratuitously inflate IR for what hardware executes as a
  // single broadcast hint.
  //
  // gfx942 has no VMEM-prefetch encoding (the intrinsic is gated by
  // `HasVmemPrefInsts`, only set on gfx1250+), so a cross-target
  // lift is refused loudly. See the CanonicalOp docstrings in
  // `canonical-op.h` for the design rationale.
  if (Sop == CanonicalOp::GLOBAL_PREFETCH_B8 ||
      Sop == CanonicalOp::FLAT_PREFETCH_B8) {
    const bool IsFlatPrefetch = (Sop == CanonicalOp::FLAT_PREFETCH_B8);
    Type *PrefetchPtrTy = IsFlatPrefetch
                              ? PointerType::get(Ctx.C, AMDGPUAS::FLAT_ADDRESS)
                              : Ctx.PtrGlobalTy;
    Intrinsic::ID PrefetchIntrinsic = IsFlatPrefetch
                                          ? Intrinsic::amdgcn_flat_prefetch
                                          : Intrinsic::amdgcn_global_prefetch;

    if (!Ctx.TargetIsa.HasTensorOps) {
      llvm::errs()
          << "transpiler: FLAT: " << Di.Mnemonic
          << " has no equivalent on the compilation target "
          << "(gfx1250 VMEM-prefetch unit; LLVM intrinsic "
          << Intrinsic::getName(PrefetchIntrinsic)
          << " is gated by HasVmemPrefInsts, "
          << "only set on gfx1250+). The closest sibling "
          << "amdgcn.s.prefetch.data requires a uniform SGPR "
          << "pointer which we cannot prove for the divergent "
          << "VGPR address used here without divergence "
          << "analysis -- refusing to emit a fallback or silently "
          << "drop the hint.\n";
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "FLAT",
          "gfx1250-only VMEM prefetch (HasVmemPrefInsts); no "
          "equivalent on non-gfx1250 compilation target. The "
          "amdgcn.s.prefetch.data sibling requires a uniform "
          "pointer (the VMEM prefetch is divergent), and a silent "
          "drop would mask both the cross-target capability gap "
          "and any pipeline-tuning regression downstream.");
      return Hr;
    }

    bool IsSaddr = false;
    if (Op.nSrcs() == 4) {
      IsSaddr = true;
    } else if (Op.nSrcs() != 3) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "FLAT",
          Twine(Di.Mnemonic)
              + ": expected 3 srcs (plain) or 4 srcs (SADDR) per "
                "FLAT_Prefetch_Pseudo");
      return Hr;
    }

    Value *PrefetchAddr = nullptr;
    unsigned ImmStart = 0;
    if (IsSaddr) {
      ParsedReg SaddrPr = Op.srcReg(0);
      ParsedReg VaddrPr = Op.srcReg(1);
      if (SaddrPr.RegKind != ParsedReg::SGPR ||
          VaddrPr.RegKind != ParsedReg::VGPR) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "FLAT",
            Twine(Di.Mnemonic)
                + " SADDR: expected (SGPR_64, VGPR_32) for (saddr, vaddr)");
        return Hr;
      }
      Value *Saddr = Ctx.Regs.readReg64(Ctx.B, SaddrPr);
      Value *Voff = Ctx.B.CreateZExt(
          Ctx.Regs.readReg32(Ctx.B, VaddrPr), Ctx.I64Ty, "voff_zext");
      PrefetchAddr = Ctx.B.CreateAdd(Saddr, Voff, "saddr_vaddr");
      ImmStart = 2;
    } else {
      ParsedReg VaddrPr = Op.srcReg(0);
      if (VaddrPr.RegKind != ParsedReg::VGPR) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "FLAT",
            Twine(Di.Mnemonic) + " plain: expected VGPR_64 for vaddr");
        return Hr;
      }
      PrefetchAddr = Ctx.Regs.readReg64(Ctx.B, VaddrPr);
      ImmStart = 1;
    }

    int64_t FlatOffset = 0;
    int64_t CpolImm = 0;
    bool SawOffset = false;
    for (unsigned K = ImmStart; K < Op.nSrcs(); ++K) {
      if (!Di.isImm(Op.srcIdx(K))) continue;
      int64_t V = Di.getImm(Op.srcIdx(K));
      if (!SawOffset) {
        FlatOffset = V;
        SawOffset = true;
      } else {
        CpolImm = V;
      }
    }

    Value *PrefetchPtr = PrefetchAddr;
    if (PrefetchPtr->getType() != PrefetchPtrTy)
      PrefetchPtr = Ctx.B.CreateIntToPtr(PrefetchPtr, PrefetchPtrTy);
    if (FlatOffset != 0)
      PrefetchPtr = Ctx.B.CreateGEP(Ctx.I8Ty, PrefetchPtr,
                                     Ctx.B.getInt64(FlatOffset),
                                     "prefetch_addr");

    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, PrefetchIntrinsic);
    Value *CpolArg = ConstantInt::get(Ctx.I32Ty, CpolImm);
    Ctx.B.CreateCall(Fn, {PrefetchPtr, CpolArg});

    Hr.Handled = true;
    return Hr;
  }

  // flat_load/flat_store -- same structure as global but uses flat address
  // space on the plain-VGPR64 form (gfx9/10/11) and global address space
  // on the gfx12+ SADDR form.  Detection is by operand shape, matching
  // `decodeGlobalLoadAddr`'s discriminator but with address-space
  // semantics preserved per case (the shared decoder unconditionally
  // casts to `ctx.ptrGlobalTy` which is addrspace(1); we route through
  // it only for the SADDR form where that cast is hardware-correct).
  if (Sop == CanonicalOp::FLAT_LOAD_USHORT || Sop == CanonicalOp::FLAT_LOAD_SSHORT ||
      Sop == CanonicalOp::FLAT_LOAD_UBYTE || Sop == CanonicalOp::FLAT_LOAD_SBYTE) {
    ParsedReg Dest = Op.dst();
    bool IsByte = Sop == CanonicalOp::FLAT_LOAD_UBYTE || Sop == CanonicalOp::FLAT_LOAD_SBYTE;
    Value *Addr = nullptr;
    // SADDR form: saddr(SGPR64), vaddr(VGPR32), [scale_offset] [offset:imm]
    // -- semantically a global_load (SGPR base + per-lane VGPR offset),
    // so delegate to the shared decoder and accept its addrspace(1)
    // conversion.
    if (Op.nSrcs() >= 2 && Op.isSrcReg(0) && Op.isSrcReg(1) &&
        Op.srcReg(0).RegKind == ParsedReg::SGPR &&
        Op.srcReg(1).RegKind == ParsedReg::VGPR) {
      FlatAddr Fa = decodeGlobalLoadAddr(Ctx, Di, Op, IsByte ? 1 : 2,
                                          "FLAT_LOAD sub-dword (SADDR)");
      Addr = Fa.Ptr;
    } else {
      // Plain-flat form: VGPR64 holds the full per-lane flat address.
      // Preserve addrspace(0) so the backend re-emits `flat_load_*` on
      // targets where plain-flat may legitimately reach LDS or private
      // (gfx9/10/11 AMDGPU lowering keys off the AS to choose between
      // flat/global/ds/scratch load classes).
      Addr = Ctx.Regs.readReg64(Ctx.B, Op.srcReg(0));
      Type *PtrFlatTy = PointerType::get(Ctx.C, 0);
      if (Addr->getType() != PtrFlatTy)
        Addr = Ctx.B.CreateIntToPtr(Addr, PtrFlatTy);
      int64_t MemOffset = 0;
      for (unsigned K = 1; K < Op.nSrcs(); K++)
        if (Di.isImm(Op.srcIdx(K)) && Di.getImm(Op.srcIdx(K)) != 0)
          MemOffset = Di.getImm(Op.srcIdx(K));
      if (MemOffset != 0)
        Addr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Addr,
                                        Ctx.B.getInt64(MemOffset));
    }
    Type *LoadTy = IsByte ? Ctx.I8Ty : Type::getInt16Ty(Ctx.C);
    // SPE-gate the memory access itself. Same rationale as the
    // GLOBAL_LOAD sub-dword block above: unguarded loads fault on
    // WaveNative phantom lanes whose pointer VGPRs hold `undef` or
    // stale data. FLAT (plain-VGPR64) loads can legitimately reach
    // LDS / private / global -- an out-of-range phantom-lane pointer
    // will still page-fault in whichever aperture it lands in.
    Ctx.emitUnderExec([&] {
      Value *Loaded = Ctx.B.CreateLoad(LoadTy, Addr, "flat_load_sub");
      bool IsUnsigned = Sop == CanonicalOp::FLAT_LOAD_UBYTE ||
                        Sop == CanonicalOp::FLAT_LOAD_USHORT;
      Value *Ext = IsUnsigned ? Ctx.B.CreateZExt(Loaded, Ctx.I32Ty)
                              : Ctx.B.CreateSExt(Loaded, Ctx.I32Ty);
      Ctx.Regs.writeReg32(Ctx.B, Dest, Ext);
    });
    Hr.Handled = true;
    return Hr;
  }

  if (Sop == CanonicalOp::FLAT_LOAD_DWORD || Sop == CanonicalOp::FLAT_LOAD_DWORDX2 ||
      Sop == CanonicalOp::FLAT_LOAD_DWORDX3 || Sop == CanonicalOp::FLAT_LOAD_DWORDX4) {
    int LoadDwords = 1;
    if (Sop == CanonicalOp::FLAT_LOAD_DWORDX2) LoadDwords = 2;
    else if (Sop == CanonicalOp::FLAT_LOAD_DWORDX4) LoadDwords = 4;
    else if (Sop == CanonicalOp::FLAT_LOAD_DWORDX3) LoadDwords = 3;

    // Two operand-shape variants, with distinct address-space semantics:
    //
    //   (plain)  vaddr:VGPR64, [imms]
    //              -- gfx9/10/11 `flat_load_dword`: VGPR64 holds a full
    //              per-lane flat address that may legitimately reach
    //              LDS or private; lift as addrspace(0) so the target
    //              backend can re-emit `flat_load_*`.
    //
    //   (SADDR)  saddr:SGPR64, vaddr:VGPR32, [scale_offset] [offset:imm]
    //              -- gfx12+ `flat_load_b32 … scale_offset`: uniform
    //              SGPR64 base + per-lane VGPR32 offset, semantically
    //              identical to `global_load_dword`'s SADDR form (the
    //              hardware's scale-offset + signed-imm arithmetic can
    //              only reach globally-allocated memory, which the
    //              compiler encodes by choosing this variant).  Lift
    //              as addrspace(1) via the shared `decodeGlobalLoadAddr`
    //              helper so the target backend re-emits
    //              `global_load_*` with matching scale-offset arithmetic.
    //
    // Before the SADDR arm was added, the handler took `op.srcReg(0)`
    // as a 64-bit pointer unconditionally.  On the SADDR form that
    // silently picked up the saddr SGPR pair as the full address and
    // DROPPED the per-lane VGPR offset, producing a kernel where every
    // lane addresses the same memory location (observable on the
    // `rcp_sqrt_kernel` gfx1250 fixture: 256 output lanes all reading
    // `in[0x206/4] = in[518]` regardless of tid).
    // The shape discriminator below (`op.srcReg(0).kind == SGPR` AND
    // `op.srcReg(1).kind == VGPR`) matches `decodeGlobalLoadAddr`'s
    // inner predicate but keeps the plain-form AS choice local -- the
    // shared helper unconditionally casts its result to addrspace(1),
    // which is correct for SADDR and wrong for plain-flat on pre-gfx12.
    Value *Addr = nullptr;
    if (Op.nSrcs() >= 2 && Op.isSrcReg(0) && Op.isSrcReg(1) &&
        Op.srcReg(0).RegKind == ParsedReg::SGPR &&
        Op.srcReg(1).RegKind == ParsedReg::VGPR) {
      FlatAddr Fa = decodeGlobalLoadAddr(Ctx, Di, Op, LoadDwords * 4,
                                          "FLAT_LOAD dword (SADDR)");
      Addr = Fa.Ptr;
    } else {
      Addr = Ctx.Regs.readReg64(Ctx.B, Op.srcReg(0));
      Type *PtrFlatTy = PointerType::get(Ctx.C, 0);
      if (Addr->getType() != PtrFlatTy)
        Addr = Ctx.B.CreateIntToPtr(Addr, PtrFlatTy);
      int64_t MemOffset = 0;
      for (unsigned K = 1; K < Op.nSrcs(); K++)
        if (Di.isImm(Op.srcIdx(K)) && Di.getImm(Op.srcIdx(K)) != 0)
          MemOffset = Di.getImm(Op.srcIdx(K));
      if (MemOffset != 0)
        Addr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Addr,
                                        Ctx.B.getInt64(MemOffset));
    }

    ParsedReg Dest = Op.dst();
    // SPE-gate the memory access itself (same rationale as
    // GLOBAL_LOAD dword above).  For DWORDX{2,3,4} the single vector
    // load + N element-extract writes all live inside one
    // `emitUnderExec` block so inactive lanes skip the whole
    // sequence and the store phase's VGPR consumers observe the
    // alloca state from the most recent active write.
    Ctx.emitUnderExec([&] {
      if (LoadDwords == 1) {
        Ctx.Regs.writeReg32(
            Ctx.B, Dest,
            Ctx.B.CreateBitCast(
                Ctx.B.CreateLoad(Ctx.F32Ty, Addr, "flat_load"),
                Ctx.I32Ty));
      } else {
        Type *VecTy = FixedVectorType::get(Ctx.I32Ty, LoadDwords);
        Value *Loaded = Ctx.B.CreateLoad(VecTy, Addr, "flat_load");
        for (int D = 0; D < LoadDwords; D++) {
          ParsedReg Sub = Dest;
          Sub.BaseIdx = Dest.BaseIdx + D;
          Sub.Width = 1;
          Ctx.Regs.writeReg32(
              Ctx.B, Sub,
              Ctx.B.CreateExtractElement(Loaded, Ctx.B.getInt32(D)));
        }
      }
    });
    Hr.Handled = true;
    return Hr;
  }

  if (Sop == CanonicalOp::FLAT_STORE_DWORD || Sop == CanonicalOp::FLAT_STORE_DWORDX2 ||
      Sop == CanonicalOp::FLAT_STORE_DWORDX3 || Sop == CanonicalOp::FLAT_STORE_DWORDX4 ||
      Sop == CanonicalOp::FLAT_STORE_BYTE || Sop == CanonicalOp::FLAT_STORE_SHORT ||
      Sop == CanonicalOp::FLAT_STORE_SHORT_D16_HI) {
    int StoreDwords = 1;
    int StoreBits = 32;
    // `FLAT_STORE_SHORT_D16_HI` stores bits [31:16] of the source
    // VGPR, not [15:0] -- same half-register selector as
    // `GLOBAL_STORE_SHORT_D16_HI` above; see the comment block on
    // the GLOBAL_STORE_ branch for the full rationale (bf16 RNE
    // epilogue, pre-fix miscompile shape, etc.).
    bool StoreHiHalf = false;
    if (Sop == CanonicalOp::FLAT_STORE_DWORDX4) StoreDwords = 4;
    else if (Sop == CanonicalOp::FLAT_STORE_DWORDX3) StoreDwords = 3;
    else if (Sop == CanonicalOp::FLAT_STORE_DWORDX2) StoreDwords = 2;
    else if (Sop == CanonicalOp::FLAT_STORE_DWORD) StoreDwords = 1;
    else if (Sop == CanonicalOp::FLAT_STORE_SHORT ||
             Sop == CanonicalOp::FLAT_STORE_SHORT_D16_HI) {
      StoreBits = 16; StoreDwords = 0;
      StoreHiHalf = (Sop == CanonicalOp::FLAT_STORE_SHORT_D16_HI);
    }
    else if (Sop == CanonicalOp::FLAT_STORE_BYTE) { StoreBits = 8; StoreDwords = 0; }

    // Two operand-shape variants with distinct AS semantics; mirror
    // the FLAT_LOAD_DWORD handler's case split.  For stores:
    //
    //   (plain)  vaddr:VGPR64, vdata:VGPR*, [imms]            -> addrspace(0)
    //   (SADDR)  vaddr:VGPR32, vdata:VGPR*, saddr:SGPR64, ... -> addrspace(1)
    //
    // See the FLAT_LOAD_DWORD comment block above for the full
    // derivation and rcp_sqrt_kernel regression anchor.
    int ElemBytes = StoreBits < 32 ? (StoreBits / 8)
                                    : std::max(StoreDwords, 1) * 4;
    Value *Addr = nullptr;
    ParsedReg StData;
    if (Op.nSrcs() >= 3 && Op.isSrcReg(0) && Op.isSrcReg(1) &&
        Op.isSrcReg(2) &&
        Op.srcReg(0).RegKind == ParsedReg::VGPR &&
        Op.srcReg(1).RegKind == ParsedReg::VGPR &&
        Op.srcReg(2).RegKind == ParsedReg::SGPR) {
      FlatAddr Fa = decodeGlobalStoreAddr(Ctx, Di, Op, ElemBytes,
                                           "FLAT_STORE (SADDR)");
      Addr = Fa.Ptr;
      StData = Fa.StData;
    } else {
      Addr = Ctx.Regs.readReg64(Ctx.B, Op.srcReg(0));
      Type *PtrFlatTy = PointerType::get(Ctx.C, 0);
      if (Addr->getType() != PtrFlatTy)
        Addr = Ctx.B.CreateIntToPtr(Addr, PtrFlatTy);
      int64_t MemOffset = 0;
      for (unsigned K = 2; K < Op.nSrcs(); K++)
        if (Di.isImm(Op.srcIdx(K)) && Di.getImm(Op.srcIdx(K)) != 0)
          MemOffset = Di.getImm(Op.srcIdx(K));
      if (MemOffset != 0)
        Addr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Addr,
                                        Ctx.B.getInt64(MemOffset));
      StData = Op.srcReg(1);
    }
    if (StoreDwords == 0) {
      Value *Src32 = Ctx.Regs.readReg32(Ctx.B, StData);
      // See emitD16HiHalfTruncI16's doc block for the shared-helper
      // rationale; this branch mirrors the GLOBAL_STORE path above
      // so both FLAT and GLOBAL `_D16_HI` variants graduate through
      // the same emission shape.
      Value *Val = StoreHiHalf
                      ? emitD16HiHalfTruncI16(Ctx, Src32)
                      : Ctx.B.CreateTrunc(
                            Src32, Type::getIntNTy(Ctx.C, StoreBits));
      Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Addr); });
    } else if (StoreDwords == 1) {
      Value *Val = Ctx.Regs.readReg32(Ctx.B, StData);
      Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Addr); });
    } else {
      auto *VecTy = FixedVectorType::get(Ctx.I32Ty, StoreDwords);
      Value *Val = Ctx.Regs.readRegVec(Ctx.B, StData, VecTy);
      Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Addr); });
    }
    Hr.Handled = true;
    return Hr;
  }

  // flat_atomic_* -- same as global_atomic but flat address space
  if (Sop >= CanonicalOp::FLAT_ATOMIC_ADD && Sop <= CanonicalOp::FLAT_ATOMIC_ADD_F32) {
    // Contract: the RTN/non-RTN collapse in OpcodeMap relies on
    // IsAtomicRet <=> (numDefs > 0) to decide result writeback below.
    assert(((Di.TsFlags & SIInstrFlags::IsAtomicRet) != 0) == (Di.NumDefs > 0) &&
           "flat atomic: IsAtomicRet disagrees with numDefs");
    // Two operand-shape variants, mirroring the FLAT_LOAD/STORE case
    // split and the GLOBAL_ATOMIC block below:
    //
    //   (plain)  vaddr:VGPR64, vdata:VGPR*, [imms] -> addrspace(0)
    //              gfx9/10/11 `flat_atomic_add v[A:B], vData, off[,offset]`:
    //              VGPR64 holds a full per-lane flat address that may
    //              legitimately reach LDS/private/global; preserve
    //              addrspace(0) so the backend re-emits `flat_atomic_*`
    //              on targets where plain-flat is still valid.
    //
    //   (SADDR)  vaddr:VGPR32, vdata:VGPR*, saddr:SGPR64,
    //            [scale_offset] [offset:imm] -> addrspace(1)
    //              gfx12+ `flat_atomic_* vAddr, vData, s[A:B]
    //              [scale_offset]`: scale-offset + signed-imm
    //              arithmetic can only reach globally-allocated memory
    //              (same semantic-narrowing as FLAT_LOAD/STORE SADDR),
    //              so delegate to `decodeGlobalStoreAddr` which casts
    //              to `ctx.ptrGlobalTy` (addrspace(1)) and lets the
    //              backend re-emit `global_atomic_*` with matching
    //              scale-offset arithmetic.
    //
    // Before this split, the handler hard-coded the plain shape and
    // used a "first non-zero imm wins" scan that misidentified the
    // CPol operand (packed scale_offset + scope bits) as the signed
    // offset field.  The bug is DORMANT today -- no recipe in the
    // compare_correctness Triton corpus emits gfx12+ `flat_atomic_*`
    // (Triton's gfx1250 codegen prefers `global_atomic_*` whenever it
    // knows the buffer lives in global) -- but it's the identical
    // class that bit GLOBAL_ATOMIC (see that block's comment for the
    // `sum_bitmatrix_rows_u32` failure and the
    // sum-bitmatrix-rows regression gate).  Fixed here
    // for symmetry to close the "handler assumes operand shape that
    // varies by subtarget" bug class across FLAT_LOAD / FLAT_STORE /
    // FLAT_ATOMIC / GLOBAL_LOAD / GLOBAL_STORE / GLOBAL_ATOMIC -- all
    // six now route SADDR through the shared decoder.
    Value *Addr = nullptr;
    ParsedReg StData;
    const bool IsSaddr = Op.nSrcs() >= 3 && Op.isSrcReg(0) &&
                         Op.isSrcReg(1) && Op.isSrcReg(2) &&
                         Op.srcReg(0).RegKind == ParsedReg::VGPR &&
                         Op.srcReg(1).RegKind == ParsedReg::VGPR &&
                         Op.srcReg(2).RegKind == ParsedReg::SGPR;
    if (IsSaddr) {
      FlatAddr Fa = decodeGlobalStoreAddr(Ctx, Di, Op, /*elemBytes=*/4,
                                           "FLAT_ATOMIC (SADDR)");
      Addr = Fa.Ptr;
      StData = Fa.StData;
    } else {
      ParsedReg AddrReg = Op.srcReg(0);
      Addr = Ctx.Regs.readReg64(Ctx.B, AddrReg);
      Type *PtrFlatTy = PointerType::get(Ctx.C, 0);
      if (Addr->getType() != PtrFlatTy) Addr = Ctx.B.CreateIntToPtr(Addr, PtrFlatTy);
      int64_t MemOffset = 0;
      unsigned DataIdx = 1;
      for (unsigned K = 1; K < Op.nSrcs(); K++) {
        if (Di.isImm(Op.srcIdx(K)) && Di.getImm(Op.srcIdx(K)) != 0)
          MemOffset = Di.getImm(Op.srcIdx(K));
        else if (Di.isReg(Op.srcIdx(K)))
          DataIdx = K;
      }
      if (MemOffset != 0) Addr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Addr, Ctx.B.getInt64(MemOffset));
      StData = Op.srcReg(DataIdx);
    }
    Value *Data = Ctx.Regs.readReg32(Ctx.B, StData);

    if (Sop == CanonicalOp::FLAT_ATOMIC_CMPSWAP) {
      // CMPSWAP's vdata is a 2-vgpr pair (cmp, new); read low half as
      // cmpVal and the adjacent baseIdx+1 as newVal.
      Value *CmpVal = Data;
      ParsedReg NewReg = StData; NewReg.BaseIdx += 1; NewReg.Width = 1;
      Value *NewVal = Ctx.Regs.readReg32(Ctx.B, NewReg);
      Ctx.emitUnderExec([&] {
        auto *Cas = Ctx.B.CreateAtomicCmpXchg(
            Addr, CmpVal, NewVal, MaybeAlign(),
            AtomicOrdering::SequentiallyConsistent,
            AtomicOrdering::SequentiallyConsistent);
        if (Di.NumDefs > 0)
          Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                              Ctx.B.CreateExtractValue(Cas, 0));
      });
      Hr.Handled = true;
    return Hr;
    }

    AtomicRMWInst::BinOp AtomicOp;
    Type *AtomicTy = Ctx.I32Ty;
    bool IsFp = false;
    switch (Sop) {
    case CanonicalOp::FLAT_ATOMIC_ADD:  AtomicOp = AtomicRMWInst::Add; break;
    case CanonicalOp::FLAT_ATOMIC_SUB:  AtomicOp = AtomicRMWInst::Sub; break;
    case CanonicalOp::FLAT_ATOMIC_AND:  AtomicOp = AtomicRMWInst::And; break;
    case CanonicalOp::FLAT_ATOMIC_OR:   AtomicOp = AtomicRMWInst::Or; break;
    case CanonicalOp::FLAT_ATOMIC_XOR:  AtomicOp = AtomicRMWInst::Xor; break;
    case CanonicalOp::FLAT_ATOMIC_SMIN: AtomicOp = AtomicRMWInst::Min; break;
    case CanonicalOp::FLAT_ATOMIC_SMAX: AtomicOp = AtomicRMWInst::Max; break;
    case CanonicalOp::FLAT_ATOMIC_UMIN: AtomicOp = AtomicRMWInst::UMin; break;
    case CanonicalOp::FLAT_ATOMIC_UMAX: AtomicOp = AtomicRMWInst::UMax; break;
    case CanonicalOp::FLAT_ATOMIC_SWAP: AtomicOp = AtomicRMWInst::Xchg; break;
    case CanonicalOp::FLAT_ATOMIC_ADD_F32:
      AtomicOp = AtomicRMWInst::FAdd; IsFp = true;
      Data = Ctx.B.CreateBitCast(Data, Ctx.F32Ty); AtomicTy = Ctx.F32Ty; break;
    default:
      llvm::errs() << "transpiler: Unhandled flat atomic: " << Mn << "\n";
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(Di, "FLAT",
                                                   "unhandled flat atomic");
      return Hr;
    }
    Ctx.emitUnderExec([&] {
      auto *Rmw = Ctx.B.CreateAtomicRMW(
          AtomicOp, Addr, Data, MaybeAlign(),
          AtomicOrdering::SequentiallyConsistent);
      if (Di.NumDefs > 0) {
        Value *RetVal = Rmw;
        if (IsFp) RetVal = Ctx.B.CreateBitCast(RetVal, Ctx.I32Ty);
        Ctx.Regs.writeReg32(Ctx.B, Op.dst(), RetVal);
      }
    });
    Hr.Handled = true;
    return Hr;
  }

  // ---- Global atomics ----
  if (Sop >= CanonicalOp::GLOBAL_ATOMIC_ADD && Sop <= CanonicalOp::GLOBAL_ATOMIC_PK_ADD_F16) {
    assert(((Di.TsFlags & SIInstrFlags::IsAtomicRet) != 0) == (Di.NumDefs > 0) &&
           "global atomic: IsAtomicRet disagrees with numDefs");
    // Delegate addressing to `decodeGlobalStoreAddr`.  Global atomics
    // share the store's operand shape -- (vaddr, vdata, [saddr], [imms]) --
    // and we reuse the exact same decoder to get both shape variants
    // handled consistently:
    //
    //   (plain)  vaddr:VGPR64, vdata:VGPR*, [imms]
    //   (SADDR)  vaddr:VGPR32, vdata:VGPR*, saddr:SGPR64,
    //            [scale_offset] [offset:imm]
    //
    // Before this switch, the handler hard-coded the plain shape
    // (`readReg64(srcReg(0))`) AND used a heuristic "first non-zero imm
    // wins" scan to pick up the signed offset.  Two bugs compounded on
    // `_sum_bitmatrix_rows`'s
    //   global_atomic_add_u32 v1, v0, s[4:5] scale_offset scope:SCOPE_DEV
    // -- the SADDR form:
    //
    //   (1) `op.srcReg(0)` is VGPR32 `v1` (the vaddr offset), not a
    //       pointer; reading it as a VGPR64 pair pulled in v2 as the
    //       high half and produced garbage.
    //   (2) After the saddr SGPR, the CPol operand carries the
    //       `scale_offset` bit + scope bits packed into a non-zero
    //       immediate (0x810 = 2064 for `scale_offset scope:SCOPE_DEV`);
    //       "first non-zero imm wins" mistook that for the signed
    //       offset field and added a 2064-byte GEP before the atomic,
    //       far past the 32-element `Out` buffer.
    //   (3) The same loop's "last reg-valued src = data" heuristic
    //       then selected the SGPR saddr (s[4:5] = Out pointer) as the
    //       data to atomic-add, instead of the VGPR vdata (v0 = the
    //       popcount sum).
    //
    // Net device-visible symptom: `HIP error 700 (illegal memory
    // access)` on every launch of `sum_bitmatrix_rows_u32` (and the
    // `_nw4` variant) -- the atomic fired at the wrong address with
    // the wrong value.  `decodeGlobalStoreAddr` handles the shape
    // discriminator and uses `firstImmOffset` (FIRST imm, regardless
    // of value) for the offset field, so CPol no longer leaks into
    // the offset lookup.  See `hotswap/docs/learnings.md` entry
    // "2026-04-23 -- global_atomic SADDR form silently miscompiled"
    // for the full investigation and the regression gate.
    //
    // Element size for `scale_offset`: every atomic in the CanonicalOp range
    // [GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_PK_ADD_F16] operates on a
    // 32-bit memory slot (the 64-bit `_X2` variants are outside this
    // range), so elemBytes=4 is correct for all of
    //   ADD/SUB/AND/OR/XOR/MIN/MAX/SWAP/ADD_F32/PK_ADD_{BF16,F16}/CMPSWAP
    // When `hasScaleOffset` is false the multiply is elided by the
    // decoder.
    FlatAddr Fa = decodeGlobalStoreAddr(Ctx, Di, Op, /*elemBytes=*/4,
                                         "GLOBAL_ATOMIC");
    Value *Addr = Fa.Ptr;
    // `stData` points to the base of the vdata VGPR range (32-bit for
    // the scalar atomics; CMPSWAP treats it as a 2-vgpr pair
    // cmp/new -- see the CMPSWAP branch below).
    Value *Data = Ctx.Regs.readReg32(Ctx.B, Fa.StData);

    if (Sop == CanonicalOp::GLOBAL_ATOMIC_CMPSWAP) {
      // CMPSWAP's vdata is declared as VReg_64 in the .td (a 2-vgpr
      // pair (cmpVal, newVal) in that order).  `fa.stData` is the
      // base of that pair; increment baseIdx to reach newVal.
      Value *CmpVal = Data;
      ParsedReg NewReg = Fa.StData; NewReg.BaseIdx += 1; NewReg.Width = 1;
      Value *NewVal = Ctx.Regs.readReg32(Ctx.B, NewReg);
      Ctx.emitUnderExec([&] {
        auto *Cas = Ctx.B.CreateAtomicCmpXchg(
            Addr, CmpVal, NewVal, MaybeAlign(), AtomicOrdering::Monotonic,
            AtomicOrdering::Monotonic);
        if (Di.NumDefs > 0)
          Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                              Ctx.B.CreateExtractValue(Cas, 0));
      });
      Hr.Handled = true;
    return Hr;
    }

    AtomicRMWInst::BinOp AtomicOp;
    Type *AtomicTy = Ctx.I32Ty;
    bool IsFp = false;
    switch (Sop) {
    case CanonicalOp::GLOBAL_ATOMIC_ADD:  AtomicOp = AtomicRMWInst::Add; break;
    case CanonicalOp::GLOBAL_ATOMIC_SUB:  AtomicOp = AtomicRMWInst::Sub; break;
    case CanonicalOp::GLOBAL_ATOMIC_AND:  AtomicOp = AtomicRMWInst::And; break;
    case CanonicalOp::GLOBAL_ATOMIC_OR:   AtomicOp = AtomicRMWInst::Or; break;
    case CanonicalOp::GLOBAL_ATOMIC_XOR:  AtomicOp = AtomicRMWInst::Xor; break;
    case CanonicalOp::GLOBAL_ATOMIC_SMIN: AtomicOp = AtomicRMWInst::Min; break;
    case CanonicalOp::GLOBAL_ATOMIC_SMAX: AtomicOp = AtomicRMWInst::Max; break;
    case CanonicalOp::GLOBAL_ATOMIC_UMIN: AtomicOp = AtomicRMWInst::UMin; break;
    case CanonicalOp::GLOBAL_ATOMIC_UMAX: AtomicOp = AtomicRMWInst::UMax; break;
    case CanonicalOp::GLOBAL_ATOMIC_SWAP: AtomicOp = AtomicRMWInst::Xchg; break;
    case CanonicalOp::GLOBAL_ATOMIC_ADD_F32:
      AtomicOp = AtomicRMWInst::FAdd; AtomicTy = Ctx.F32Ty; IsFp = true; break;
    case CanonicalOp::GLOBAL_ATOMIC_PK_ADD_BF16:
      AtomicOp = AtomicRMWInst::FAdd;
      AtomicTy = FixedVectorType::get(Type::getBFloatTy(Ctx.C), 2); IsFp = true; break;
    case CanonicalOp::GLOBAL_ATOMIC_PK_ADD_F16:
      AtomicOp = AtomicRMWInst::FAdd;
      AtomicTy = FixedVectorType::get(Type::getHalfTy(Ctx.C), 2); IsFp = true; break;
    default:
      llvm::errs() << "transpiler: Unsupported global atomic variant: " << Mn << "\n";
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "FLAT", "unsupported global atomic variant");
      return Hr;
    }
    if (IsFp) Data = Ctx.B.CreateBitCast(Data, AtomicTy);
    Ctx.emitUnderExec([&] {
      Value *Prev = Ctx.B.CreateAtomicRMW(AtomicOp, Addr, Data, MaybeAlign(),
                                          AtomicOrdering::Monotonic);
      if (Di.NumDefs > 0) {
        if (IsFp) Prev = Ctx.B.CreateBitCast(Prev, Ctx.I32Ty);
        Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Prev);
      }
    });
    Hr.Handled = true;
    return Hr;
  }
  return Hr;
}

} // namespace COMGR::hotswap
