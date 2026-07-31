//===- handle-smem.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"
#include "source-image-address.h"

#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <optional>

#define DEBUG_TYPE "transpiler"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

constexpr uint64_t kSourceDwordBytes = sizeof(uint32_t);

// Add the decoded static SMEM byte immediate to an already 64-bit dynamic
// offset, preserving `Offset` when the instruction has no non-zero immediate.
Value *addStaticSmemByteOffset64(RaiseContext &Ctx, const DecodedInst &Di,
                                 Value *Offset, StringRef Name) {
  if (!Di.StaticOffset || *Di.StaticOffset == 0)
    return Offset;
  return Ctx.B.CreateAdd(Offset, Ctx.B.getInt64(*Di.StaticOffset), Name);
}

// Match scalar buffer-resource loads as a single handler family.
bool isScalarBufferLoad(CanonicalOp Sop) {
  switch (Sop) {
  case CanonicalOp::S_BUFFER_LOAD_B32:
  case CanonicalOp::S_BUFFER_LOAD_B64:
  case CanonicalOp::S_BUFFER_LOAD_B96:
  case CanonicalOp::S_BUFFER_LOAD_B128:
  case CanonicalOp::S_BUFFER_LOAD_B256:
  case CanonicalOp::S_BUFFER_LOAD_B512:
    return true;
  default:
    return false;
  }
}

// Return the TableGen destination register-class width for an explicit def.
unsigned defRegClassDwordCount(RaiseContext &Ctx, const DecodedInst &Di,
                               unsigned DefIdx) {
  const MCInstrInfo &MII = *Ctx.Mc.InstrInfo;
  const MCRegisterInfo &MRI = *Ctx.Mc.RegInfo;
  const MCSubtargetInfo &STI = *Ctx.Mc.SubtargetInfo;
  const MCInstrDesc &Desc = MII.get(Di.Inst.getOpcode());
  ArrayRef<MCOperandInfo> Operands = Desc.operands();
  assert(DefIdx < Operands.size() && "missing SMEM def operand metadata");

  int16_t RegClassId = MII.getOpRegClassID(
      Operands[DefIdx], STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo));
  assert(RegClassId >= 0 && "SMEM def operand must have a register class");

  unsigned Bits = MRI.getRegClass(RegClassId).getSizeInBits();
  assert(Bits != 0 && Bits % 32 == 0 &&
         "SMEM def register class must have dword width");
  return Bits / 32;
}

// Build descriptor arithmetic in a common 64-bit integer type.
Value *zextToI64(RaiseContext &Ctx, Value *V, const Twine &Name = "") {
  return Ctx.B.CreateZExt(V, Ctx.I64Ty, Name);
}

// Sign-extend a low-bit address field after extracting it from a descriptor.
Value *signExtendLowBitsI64(RaiseContext &Ctx, Value *V, unsigned Bits,
                            const Twine &Name) {
  assert(Bits > 0 && Bits < 64 && "expected sign extension from sub-i64 width");
  unsigned Shift = 64 - Bits;
  Value *Shifted =
      Ctx.B.CreateShl(V, ConstantInt::get(Ctx.I64Ty, Shift), Name + ".shl");
  return Ctx.B.CreateAShr(Shifted, ConstantInt::get(Ctx.I64Ty, Shift), Name);
}

// Dword scalar buffer loads ignore the low two address bits.
Value *alignDwordAddress64(RaiseContext &Ctx, Value *Addr, const Twine &Name) {
  return Ctx.B.CreateAnd(Addr, Ctx.B.getInt64(~uint64_t(3)), Name);
}

// Dword scalar buffer loads ignore the low two offset bits.
Value *alignDwordOffset32(RaiseContext &Ctx, Value *Offset, const Twine &Name) {
  return Ctx.B.CreateAnd(Offset, ConstantInt::get(Ctx.I32Ty, ~uint32_t(3)),
                         Name);
}

// Read a little-endian dword from `.text` using a text-relative byte offset.
// Kept as the fallback for older callers that only carried the text image.
uint32_t readSourceTextDword(RaiseContext &Ctx, uint64_t ByteOffset) {
  assert(ByteOffset <= Ctx.SourceTextBytes.size() &&
         kSourceDwordBytes <= Ctx.SourceTextBytes.size() - ByteOffset &&
         "source text dword read must be bounds-checked first");
  return support::endian::read32le(Ctx.SourceTextBytes.data() + ByteOffset);
}

// Read a dword from the captured source code-object image by source address.
// PC-relative literal tables in clang/rocPRIM live in allocatable sections
// such as .rodata, not necessarily in .text.
std::optional<uint32_t> readSourceImageDword(RaiseContext &Ctx,
                                             uint64_t Address) {
  for (const TextSection::ImageSection &Section : Ctx.SourceImageSections) {
    if (Address < Section.Address)
      continue;
    uint64_t Offset = Address - Section.Address;
    uint64_t SectionSize = static_cast<uint64_t>(Section.Bytes.size());
    if (Offset > SectionSize || kSourceDwordBytes > SectionSize - Offset)
      continue;
    return support::endian::read32le(Section.Bytes.data() + Offset);
  }
  if (Address >= Ctx.SourceTextBaseAddress) {
    uint64_t Offset = Address - Ctx.SourceTextBaseAddress;
    uint64_t TextSize = static_cast<uint64_t>(Ctx.SourceTextBytes.size());
    if (Offset <= TextSize && kSourceDwordBytes <= TextSize - Offset)
      return readSourceTextDword(Ctx, Offset);
  }
  return std::nullopt;
}

// Emit a branch to llvm.trap when a dynamic translation contract is violated.
void emitTrapUnless(RaiseContext &Ctx, Value *Condition,
                    const Twine &ReasonName) {
  Function *Trap = Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::trap);
  BasicBlock *CurBB = Ctx.B.GetInsertBlock();
  Function *F = CurBB->getParent();
  BasicBlock *TrapBB = BasicBlock::Create(Ctx.C, ReasonName + ".trap", F);
  BasicBlock *ContBB = BasicBlock::Create(Ctx.C, ReasonName + ".cont", F);

  Ctx.B.CreateCondBr(Condition, ContBB, TrapBB);
  Ctx.B.SetInsertPoint(TrapBB);
  Ctx.B.CreateCall(Trap);
  Ctx.B.CreateUnreachable();
  Ctx.B.SetInsertPoint(ContBB);
}

// Check that the source buffer-resource base fits in the target descriptor.
void emitBufferBaseRepresentabilityGuard(RaiseContext &Ctx, Value *BaseAddr) {
  if (Ctx.TargetIsa.BufferResourceBaseBits >= Ctx.Isa.BufferResourceBaseBits)
    return;

  unsigned TargetBits = Ctx.TargetIsa.BufferResourceBaseBits;
  if (TargetBits >= 64)
    return;

  auto *TargetBaseTy = IntegerType::get(Ctx.C, TargetBits);
  Value *Narrow =
      Ctx.B.CreateTrunc(BaseAddr, TargetBaseTy, "sbuf_base_target_bits");
  Value *RoundTrip =
      Ctx.B.CreateSExt(Narrow, Ctx.I64Ty, "sbuf_base_target_sext");
  Value *Representable =
      Ctx.B.CreateICmpEQ(RoundTrip, BaseAddr, "sbuf_base_target_ok");
  // A gfx12 descriptor can carry a wider base than a gfx942 descriptor.
  // The runtime normally constructs descriptors from target-valid process
  // pointers; this guard makes that contract explicit for dynamic descriptors.
  emitTrapUnless(Ctx, Representable, "sbuf_base_unrepresentable");
}

// Source descriptor fields after projecting them to the target raw-buffer form.
struct SourceScalarBufferResource {
  Value *BasePtr = nullptr;
  Value *ExtentBytes = nullptr;
};

// Decode the source buffer-resource descriptor fields used by S_BUFFER_LOAD.
SourceScalarBufferResource decodeSourceScalarBufferResource(RaiseContext &Ctx,
                                                            ParsedReg Base) {
  Value *Dw0 = Ctx.Regs.loadSGPR32(Ctx.B, Base.BaseIdx);
  Value *Dw1 = Ctx.Regs.loadSGPR32(Ctx.B, Base.BaseIdx + 1);
  Value *Dw2 = Ctx.Regs.loadSGPR32(Ctx.B, Base.BaseIdx + 2);
  Value *Dw3 = Ctx.Regs.loadSGPR32(Ctx.B, Base.BaseIdx + 3);

  Value *BaseAddr = nullptr;
  Value *Stride = nullptr;
  Value *NumRecords = nullptr;

  if (Ctx.Isa.Has45BitNumRecordsBufferResource) {
    Value *Low64 =
        Ctx.B.CreateOr(zextToI64(Ctx, Dw0),
                       Ctx.B.CreateShl(zextToI64(Ctx, Dw1), Ctx.B.getInt64(32)),
                       "sbuf_rsrc_lo");
    Value *High64 =
        Ctx.B.CreateOr(zextToI64(Ctx, Dw2),
                       Ctx.B.CreateShl(zextToI64(Ctx, Dw3), Ctx.B.getInt64(32)),
                       "sbuf_rsrc_hi");

    // S_BUFFER_LOAD uses these gfx12+ buffer resource fields:
    //   base_address = resource[56:0]
    //   num_records  = resource[101:57]
    //   stride       = resource[121:108]
    // S_BUFFER_LOAD ignores the other descriptor bits, so they are not copied
    // into the target resource.
    constexpr uint64_t Base57Mask = (1ULL << 57) - 1;
    constexpr uint64_t NumRecordsHigh38Mask = (1ULL << 38) - 1;
    Value *Base57 =
        Ctx.B.CreateAnd(Low64, Ctx.B.getInt64(Base57Mask), "sbuf_base57");
    BaseAddr = signExtendLowBitsI64(Ctx, Base57, 57, "sbuf_base64");
    Value *NumLo =
        Ctx.B.CreateLShr(Low64, Ctx.B.getInt64(57), "sbuf_num_records_lo");
    Value *NumHi = Ctx.B.CreateAnd(High64, Ctx.B.getInt64(NumRecordsHigh38Mask),
                                   "sbuf_num_records_hi");
    NumRecords = Ctx.B.CreateOr(
        NumLo, Ctx.B.CreateShl(NumHi, Ctx.B.getInt64(7)), "sbuf_num_records");
    Value *Stride64 =
        Ctx.B.CreateAnd(Ctx.B.CreateLShr(High64, Ctx.B.getInt64(44)),
                        Ctx.B.getInt64(0x3fffull), "sbuf_stride64");
    Stride =
        Ctx.B.CreateTrunc(Stride64, Type::getInt16Ty(Ctx.C), "sbuf_stride");
  } else {
    Value *BaseLo = zextToI64(Ctx, Dw0);
    Value *BaseHi16 =
        Ctx.B.CreateAnd(zextToI64(Ctx, Dw1), Ctx.B.getInt64(0xffff));
    Value *Base48 = Ctx.B.CreateOr(
        BaseLo, Ctx.B.CreateShl(BaseHi16, Ctx.B.getInt64(32)), "sbuf_base48");
    BaseAddr = signExtendLowBitsI64(Ctx, Base48, 48, "sbuf_base64");
    Stride = Ctx.B.CreateTrunc(Ctx.B.CreateLShr(Dw1, Ctx.B.getInt32(16)),
                               Type::getInt16Ty(Ctx.C), "sbuf_stride");
    NumRecords = zextToI64(Ctx, Dw2, "sbuf_num_records");
  }

  // This handler covers only the dword-width S_BUFFER_LOAD_B* family. Source
  // hardware forces both the descriptor base and byte offsets to dword
  // alignment for these loads.
  BaseAddr = alignDwordAddress64(Ctx, BaseAddr, "sbuf_base_aligned");
  emitBufferBaseRepresentabilityGuard(Ctx, BaseAddr);

  Value *Stride64 = zextToI64(Ctx, Stride, "sbuf_stride_zext");
  Value *StrideIsZero =
      Ctx.B.CreateICmpEQ(Stride64, Ctx.B.getInt64(0), "sbuf_stride_zero");
  Value *SizeStride = Ctx.B.CreateSelect(StrideIsZero, Ctx.B.getInt64(1),
                                         Stride64, "sbuf_size_stride");
  // Source S_BUFFER_LOAD bounds use:
  //   m_size = (stride == 0 ? 1 : stride) * num_records
  // Raw pointer buffer loads take the extent in bytes, so rebuild the target
  // resource with that byte extent rather than passing source num_records
  // through directly.
  Value *ExtentBytes =
      Ctx.B.CreateMul(SizeStride, NumRecords, "sbuf_extent_bytes");

  return {Ctx.B.CreateIntToPtr(BaseAddr, Ctx.PtrGlobalTy, "sbuf_base_ptr"),
          ExtentBytes};
}

// Rebuild a target buffer resource with the source scalar-buffer byte extent.
Value *emitTargetBufferResource(RaiseContext &Ctx,
                                const SourceScalarBufferResource &Resource) {
  Function *MakeRsrc = Intrinsic::getOrInsertDeclaration(
      &Ctx.M, Intrinsic::amdgcn_make_buffer_rsrc,
      {PointerType::get(Ctx.C, 8), PointerType::get(Ctx.C, 1)});
  // Build a target raw-buffer resource whose byte extent matches the source
  // scalar-buffer bound. `stride=0` is intentional here: S_BUFFER_LOAD does not
  // use stride to compute the address, only to derive its OOB size.
  return Ctx.B.CreateCall(
      MakeRsrc,
      {Resource.BasePtr, ConstantInt::get(Type::getInt16Ty(Ctx.C), 0),
       Resource.ExtentBytes, ConstantInt::get(Ctx.I32Ty, 0)},
      "sbuf_rsrc");
}

} // namespace

Expected<HandlerResult> handleSMEM(RaiseContext &Ctx, const DecodedInst &Di,
                                   OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  if (Sop == CanonicalOp::S_LOAD_B32 || Sop == CanonicalOp::S_LOAD_B64 ||
      Sop == CanonicalOp::S_LOAD_B96 || Sop == CanonicalOp::S_LOAD_B128 ||
      Sop == CanonicalOp::S_LOAD_B256 || Sop == CanonicalOp::S_LOAD_B512) {
    int LoadDwords = 1;
    switch (Sop) {
    case CanonicalOp::S_LOAD_B32:
      LoadDwords = 1;
      break;
    case CanonicalOp::S_LOAD_B64:
      LoadDwords = 2;
      break;
    case CanonicalOp::S_LOAD_B96:
      LoadDwords = 3;
      break;
    case CanonicalOp::S_LOAD_B128:
      LoadDwords = 4;
      break;
    case CanonicalOp::S_LOAD_B256:
      LoadDwords = 8;
      break;
    case CanonicalOp::S_LOAD_B512:
      LoadDwords = 16;
      break;
    default:
      break;
    }
    int LoadBytes = LoadDwords * 4;

    ParsedReg Dest = Op.dst();
    ParsedReg Base = Op.srcReg(0);

    unsigned OffIdx = Op.srcIdx(1);
    bool ImmOffset = Di.isImm(OffIdx);
    int64_t ByteOffset = ImmOffset ? Op.srcImm(1) : 0;
    // A hidden-argument read needs no translation: the runtime populated this
    // buffer at the source byte offsets, so it is an ordinary load. See
    // `hotswap/docs/abi-translation.md` sec. 3.3.
    //
    // Generic GEP+load against `addrspace(1)`. AMDGPU ISel selects the final
    // memory path from the pointer value's uniformity and provenance.
    {
      Value *BaseAddr = Ctx.Regs.loadSGPR64(Ctx.B, Base.BaseIdx);
      std::optional<uint64_t> SourceImageBase =
          Ctx.lookupSourceImageSgprPairAddr(Base.BaseIdx);
      if (SourceImageBase) {
        if (!ImmOffset) {
          return RaiseFailure::unsupportedInstructionForm(
              Di, "SMEM",
              "source-image SMEM base with dynamic offset cannot be resolved "
              "at raise time");
        }

        Expected<uint64_t> SourceAddr = applySourceImageByteOffset(
            Di, "SMEM", *SourceImageBase, ByteOffset);
        if (!SourceAddr)
          return SourceAddr.takeError();

        // Source PC-relative literal tables are not target GPU memory.
        // Resolve proven source-image loads here; otherwise the backend would
        // turn the source code-object address into a target VMEM access.
        for (int D = 0; D < LoadDwords; D++) {
          std::optional<uint64_t> DwordAddr = checkedMulAddUnsigned<uint64_t>(
              static_cast<uint64_t>(D), kSourceDwordBytes, *SourceAddr);
          if (!DwordAddr)
            return RaiseFailure::unsupportedInstructionForm(
                Di, "SMEM",
                "source-image SMEM dword address overflows while "
                "materialising a multi-dword load");
          std::optional<uint32_t> Dword = readSourceImageDword(Ctx, *DwordAddr);
          if (!Dword) {
            return RaiseFailure::unsupportedInstructionForm(
                Di, "SMEM",
                "constant SMEM address does not point inside a supported "
                "source code-object section");
          }
          Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx + D,
                               ConstantInt::get(Ctx.I32Ty, *Dword));
        }
        Hr.Handled = true;
        return Hr;
      }
      Value *Ptr = Ctx.B.CreateIntToPtr(BaseAddr, Ctx.PtrGlobalTy);
      if (ImmOffset) {
        if (ByteOffset != 0)
          Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr,
                                        Ctx.B.getInt64(ByteOffset));
      } else {
        // gfx12+ SMEM: when the `scale_offset` (CPol::SCAL) bit is
        // set the SGPR offset is an element index, not a byte
        // offset -- hardware multiplies it by the load's data-type
        // size before adding to sbase. Mirror that here (the
        // FLAT/GLOBAL counterparts in flat-addr.cpp do the same
        // against `elemBytes`). The "element size" for the scalar
        // dword family is the full load width: 4B for B32, 8B for
        // B64, 16B for B128, etc. -- i.e. `loadBytes`. Ignoring the
        // scale produced a silent off-by-N* miscompile on
        // `mask[blockIdx.x]`-style uses of a uniform SGPR index.
        Value *RegOff = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty, "smem_roff");
        if (Di.HasScaleOffset)
          RegOff =
              Ctx.B.CreateMul(RegOff, ConstantInt::get(Ctx.I64Ty, LoadBytes),
                              "smem_roff_scaled");
        RegOff =
            addStaticSmemByteOffset64(Ctx, Di, RegOff, "smem_roff_plus_imm");
        Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, RegOff);
      }
      for (int D = 0; D < LoadDwords; D++) {
        Value *Ep = (D == 0) ? Ptr
                             : Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr,
                                                       Ctx.B.getInt64(D * 4));
        Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx + D,
                             Ctx.B.CreateLoad(Ctx.I32Ty, Ep, "smem_load"));
      }
    }
    Hr.Handled = true;
    return Hr;
  }

  if (isScalarBufferLoad(Sop)) {
    ParsedReg Dest = Op.dst();
    ParsedReg Base = Op.srcReg(0);

    if (Dest.RegKind != ParsedReg::SGPR || Dest.BaseIdx < 0) {
      return RaiseFailure::unsupportedInstructionForm(
          Di, "SMEM", "S_BUFFER_LOAD expects an SGPR destination");
    }
    // The payload width is encoded in the TableGen destination operand class.
    unsigned LoadDwords = defRegClassDwordCount(Ctx, Di, /*DefIdx=*/0);
    if (Base.RegKind != ParsedReg::SGPR || Base.BaseIdx < 0 ||
        (Base.BaseIdx % 4) != 0 ||
        static_cast<size_t>(Base.BaseIdx + 3) >= Ctx.Regs.Sgpr.size()) {
      return RaiseFailure::unsupportedInstructionForm(
          Di, "SMEM",
          "S_BUFFER_LOAD expects a four-SGPR resource descriptor in sbase");
    }
    if (Di.HasScaleOffset) {
      return RaiseFailure::unsupportedInstructionForm(
          Di, "SMEM",
          "S_BUFFER_LOAD does not support scale_offset; refusing malformed "
          "or unmodelled SMEM buffer offset semantics");
    }
    // SMEM CPol carries TH (temporal hint) and SCOPE (cache-coherence scope)
    // bits. For a read-only S_BUFFER_LOAD these are cache *hints*: they steer
    // which cache level is used and how the line is aged, but they do not
    // change the value returned. Rebuilding the access below with the default
    // target cache policy (AuxFlags=0) therefore preserves load correctness --
    // at worst it differs in caching/perf, or, for a SCOPE_SYS load, it could
    // read from a nearer cache. That staleness window cannot be observed here:
    // such scalar loads read kernarg/descriptor/tile data produced before the
    // launch and made visible by the queue's inter-kernel synchronization, not
    // concurrently by another agent. Store semantics would be different, but
    // S_BUFFER_LOAD is always a load, so ignoring CPol is sound.

    unsigned OffIdx = Op.srcIdx(1);
    Value *Offset = nullptr;
    if (Di.isImm(OffIdx)) {
      int64_t Imm = Op.srcImm(1);
      if (Imm < 0) {
        return RaiseFailure::unsupportedInstructionForm(
            Di, "SMEM",
            "S_BUFFER_LOAD negative static offset would MEMVIOL on source");
      }
      Offset = ConstantInt::get(Ctx.I32Ty, static_cast<uint32_t>(Imm) & ~3u);
    } else if (Di.isReg(OffIdx)) {
      if (Di.StaticOffset && *Di.StaticOffset < 0) {
        return RaiseFailure::unsupportedInstructionForm(
            Di, "SMEM",
            "S_BUFFER_LOAD negative static offset would MEMVIOL on source");
      }
      Offset = alignDwordOffset32(Ctx, Op.src(1), "sbuf_soffset_aligned");
      if (Di.StaticOffset && *Di.StaticOffset != 0)
        Offset = Ctx.B.CreateAdd(
            Offset,
            ConstantInt::get(Ctx.I32Ty,
                             static_cast<uint32_t>(*Di.StaticOffset) & ~3u),
            "sbuf_offset_plus_imm");
    } else {
      return RaiseFailure::unsupportedInstructionForm(
          Di, "SMEM", "S_BUFFER_LOAD offset must be an immediate or SGPR");
    }

    SourceScalarBufferResource Resource =
        decodeSourceScalarBufferResource(Ctx, Base);
    Value *Rsrc = emitTargetBufferResource(Ctx, Resource);
    Value *Soffset = ConstantInt::get(Ctx.I32Ty, 0);
    Value *AuxFlags = ConstantInt::get(Ctx.I32Ty, 0);
    // S_BUFFER_LOAD reads through the buffer resource in sbase. Decode the
    // source fields this instruction uses, then rebuild a target resource with
    // the same base and byte extent. Target buffer hardware then returns zero
    // for out-of-bounds load elements.
    // This path always uses the default cache policy; any source SMEM TH/SCOPE
    // cpol bits are ignored above (sound for a read-only S_BUFFER_LOAD).
    //
    // Use raw-pointer buffer loads because WaveNative can carry different
    // descriptor values for the two source-wave halves inside one target wave.
    // The backend lowers such non-uniform resource values correctly.
    for (unsigned D = 0; D < LoadDwords;) {
      // Target raw-buffer loads select up to dwordx4, so split wider scalar
      // buffer loads into consecutive chunks.
      unsigned ChunkDwords = std::min(4u, LoadDwords - D);
      Type *LoadTy = ChunkDwords == 1
                         ? Ctx.I32Ty
                         : static_cast<Type *>(
                               FixedVectorType::get(Ctx.I32Ty, ChunkDwords));
      Function *BufLd = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_raw_ptr_buffer_load, {LoadTy});
      Value *ChunkOffset = Offset;
      if (D != 0)
        ChunkOffset = Ctx.B.CreateAdd(
            Offset, ConstantInt::get(Ctx.I32Ty, static_cast<uint32_t>(D * 4)),
            "sbuf_chunk_offset");
      Value *Loaded = Ctx.B.CreateCall(
          BufLd, {Rsrc, ChunkOffset, Soffset, AuxFlags}, "sbuf_load");

      if (ChunkDwords == 1) {
        Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx + static_cast<int>(D), Loaded);
      } else {
        for (unsigned I = 0; I < ChunkDwords; ++I) {
          Value *Dw = Ctx.B.CreateExtractElement(
              Loaded, ConstantInt::get(Ctx.I32Ty, I), "sbuf_load_dw");
          Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx + static_cast<int>(D + I),
                               Dw);
        }
      }
      D += ChunkDwords;
    }
    Hr.Handled = true;
    return Hr;
  }

  // gfx12+ scalar narrow loads: s_load_{u8,i8,u16,i16}. These fetch 1 or 2
  // bytes from a uniform address materialised in an SGPR-pair base and
  // zero/sign-extend the result into a 32-bit SGPR. MC operand shape
  // matches the dword-granular s_load_* family (sbase + imm-or-sgpr
  // offset), so operand decoding mirrors the S_LOAD_B* block above.
  //
  // Design notes:
  //  * IR shape: `load iN, ptr addrspace(1) %p, align N` + `zext`/`sext`
  //    to i32 -> `storeSGPR32`. No AMDGPU-specific intrinsic exists for
  //    narrow scalar loads; the backend's ISel matches the uniform-address
  //    pattern directly.
  //  * Same-target gfx1250 -> gfx1250: the backend re-codegens to the
  //    original `s_load_u16` / `s_load_u8` / etc. (identity-preserving).
  //  * Cross-target gfx1250 -> gfx942: the backend has no native narrow
  //    SMEM load, so it lowers to VMEM (`global_load_ushort` / ubyte).
  //    The lifted kernel stays correct -- the value appears on every lane
  //    with the same content, matching the SMEM broadcast semantics --
  //    but the register class shifts SGPR->VGPR and the memory path
  //    shifts scalar-cache->vector-cache.
  //  * Alignment: explicit `Align(1)` for byte, `Align(2)` for halfword.
  //
  // Test back-reference: lit_tests/s_load_u16/ exercises the halfword
  // same-target happy path. The byte (u8/i8) and signed (i8/i16)
  // variants share this handler body.
  if (Sop == CanonicalOp::S_LOAD_U8 || Sop == CanonicalOp::S_LOAD_I8 ||
      Sop == CanonicalOp::S_LOAD_U16 || Sop == CanonicalOp::S_LOAD_I16) {
    bool IsHalfWord =
        (Sop == CanonicalOp::S_LOAD_U16 || Sop == CanonicalOp::S_LOAD_I16);
    bool IsSigned =
        (Sop == CanonicalOp::S_LOAD_I8 || Sop == CanonicalOp::S_LOAD_I16);
    Type *I16Ty = Type::getInt16Ty(Ctx.C);
    Type *NarrowTy = IsHalfWord ? I16Ty : Ctx.I8Ty;
    Align NarrowAlign = Align(IsHalfWord ? 2 : 1);
    const char *NarrowLoadName = IsHalfWord ? "smem_load_h" : "smem_load_b";
    const char *ExtName = IsSigned ? "smem_load_sext" : "smem_load_zext";

    ParsedReg Dest = Op.dst();
    ParsedReg Base = Op.srcReg(0);

    Value *BaseAddr = Ctx.Regs.loadSGPR64(Ctx.B, Base.BaseIdx);
    Value *Ptr = Ctx.B.CreateIntToPtr(BaseAddr, Ctx.PtrGlobalTy);
    unsigned OffIdx = Op.srcIdx(1);
    if (Di.isImm(OffIdx)) {
      // Narrow reads of a source hidden field (`hidden_group_size_*` and
      // `hidden_remainder_*` are 16-bit) are ordinary loads for the same
      // reason as the dword family above.
      int64_t Off = Op.srcImm(1);
      if (Off != 0)
        Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, Ctx.B.getInt64(Off));
    } else {
      // Narrow SMEM element size for `scale_offset`: 1B for byte,
      // 2B for halfword. Same SCAL-scales-the-SGPR-offset rule as
      // the dword family above.
      int NarrowBytes = IsHalfWord ? 2 : 1;
      Value *RegOff = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty, "smem_nroff");
      if (Di.HasScaleOffset && NarrowBytes != 1)
        RegOff =
            Ctx.B.CreateMul(RegOff, ConstantInt::get(Ctx.I64Ty, NarrowBytes),
                            "smem_nroff_scaled");
      RegOff =
          addStaticSmemByteOffset64(Ctx, Di, RegOff, "smem_nroff_plus_imm");
      Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, RegOff);
    }

    Value *Narrow =
        Ctx.B.CreateAlignedLoad(NarrowTy, Ptr, NarrowAlign, NarrowLoadName);
    Value *Ext = IsSigned ? Ctx.B.CreateSExt(Narrow, Ctx.I32Ty, ExtName)
                          : Ctx.B.CreateZExt(Narrow, Ctx.I32Ty, ExtName);
    Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx, Ext);
    Hr.Handled = true;
    return Hr;
  }

  // s_store_* (scalar store through SGPR base + imm/sgpr offset).
  // MC operand layout: (sdata, sbase, soffset/imm, cpol).
  if (Sop == CanonicalOp::S_STORE_B32 || Sop == CanonicalOp::S_STORE_B64 ||
      Sop == CanonicalOp::S_STORE_B128) {
    int StoreDwords = (Sop == CanonicalOp::S_STORE_B32)   ? 1
                      : (Sop == CanonicalOp::S_STORE_B64) ? 2
                                                          : 4;
    ParsedReg Data = Op.srcReg(0);
    ParsedReg Base = Op.srcReg(1);
    if (Data.RegKind != ParsedReg::SGPR || Base.RegKind != ParsedReg::SGPR) {
      llvm::errs() << "transpiler: " << Di.Mnemonic
                   << ": S_STORE expects SGPR data and base\n";
      return RaiseFailure::unsupportedInstructionForm(
          Di, "SMEM", "S_STORE expects SGPR data and base");
    }
    Value *BaseAddr = Ctx.Regs.loadSGPR64(Ctx.B, Base.BaseIdx);
    Value *Ptr = Ctx.B.CreateIntToPtr(BaseAddr, Ctx.PtrGlobalTy);
    int StoreBytes = StoreDwords * 4;
    if (Op.nSrcs() >= 3) {
      unsigned OffIdx = Op.srcIdx(2);
      if (Di.isImm(OffIdx)) {
        int64_t Off = Op.srcImm(2);
        if (Off != 0)
          Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, Ctx.B.getInt64(Off));
      } else if (Di.isReg(OffIdx)) {
        // Same `scale_offset` scaling as the S_LOAD path -- the
        // SCAL bit multiplies the SGPR offset by the store's
        // data-type size (4/8/16B for B32/B64/B128).
        Value *RegOff = Ctx.B.CreateZExt(Op.src(2), Ctx.I64Ty, "smem_st_roff");
        if (Di.HasScaleOffset)
          RegOff =
              Ctx.B.CreateMul(RegOff, ConstantInt::get(Ctx.I64Ty, StoreBytes),
                              "smem_st_roff_scaled");
        RegOff =
            addStaticSmemByteOffset64(Ctx, Di, RegOff, "smem_st_roff_plus_imm");
        Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, RegOff);
      }
    }
    for (int D = 0; D < StoreDwords; D++) {
      Value *Ep = (D == 0) ? Ptr
                           : Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr,
                                                     Ctx.B.getInt64(D * 4));
      Value *V = Ctx.Regs.loadSGPR32(Ctx.B, Data.BaseIdx + D);
      Ctx.B.CreateStore(V, Ep);
    }
    Hr.Handled = true;
    return Hr;
  }

  // SMEM dword atomics (returned-old-value / GLC=1 / `_RTN` form only):
  // operate on memory through an SGPR-pair base pointer with an IMM /
  // SGPR / SGPR_IMM offset, and publish the pre-modification value into
  // the sdst (== tied sdst_in) SGPR slot.
  //
  // Dispatch is keyed on the CanonicalOp; each arm picks the `atomicrmw`
  // BinOp that matches the hardware's scalar-cache semantics exactly:
  //   S_ATOMIC_SWAP -> Xchg      (pure exchange)
  //   S_ATOMIC_DEC  -> UDecWrap  (AMDGPU HW wrap-at-zero decrement:
  //                               new = (old == 0 || old > src) ? src
  //                                                             : old - 1;
  //                               NOT `atomicrmw sub`, which would
  //                               silently underflow past zero)
  //
  // The returned-old-value write-back is the hot path: AITER split-k
  // reductions (bf16gemm_*_splitk_clean) key the "last workgroup runs
  // the epilogue" barrier on `old == 1`, so dropping the dst store
  // would silently break the branch that follows.
  //
  // Non-matching SemOps fall through the `default:` arm without the
  // `hr.Handled` flip, so the raiser's Phase-5 unsupportedOpcode path
  // reports the missing lowering.
  AtomicRMWInst::BinOp RmwOp;
  switch (Sop) {
  case CanonicalOp::S_ATOMIC_SWAP:
    RmwOp = AtomicRMWInst::Xchg;
    break;
  case CanonicalOp::S_ATOMIC_DEC:
    RmwOp = AtomicRMWInst::UDecWrap;
    break;
  default:
    return Hr;
  }

  // Two disassembler shapes for the same CanonicalOp, distinguished by the
  // instruction's GLC bit (encoding-level) and `di.numDefs`
  // (decode-level).  The TableGen source of truth is
  // `llvm/lib/Target/AMDGPU/SMInstructions.td`, class
  // `SM_Pseudo_Atomic<..., bit isRet, ...>`:
  //
  //     !if(isRet, (outs dataClass:$sdst), (outs))
  //     (ins dataClass:$sdata, baseClass:$sbase, <offset>, CPolTy:$cpol)
  //     let Constraints = !if(isRet, "$sdst = $sdata", "")
  //
  // i.e. both forms always carry `$sdata`, `$sbase`, the offset, and
  // `$cpol` in the `ins` list; only RTN adds `$sdst` in the `outs`
  // list and ties it to `$sdata` (which the MC layer then elides
  // from the operand-print list, leaving the decoded MCInst as):
  //
  //   RTN     (GLC=1, numDefs=1, isRet=1):
  //           `(sdst, sbase, offset, cpol)` -- TableGen's tied
  //           `"$sdst = $sdata"` constraint elides the tied input
  //           from the operand list; the atomic's input value (xchg
  //           data for swap, wrap-threshold for dec) comes from the
  //           *pre-instruction* value of the dst SGPR; the post-
  //           instruction value is the returned `old`.  AITER's
  //           split-k barrier keys its "am I the last workgroup?"
  //           branch on `old == 1`, so this is the hot path for
  //           `bf16gemm_*_splitk_clean.co` lowering.
  //
  //   non-RTN (GLC=0, numDefs=0, isRet=0):
  //           `(sdata, sbase, offset, cpol)` -- no dst at all; `sdata`
  //           stays as an explicit source operand.  The atomic runs
  //           and the returned `old` is dropped on the floor.  hipcc's
  //           inline-asm lowering of `s_atomic_dec %[rmw], %[ptr],
  //           %[off]` (no `_rtn` suffix in the mnemonic string)
  //           produces this shape even when the `"+s"(rmw)`
  //           constraint suggests a tied in/out, so the non-RTN arm
  //           has to exist for any HIP fixture that spells the
  //           instruction via inline asm.  See `lit_tests/s_atomic_dec/`
  //           for the canonical fixture (split-k barrier reproducer).
  //
  // Common to both arms: the atomic binop (`rmwOp` above), the base
  // pointer in `sbase` (SGPR pair), a `soffset` that is either an
  // inline imm or an SGPR element index scaled by the dword width
  // when the `scale_offset` (CPol::SCAL) bit is set, and the
  // explicit `Align(4)` / `AtomicOrdering::Monotonic`.  `cpol` is
  // not consumed by the lift (the GLC bit it carries is already
  // reflected in `di.numDefs`; the non-GLC CPol bits -- DLC, SCOPE,
  // SCAL -- are either already threaded through `di.hasScaleOffset`
  // or are cache-hint-only and thus lift-invariant).
  //
  // No SMEM atomic today has more than one def.  The assertion below
  // pins that invariant so a hypothetical future 2-def form (none
  // exists in any ISA the raiser targets) fails loudly rather than
  // silently taking the RTN arm and mis-decoding `op.dst()` /
  // `op.dst(1)`.
  assert(Di.NumDefs <= 1 &&
         "SMEM atomic with >1 defs is not a shape the lift recognises");
  ParsedReg Base;
  unsigned OffIdx;
  Value *Data = nullptr;
  ParsedReg DataDst; // Set only on the RTN arm.
  if (Di.NumDefs == 0) {
    // Non-RTN: (sdata, sbase, offset).  Read sdata as the atomic's
    // input value; the returned `old` is intentionally discarded
    // below (the HW already wrote the new value to memory, which is
    // the whole point of the non-RTN form's existence).
    Base = Op.srcReg(1);
    OffIdx = Op.srcIdx(2);
    Data = Ctx.Regs.readReg32(Ctx.B, Op.srcReg(0));
  } else {
    // RTN: (sdst_tied, sbase, offset).  Read the pre-instruction
    // value of sdst as the atomic's input (this is the tied-input
    // slot TableGen's `"$sdst = $sdata"` constraint names), then
    // write the returned `old` back to the same SGPR after the
    // atomicrmw.
    DataDst = Op.dst();
    Base = Op.srcReg(0);
    OffIdx = Op.srcIdx(1);
    Data = Ctx.Regs.readReg32(Ctx.B, DataDst);
  }

  Value *BaseAddr = Ctx.Regs.loadSGPR64(Ctx.B, Base.BaseIdx);
  Value *Ptr = Ctx.B.CreateIntToPtr(BaseAddr, Ctx.PtrGlobalTy);

  // Positional source index of the offset operand in OpResolver's
  // operand view: slot 1 for the RTN shape (sbase, offset), slot 2
  // for the non-RTN shape (sdata, sbase, offset).  Keeps the
  // imm/SGPR-offset arm routed through the generic `op.src()` reader
  // (which handles imm, SGPR, and any other source kind uniformly)
  // so the RTN path's IR is bit-identical to the pre-split handler.
  unsigned OffSrcPos = (Di.NumDefs == 0 ? 2u : 1u);
  if (Di.isImm(OffIdx)) {
    int64_t Off = Di.getImm(OffIdx);
    if (Off != 0)
      Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, Ctx.B.getInt64(Off));
  } else {
    // Dword-width atomic, so the `scale_offset` (CPol::SCAL) bit
    // scales the SGPR element-index by 4 to recover the byte offset
    // -- same rule as the other SMEM paths.
    Value *RegOff =
        Ctx.B.CreateZExt(Op.src(OffSrcPos), Ctx.I64Ty, "smem_at_roff");
    if (Di.HasScaleOffset)
      RegOff = Ctx.B.CreateMul(RegOff, ConstantInt::get(Ctx.I64Ty, 4),
                               "smem_at_roff_scaled");
    RegOff =
        addStaticSmemByteOffset64(Ctx, Di, RegOff, "smem_at_roff_plus_imm");
    Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, RegOff);
  }

  // Pin `Align(4)` explicitly instead of letting the IRBuilder infer
  // ABI alignment. Matches the explicit-Align convention the narrow
  // SMEM load block above sets (see its "Alignment:" design bullet);
  // relying on ABI inference happens to produce `align 4` for i32 on
  // AS(1) today, but inferred alignment is fragile against future LLVM
  // default-alignment changes and masks pointer-alignment bugs from
  // callers.
  Value *Old = Ctx.B.CreateAtomicRMW(RmwOp, Ptr, Data, Align(4),
                                     AtomicOrdering::Monotonic);
  // RTN arm only: publish the pre-modification value to the tied
  // sdst SGPR.  The non-RTN arm has no write-back -- the HW has
  // already committed the new value to memory, which is all that
  // form guarantees, and `old` is left as a dead SSA value that
  // LLVM's DCE will remove in the usual way.
  if (Di.NumDefs != 0)
    Ctx.Regs.storeSGPR32(Ctx.B, DataDst.BaseIdx, Old);
  Hr.Handled = true;
  return Hr;
}

} // namespace COMGR::hotswap
