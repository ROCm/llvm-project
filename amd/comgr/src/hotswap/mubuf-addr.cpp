//===- mubuf-addr.cpp - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mubuf-addr.h"

#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName::offset
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Classify source operands of a MUBUF load/store into {srsrc, vaddr,
// soff, imm, vdata}. Keys on `ParsedReg::Kind` rather than position so
// MUBUF and VBUFFER encodings with different operand orders both
// route here. `isStore` controls the skip-first-VGPR rule: stores
// carry vdata (VGPR) ahead of vaddr (VGPR) in the operand list.
//
// `immOff` is read by name (`AMDGPU::OpName::offset`) rather than
// scanning for the first non-zero immediate. The greedy scan was
// unsound: MUBUF / VBUFFER carry both an `OpName::offset` byte
// immediate AND an `OpName::cpol` cache-policy immediate, and when
// the byte offset is zero the scan would silently grab cpol's value
// (e.g. `0x20` for `scope:SCOPE_DEV`) and use it as a per-lane voffset
// -- every store/load with `offset:0 scope:SCOPE_*` would land 32 B
// past the intended address. The R1 lit regression guard issues
// exactly this shape (`buffer_store_b128 ... offset:0 scope:SCOPE_DEV`),
// so the fix is required for the BUFFER_STORE rewrite to actually
// hit the right address. CPol bits are otherwise still dropped (see
// the "Not refused here" note in handle-mubuf.cpp's refusal block,
// and sec. "Known limitations" of hotswap/docs/buffer-store-lowering.md).
struct MubufOps {
  ParsedReg Srsrc;
  ParsedReg Vaddr;
  ParsedReg Soff;
  ParsedReg Vdata;
  int64_t ImmOff = 0;
  bool HaveSrsrc = false;
  bool HaveVaddr = false;
  bool HaveSoff = false;
  bool HasSwz = false;
};

MubufOps classifyMubufOps(const DecodedInst &Di, OpResolver &Op, bool IsStore) {
  MubufOps Out;

  // Byte offset by name. Absent (-1) on encodings that don't carry it
  // (e.g. atomics with no immediate offset slot); the default `immOff
  // = 0` is correct for those -- the caller layers in `voffset` from
  // `vaddr` independently.
  int OffIdx = llvm::AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                llvm::AMDGPU::OpName::offset);
  if (OffIdx >= 0 && static_cast<unsigned>(OffIdx) < Di.Inst.getNumOperands() &&
      Di.Inst.getOperand(static_cast<unsigned>(OffIdx)).isImm()) {
    Out.ImmOff = Di.Inst.getOperand(static_cast<unsigned>(OffIdx)).getImm();
  }

  int SwzIdx = llvm::AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                llvm::AMDGPU::OpName::swz);
  if (SwzIdx >= 0 && static_cast<unsigned>(SwzIdx) < Di.Inst.getNumOperands() &&
      Di.Inst.getOperand(static_cast<unsigned>(SwzIdx)).isImm()) {
    Out.HasSwz =
        Di.Inst.getOperand(static_cast<unsigned>(SwzIdx)).getImm() != 0;
  }

  int VgprSrcCount = 0;
  for (unsigned K = 0; K < Op.nSrcs(); ++K) {
    if (!Di.isReg(Op.srcIdx(K)))
      continue;
    ParsedReg Pr = Op.srcReg(K);
    if (Pr.RegKind == ParsedReg::SGPR && Pr.BaseIdx >= 0 && !Out.HaveSrsrc) {
      Out.Srsrc = Pr;
      Out.HaveSrsrc = true;
    } else if (Pr.RegKind == ParsedReg::VGPR) {
      VgprSrcCount++;
      // For stores, the first VGPR source is vdata (the stored value);
      // the second is the per-lane buffer offset (vaddr).
      if (IsStore && VgprSrcCount == 1) {
        Out.Vdata = Pr;
        continue;
      }
      if (!Out.HaveVaddr) {
        Out.Vaddr = Pr;
        Out.HaveVaddr = true;
      }
    } else if (Pr.RegKind == ParsedReg::SGPR && Pr.BaseIdx >= 0 &&
               !Out.HaveSoff) {
      Out.Soff = Pr;
      Out.HaveSoff = true;
    }
  }
  return Out;
}

// Read the four consecutive SGPR dwords of a MUBUF/VBUFFER SRSRC
// 128-bit tuple. Returns the raw source dwords; callers that only need
// the packaged raw-buffer descriptor should use `buildMubufSRD` below.
//
// No null-check on the returned Value*s: `AllocaRegFile::readReg32`
// already fails loudly on unhandled ParsedReg kinds and out-of-range
// SGPR indices, so SGPR reads are guaranteed to hand back a real
// Value*.
struct SRSRCDwords {
  Value *Dw0;
  Value *Dw1;
  Value *Dw2;
  Value *Dw3;
};

SRSRCDwords readSRSRCDwords(RaiseContext &Ctx, ParsedReg Srsrc) {
  Value *Dw0 = Ctx.Regs.readReg32(Ctx.B, Srsrc);
  ParsedReg Srsrc1 = Srsrc;
  Srsrc1.BaseIdx = Srsrc.BaseIdx + 1;
  ParsedReg Srsrc2 = Srsrc;
  Srsrc2.BaseIdx = Srsrc.BaseIdx + 2;
  ParsedReg Srsrc3 = Srsrc;
  Srsrc3.BaseIdx = Srsrc.BaseIdx + 3;
  Value *Dw1 = Ctx.Regs.readReg32(Ctx.B, Srsrc1);
  Value *Dw2 = Ctx.Regs.readReg32(Ctx.B, Srsrc2);
  Value *Dw3 = Ctx.Regs.readReg32(Ctx.B, Srsrc3);
  return {Dw0, Dw1, Dw2, Dw3};
}

bool constantI32(Value *V, uint32_t &Out) {
  if (auto *Ci = dyn_cast<ConstantInt>(V)) {
    Out = static_cast<uint32_t>(Ci->getZExtValue());
    return true;
  }
  return false;
}

// Build a gfx942-compatible raw buffer descriptor <4 x i32> from the
// source SRSRC dwords. Same-wave descriptors are routed through
// `amdgcn.readfirstlane` so they land in SGPRs directly. Cross-widening MUBUF
// loads use the parallel addrspace(8) `rawPtrRsrc` built in decodeMubufAddr
// below, because LLVM's raw-pointer buffer intrinsic preserves the same
// high-level resource form Triton emitted (`make.buffer.rsrc` +
// `raw.ptr.buffer.load`) and avoids hand-translating gfx1250 raw-pointer
// descriptor bits into a target SRD.
//
// dw1/dw2/dw3 are target-normalised rather than blindly copied from the
// source SGPRs. gfx1250 raw-pointer descriptors carry high descriptor
// marker bits alongside the base-high dword and use a compact
// NUM_RECORDS sentinel that is not gfx942's raw-buffer maximum, while
// gfx942 still needs a non-invalid DATA_FORMAT in RSRC3 for stores to
// commit.
//
// Why dw3 must not be zero on gfx942: empirically
// `buffer_store_dword` (and the other MUBUF raw-store flavours) on
// CDNA3 silently drops every lane's write when dw3's DATA_FORMAT
// field is BUF_DATA_FORMAT_INVALID (0). The ISA manual advertises
// these ops as untyped, but the MI300 MUBUF engine still checks
// DATA_FORMAT != 0 before committing the store. This manifests as
// Softmax (the only corpus kernel that reaches us through MUBUF
// stores -- vecadd / add_fp32 use FLAT/GLOBAL) leaving its output
// buffer at sentinel after a hotswap run, with no HIP error, no GPU
// fault, and no stderr output from the runtime.
//
// Bisected with kernels/build/mubuf_store_b32 + inline-asm V#
// probes:
//
//   dw3=0x00020000  MATCH   DATA_FORMAT=32 alone (minimum working)
//   dw3=0x00024000  MATCH   DATA_FORMAT=32 + NUM_FORMAT=UINT
//   dw3=0x00027000  MATCH   Triton's native-gfx942 shape (FORMAT_32 + FLOAT)
//   dw3=0x00007000  DROP    NUM_FORMAT=FLOAT alone, DATA_FORMAT=0
//   dw3=0x00000004  DROP    DST_SEL_X identity, DATA_FORMAT=0
//   dw3=0x00000000  DROP    (what we used to emit)
//
// Use Triton's native gfx942 raw-buffer sentinels, narrowly:
//
//   RSRC2 = 0x7ffffffe  NUM_RECORDS, the largest 4-byte-aligned byte
//                       bound used by native gfx942 Triton for raw
//                       pointer-derived descriptors.
//   RSRC3 = 0x00027000  FORMAT_32 + NUM_FORMAT_FLOAT.
//
// The source descriptor value 0x00ffffff is the gfx1250 raw-pointer
// "effectively unbounded" sentinel Triton emits for these JIT MUBUF
// descriptors. Passing it through to gfx942 bounds the buffer to 16 MiB,
// so GPT-OSS decode_attention._fwd_kernel_stage2's `Mid_O` loads for
// batches >= 64 were hardware-OOB and returned zero. Map that sentinel
// to gfx942's native raw-buffer maximum, but preserve every other
// NUM_RECORDS value exactly. That keeps finite source bounds finite
// instead of turning real source-OOB accesses into target in-bounds
// accesses. A true all-ones source value (0xffffffff, OOB disabled per
// the buffer-resource contract) is also preserved.
//
// The previous DATA_FORMAT-only RSRC3 value (0x00020000) is sufficient
// for the original dword-store probe, but native gfx942 Triton uses
// FORMAT_32 + NUM_FORMAT_FLOAT for the same raw store family; the raw
// intrinsics still move the explicitly typed payload bits without
// numeric conversion.
Expected<Value *> buildMubufSRD(RaiseContext &Ctx, const SRSRCDwords &Dw) {
  constexpr uint32_t kGfx1250RawPointerWord1Bits = 0xfc000000u;
  constexpr uint32_t kGfx1250RawBufferMaxRecords = 0x00ffffffu;
  constexpr uint32_t kGfx942RawBufferMaxRecords = 0x7ffffffeu;
  constexpr uint32_t kGfx942RawBufferFormat32 = 0x00020000u;
  constexpr uint32_t kGfx942RawBufferFormat32Uint = 0x00024000u;
  constexpr uint32_t kGfx942RawBufferFormat32Float = 0x00027000u;

  const bool CrossWidening = Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize;
  Function *Readfirstlane =
      CrossWidening ? nullptr
                    : Intrinsic::getOrInsertDeclaration(
                          &Ctx.M, Intrinsic::amdgcn_readfirstlane, {Ctx.I32Ty});
  auto ScalarizeDescriptorWord = [&](Value *Word, const char *Name) -> Value * {
    if (CrossWidening)
      return Word;
    return Ctx.B.CreateCall(Readfirstlane, {Word}, Name);
  };
  Value *Dw1NonBaseBits = Ctx.B.CreateAnd(
      Dw.Dw1, ConstantInt::get(Ctx.I32Ty, 0xFFFF0000u), "srd_dw1_nonbase_bits");
  Value *Dw1HasOnlyBase = Ctx.B.CreateICmpEQ(
      Dw1NonBaseBits, ConstantInt::get(Ctx.I32Ty, 0), "srd_dw1_only_base");
  Value *Dw1HasGfx125RawBits = Ctx.B.CreateICmpEQ(
      Dw1NonBaseBits, ConstantInt::get(Ctx.I32Ty, kGfx1250RawPointerWord1Bits),
      "srd_dw1_gfx125_raw_bits");
  Value *Dw1IsRawBase =
      Ctx.B.CreateOr(Dw1HasOnlyBase, Dw1HasGfx125RawBits, "srd_dw1_raw_base");
  Value *Dw3IsZero = Ctx.B.CreateICmpEQ(Dw.Dw3, ConstantInt::get(Ctx.I32Ty, 0),
                                        "srd_dw3_zero");
  Value *Dw3IsFormat32 = Ctx.B.CreateICmpEQ(
      Dw.Dw3, ConstantInt::get(Ctx.I32Ty, kGfx942RawBufferFormat32),
      "srd_dw3_format32");
  Value *Dw3IsFormat32Uint = Ctx.B.CreateICmpEQ(
      Dw.Dw3, ConstantInt::get(Ctx.I32Ty, kGfx942RawBufferFormat32Uint),
      "srd_dw3_format32_uint");
  Value *Dw3IsFormat32Float = Ctx.B.CreateICmpEQ(
      Dw.Dw3, ConstantInt::get(Ctx.I32Ty, kGfx942RawBufferFormat32Float),
      "srd_dw3_format32_float");
  Value *Dw3IsRaw =
      Ctx.B.CreateOr(Ctx.B.CreateOr(Dw3IsZero, Dw3IsFormat32),
                     Ctx.B.CreateOr(Dw3IsFormat32Uint, Dw3IsFormat32Float),
                     "srd_dw3_raw_shape");
  Value *RawPointerShape =
      Ctx.B.CreateAnd(Dw1IsRawBase, Dw3IsRaw, "srd_raw_pointer_shape");

  uint32_t Dw1Const = 0;
  if (constantI32(Dw.Dw1, Dw1Const) && (Dw1Const & 0xFFFF0000u) != 0 &&
      (Dw1Const & 0xFFFF0000u) != kGfx1250RawPointerWord1Bits) {
    return createStringError("transpiler: MUBUF: unsupported non-raw SRSRC "
                             "descriptor: source word1 contains structured/"
                             "swizzled fields that the raw-buffer lowering "
                             "cannot preserve");
  }
  uint32_t Dw3Const = 0;
  if (constantI32(Dw.Dw3, Dw3Const) && Dw3Const != 0 &&
      Dw3Const != kGfx942RawBufferFormat32 &&
      Dw3Const != kGfx942RawBufferFormat32Uint &&
      Dw3Const != kGfx942RawBufferFormat32Float) {
    return createStringError("transpiler: MUBUF: unsupported non-raw SRSRC "
                             "descriptor: source word3 is not a raw-buffer "
                             "FORMAT_32 descriptor shape");
  }

  Value *CleanDw1 =
      Ctx.B.CreateAnd(Dw.Dw1, ConstantInt::get(Ctx.I32Ty, 0xFFFF));
  Value *SrdW0 = ScalarizeDescriptorWord(Dw.Dw0, "srd_w0");
  Value *SrdW1 = ScalarizeDescriptorWord(CleanDw1, "srd_w1");
  Value *SourceMax = ConstantInt::get(Ctx.I32Ty, kGfx1250RawBufferMaxRecords);
  Value *TargetMax = ConstantInt::get(Ctx.I32Ty, kGfx942RawBufferMaxRecords);
  Value *IsSourceMax =
      Ctx.B.CreateICmpEQ(Dw.Dw2, SourceMax, "srd_is_gfx125_max");
  Value *ShouldMapMax =
      Ctx.B.CreateAnd(IsSourceMax, RawPointerShape, "srd_map_gfx125_max");
  Value *MappedDw2 =
      Ctx.B.CreateSelect(ShouldMapMax, TargetMax, Dw.Dw2, "srd_num_records");
  Value *SrdW2 = ScalarizeDescriptorWord(MappedDw2, "srd_w2");
  Value *Word3 = ConstantInt::get(Ctx.I32Ty, kGfx942RawBufferFormat32Float);
  Value *Srd = UndefValue::get(FixedVectorType::get(Ctx.I32Ty, 4));
  Srd = Ctx.B.CreateInsertElement(Srd, SrdW0, static_cast<uint64_t>(0));
  Srd = Ctx.B.CreateInsertElement(Srd, SrdW1, static_cast<uint64_t>(1));
  Srd = Ctx.B.CreateInsertElement(Srd, SrdW2, static_cast<uint64_t>(2));
  Srd = Ctx.B.CreateInsertElement(Srd, Word3, static_cast<uint64_t>(3));
  return Srd;
}

} // namespace

Expected<MubufAddr> decodeMubufAddr(RaiseContext &Ctx, const DecodedInst &Di,
                                    OpResolver &Op, bool IsStore,
                                    StringRef DiagLabel) {
  MubufOps M = classifyMubufOps(Di, Op, IsStore);

  if (!M.HaveSrsrc) {
    std::string Msg;
    raw_string_ostream Os(Msg);
    Os << "transpiler: " << DiagLabel << ": no SRSRC found for "
       << Di.RawMnemonic;
    return createStringError(Os.str());
  }

  MubufAddr Out;
  if (M.HasSwz) {
    return createStringError("transpiler: " + DiagLabel +
                             ": swizzled buffer addressing is unsupported "
                             "by the raw-buffer lowering");
  }
  SRSRCDwords Dw = readSRSRCDwords(Ctx, M.Srsrc);
  Expected<Value *> Srd = buildMubufSRD(Ctx, Dw);
  if (!Srd)
    return Srd.takeError();
  Out.Srd = *Srd;
  Out.StData = M.Vdata;

  Value *Voffset = ConstantInt::get(Ctx.I32Ty, 0);
  if (M.HaveVaddr)
    Voffset = Ctx.B.CreateAdd(Voffset, Ctx.Regs.readReg32(Ctx.B, M.Vaddr));
  if (M.ImmOff != 0)
    Voffset = Ctx.B.CreateAdd(
        Voffset, ConstantInt::get(Ctx.I32Ty, static_cast<int32_t>(M.ImmOff)));
  Out.Voffset = Voffset;

  Out.Soffset = M.HaveSoff ? Ctx.Regs.readReg32(Ctx.B, M.Soff)
                           : ConstantInt::get(Ctx.I32Ty, 0);
  constexpr uint32_t kGfx1250RawBufferMaxRecords = 0x00ffffffu;
  constexpr uint32_t kGfx942RawBufferMaxRecords = 0x7ffffffeu;
  Value *NumRecords = nullptr;
  if (Ctx.Isa.Has45BitNumRecordsBufferResource) {
    // gfx12+ V#: base = resource[56:0], num_records = resource[101:57] (bytes).
    // The field starts at bit 57 (not the 64-bit dword boundary), so source
    // dword2 alone holds num_records[38:7] -- the byte extent shifted right by
    // 7. Reading dword2 raw yields a bound 128x too small, so every A/B tile
    // buffer_load runs out of bounds and returns zero. Reconstruct the full
    // 45-bit num_records (mirrors handle-smem.cpp / S_BUFFER_LOAD):
    //   num_records[6:0]   = dword1[31:25]
    //   num_records[38:7]  = dword2[31:0]
    //   num_records[44:39] = dword3[5:0]
    Value *NumLo = Ctx.B.CreateAnd(
        Ctx.B.CreateLShr(Dw.Dw1, ConstantInt::get(Ctx.I32Ty, 25)),
        ConstantInt::get(Ctx.I32Ty, 0x7f));
    Value *Full = Ctx.B.CreateOr(
        Ctx.B.CreateShl(Ctx.B.CreateZExt(Dw.Dw2, Ctx.I64Ty), Ctx.B.getInt64(7)),
        Ctx.B.CreateZExt(NumLo, Ctx.I64Ty));
    Value *NumHi = Ctx.B.CreateShl(
        Ctx.B.CreateZExt(
            Ctx.B.CreateAnd(Dw.Dw3, ConstantInt::get(Ctx.I32Ty, 0x3f)),
            Ctx.I64Ty),
        Ctx.B.getInt64(39));
    Full = Ctx.B.CreateOr(Full, NumHi, "mubuf_raw_num_records_full");
    // gfx942 raw buffers carry a 32-bit byte extent; clamp the wider source
    // field to the native raw-buffer max (this also maps the gfx12 all-ones
    // "OOB disabled" / effectively-unbounded encodings onto gfx942's max).
    Value *TooBig =
        Ctx.B.CreateICmpUGT(Full, Ctx.B.getInt64(kGfx942RawBufferMaxRecords));
    NumRecords =
        Ctx.B.CreateSelect(TooBig, Ctx.B.getInt64(kGfx942RawBufferMaxRecords),
                           Full, "mubuf_raw_num_records");
  } else {
    Value *SourceMax = ConstantInt::get(Ctx.I32Ty, kGfx1250RawBufferMaxRecords);
    Value *TargetMax = ConstantInt::get(Ctx.I32Ty, kGfx942RawBufferMaxRecords);
    Value *IsSourceMax =
        Ctx.B.CreateICmpEQ(Dw.Dw2, SourceMax, "mubuf_raw_is_gfx125_max");
    NumRecords =
        Ctx.B.CreateZExt(Ctx.B.CreateSelect(IsSourceMax, TargetMax, Dw.Dw2),
                         Ctx.I64Ty, "mubuf_raw_num_records");
  }
  Value *CleanDw1 =
      Ctx.B.CreateAnd(Dw.Dw1, ConstantInt::get(Ctx.I32Ty, 0xFFFF));
  Value *BaseLo = Ctx.B.CreateZExt(Dw.Dw0, Ctx.I64Ty);
  Value *BaseHi = Ctx.B.CreateShl(Ctx.B.CreateZExt(CleanDw1, Ctx.I64Ty), 32);
  Value *Base = Ctx.B.CreateOr(BaseLo, BaseHi, "mubuf_raw_base");
  Value *BasePtr = Ctx.B.CreateIntToPtr(Base, PointerType::get(Ctx.C, 1),
                                        "mubuf_raw_base_ptr");
  Function *MakeRsrc = Intrinsic::getOrInsertDeclaration(
      &Ctx.M, Intrinsic::amdgcn_make_buffer_rsrc,
      {PointerType::get(Ctx.C, 8), PointerType::get(Ctx.C, 1)});
  Out.RawPtrRsrc =
      Ctx.B.CreateCall(MakeRsrc,
                       {BasePtr, ConstantInt::get(Type::getInt16Ty(Ctx.C), 0),
                        NumRecords, ConstantInt::get(Ctx.I32Ty, 0x27000)},
                       "mubuf_raw_ptr_rsrc");
  Out.AuxFlags = ConstantInt::get(Ctx.I32Ty, 0);
  return Out;
}

} // namespace COMGR::hotswap
