//===- handle-smem.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"
#include "pipeline.h" // isStrictMode()
#include "source-hidden-args.h"

#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

#define DEBUG_TYPE "transpiler"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Look up the SGPR index that holds the *low* dword of the source-ISA
// KernargSegmentPtr at kernel entry, via `ctx.Layout`.
//
// Used by the dword-granular S_LOAD_B* block and the narrow-SMEM
// (S_LOAD_U8/I8/U16/I16) block to gate the implicit-args reroute on
// "is this load's sbase the kernarg pair?" The previous implementation
// hardcoded `baseIdx == 0`, which broke as soon as a kernel enabled
// PrivateSegmentBuffer (4 dwords), DispatchPtr (2 dwords), or QueuePtr
// (2 dwords) ahead of the kernarg pointer in the canonical
// enable_sgpr_* order -- the kernarg pair then slides up to s[2:3],
// s[6:7], s[8:9], etc.
//
// The layout object is the single source of truth for the source
// ISA's user-SGPR ABI. It is populated by
// `UserSgprLayout::fromKernelMeta` (which itself aborts loudly if the
// kernel descriptor is missing, so we never fall back to a guessed
// layout), and wired into `RaiseContext` before handler dispatch in
// raiser.cpp. A null pointer here therefore means the raiser failed
// to wire the layout into the context -- a wiring bug, not a runtime
// condition -- and we surface it with `report_fatal_error`.
//
// Returns -1 if KernargSegmentPtr is disabled in the KD (no corpus
// kernel today, but the caller must still guard against the `-1`
// match to avoid a false-positive "is kernarg" on negative sbase
// indices).
int getKernargPtrSgpr(RaiseContext &Ctx) {
  if (Ctx.Layout == nullptr)
    llvm::report_fatal_error(
        "transpiler: handle_smem: RaiseContext::userSgprLayout is null. "
        "The raiser must populate this before dispatching to handlers; "
        "missing wiring is a bug.");
  return Ctx.Layout->KernargSegmentPtrSgpr;
}

} // namespace

HandlerResult handleSMEM(RaiseContext &Ctx, const DecodedInst &Di,
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
    int KernargPtrSgpr = getKernargPtrSgpr(Ctx);
    bool BaseIsKernargPair =
        (Base.RegKind == ParsedReg::SGPR && KernargPtrSgpr >= 0 &&
         Base.BaseIdx == KernargPtrSgpr);

    // Implicit-args reroute. AMDGPU separates the explicit kernarg
    // segment from the implicit-arg block: the latter is reachable via
    // `amdgcn_implicitarg_ptr`, not via offsets past the end of the
    // kernarg segment. A source kernel that issues
    // `s_load_b* sN, kernarg_pair, off` with `off >= implicitArgsBase`
    // is reading hidden args through the source-ABI flat layout; the
    // lifted kernel must materialise those bytes via the implicit-arg
    // pointer with the offset rebased to `off - implicitArgsBase`.
    //
    // Strict-mode refusal: in `HSA_HOTSWAP_STRICT=1` the cross-arch
    // implicit-arg layout is not yet proven equivalent for every
    // `(source ISA, target ISA)` pair we lift between, so the
    // pipeline refuses to silently substitute a target-ABI implicit
    // arg for a source-ABI one. In permissive mode we trust the
    // ROCm convention that the layouts match (both gfx9-12 follow
    // the same `hidden_*` block).
    //
    // Gating: `baseIsKernargPair` (literal SGPR-index match against
    // the source-ABI kernarg pair) + `immOffset` + a positive
    // `implicitArgsBase`. We deliberately do NOT track whether the
    // pair has been mutated since entry: corpus shapes that overwrite
    // the pair (Triton/SGLang `s[0:1] = preloaded_ptr + wg_offset`,
    // Tensile UniversalArgs `+16` shift) only issue follow-up loads at
    // small offsets that fall well below `implicitArgsBase`, so the
    // gate is precise enough in practice.
    if (BaseIsKernargPair && ImmOffset &&
        Ctx.Kernargs.ImplicitArgsBase > 0 &&
        ByteOffset >= Ctx.Kernargs.ImplicitArgsBase) {
      if (isStrictMode()) {
        Hr.Failure = RaiseFailure::strictUnsafeLowering(
            Di, "implicitarg.ptr",
            "cross-arch implicitarg.ptr lowering is unresolved: source "
            "implicit-arg offsets are being applied to the target runtime "
            "hidden-arg block");
        return Hr;
      }
      SourceHiddenArgContext HiddenCtx{Ctx.C,      Ctx.M,      Ctx.B,
                                       Ctx.I8Ty,   Ctx.I32Ty,  Ctx.I64Ty,
                                       Ctx.Kernargs.Args};
      SourceHiddenArgValue HiddenBase =
          emitSourceHiddenDword(HiddenCtx, ByteOffset);
      if (!HiddenBase.Matched) {
        Function *FnImplicitArgPtr = Intrinsic::getOrInsertDeclaration(
            &Ctx.M, Intrinsic::amdgcn_implicitarg_ptr);
        Value *ImplPtr =
            Ctx.B.CreateCall(FnImplicitArgPtr, {}, "implicitarg_ptr");
        int64_t ImplOffset = ByteOffset - Ctx.Kernargs.ImplicitArgsBase;
        Value *Gep =
            (ImplOffset == 0)
                         ? ImplPtr
                         : Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, ImplPtr,
                                                   Ctx.B.getInt64(ImplOffset),
                                                   "impl_gep");
        for (int D = 0; D < LoadDwords; D++) {
          Value *Ep = (D == 0) ? Gep
                               : Ctx.B.CreateInBoundsGEP(
                                     Ctx.I8Ty, Gep, Ctx.B.getInt64(D * 4));
          Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx + D,
                               Ctx.B.CreateLoad(Ctx.I32Ty, Ep, "impl_load"));
        }
        Hr.Handled = true;
        return Hr;
      }
      if (!HiddenBase.Value) {
        if (HiddenBase.FailureDetail.empty())
          report_fatal_error(
              "matched source hidden argument without value or failure detail");
        Hr.Failure = RaiseFailure::unsupportedSourceHiddenArg(
            Di, "SMEM", HiddenBase.FailureDetail);
        return Hr;
      }
      for (int D = 0; D < LoadDwords; D++) {
        SourceHiddenArgValue Dw =
            D == 0 ? HiddenBase
                   : emitSourceHiddenDword(HiddenCtx, ByteOffset + D * 4);
        if (!Dw.Matched) {
          Hr.Failure = RaiseFailure::unsupportedInstructionForm(
              Di, "SMEM", "source hidden-arg SMEM load spans non-hidden bytes");
          return Hr;
        }
        if (!Dw.Value) {
          if (Dw.FailureDetail.empty())
            report_fatal_error("matched source hidden argument without value "
                               "or failure detail");
          Hr.Failure = RaiseFailure::unsupportedSourceHiddenArg(
              Di, "SMEM", Dw.FailureDetail);
          return Hr;
        }
        Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx + D, Dw.Value);
      }
      Hr.Handled = true;
      return Hr;
    }

    // Generic GEP+load against `addrspace(1)`. The AMDGPU backend
    // re-derives uniformity / addrspace-narrowing during lowering: a
    // load whose pointer is provably from `amdgcn_kernarg_segment_ptr`
    // is selected as `s_load_*` against the kernarg segment regardless
    // of the IR-level addrspace cast; for runtime-mutated bases (Triton/
    // SGLang `s[0:1] = preloaded_ptr + wg_offset`, Tensile HBMArgs
    // `s_load_b64 s[0:1], s[0:1], 0x10`, etc.) the backend keeps the
    // VMEM lowering. The lift no longer hand-picks the addrspace --
    // tracking pointer provenance at lift time was redundant with the
    // backend's own analysis.
    {
      Value *BaseAddr = Ctx.Regs.loadSGPR64(Ctx.B, Base.BaseIdx);
      Value *Ptr = Ctx.B.CreateIntToPtr(BaseAddr, Ctx.PtrGlobalTy);
      if (ImmOffset) {
        if (ByteOffset != 0)
          Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, Ctx.B.getInt64(ByteOffset));
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
          RegOff = Ctx.B.CreateMul(RegOff,
                                   ConstantInt::get(Ctx.I64Ty, LoadBytes),
                              "smem_roff_scaled");
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

    int KernargPtrSgpr = getKernargPtrSgpr(Ctx);
    bool BaseIsKernargPair =
        (Base.RegKind == ParsedReg::SGPR && KernargPtrSgpr >= 0 &&
         Base.BaseIdx == KernargPtrSgpr);
    Value *BaseAddr = Ctx.Regs.loadSGPR64(Ctx.B, Base.BaseIdx);
    Value *Ptr = Ctx.B.CreateIntToPtr(BaseAddr, Ctx.PtrGlobalTy);
    unsigned OffIdx = Op.srcIdx(1);
    if (Di.isImm(OffIdx)) {
      int64_t Off = Op.srcImm(1);
      if (BaseIsKernargPair && Ctx.Kernargs.ImplicitArgsBase > 0 &&
          Off >= Ctx.Kernargs.ImplicitArgsBase) {
        SourceHiddenArgContext HiddenCtx{Ctx.C,      Ctx.M,      Ctx.B,
                                         Ctx.I8Ty,   Ctx.I32Ty,  Ctx.I64Ty,
                                         Ctx.Kernargs.Args};
        SourceHiddenArgValue Hidden = emitSourceHiddenInteger(
            HiddenCtx, static_cast<int>(Off), IsHalfWord ? 2 : 1, IsSigned);
        if (Hidden.Matched) {
          if (!Hidden.Value) {
            if (Hidden.FailureDetail.empty())
              report_fatal_error("matched source hidden argument without value "
                                 "or failure detail");
            Hr.Failure = RaiseFailure::unsupportedSourceHiddenArg(
                Di, "SMEM", Hidden.FailureDetail);
            return Hr;
          }
          Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx, Hidden.Value);
          Hr.Handled = true;
          return Hr;
        }
      }
      if (Off != 0)
        Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, Ctx.B.getInt64(Off));
    } else {
      // Narrow SMEM element size for `scale_offset`: 1B for byte,
      // 2B for halfword. Same SCAL-scales-the-SGPR-offset rule as
      // the dword family above.
      int NarrowBytes = IsHalfWord ? 2 : 1;
      Value *RegOff = Ctx.B.CreateZExt(Op.src(1), Ctx.I64Ty, "smem_nroff");
      if (Di.HasScaleOffset && NarrowBytes != 1)
        RegOff = Ctx.B.CreateMul(RegOff,
                                 ConstantInt::get(Ctx.I64Ty, NarrowBytes),
                            "smem_nroff_scaled");
      Ptr = Ctx.B.CreateInBoundsGEP(Ctx.I8Ty, Ptr, RegOff);
    }

    Value *Narrow = Ctx.B.CreateAlignedLoad(NarrowTy, Ptr, NarrowAlign,
                                             NarrowLoadName);
    Value *Ext = IsSigned
                     ? Ctx.B.CreateSExt(Narrow, Ctx.I32Ty, ExtName)
                          : Ctx.B.CreateZExt(Narrow, Ctx.I32Ty, ExtName);
    Ctx.Regs.storeSGPR32(Ctx.B, Dest.BaseIdx, Ext);
    Hr.Handled = true;
    return Hr;
  }

  // s_store_* (scalar store through SGPR base + imm/sgpr offset).
  // MC operand layout: (sdata, sbase, soffset/imm, cpol).
  if (Sop == CanonicalOp::S_STORE_B32 || Sop == CanonicalOp::S_STORE_B64 ||
      Sop == CanonicalOp::S_STORE_B128) {
    int StoreDwords = (Sop == CanonicalOp::S_STORE_B32)  ? 1
                      : (Sop == CanonicalOp::S_STORE_B64) ? 2
                                                          : 4;
    ParsedReg Data = Op.srcReg(0);
    ParsedReg Base = Op.srcReg(1);
    if (Data.RegKind != ParsedReg::SGPR || Base.RegKind != ParsedReg::SGPR) {
      llvm::errs() << "transpiler: " << Di.Mnemonic
                   << ": S_STORE expects SGPR data and base\n";
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "SMEM", "S_STORE expects SGPR data and base");
      return Hr;
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
          RegOff = Ctx.B.CreateMul(RegOff,
                                   ConstantInt::get(Ctx.I64Ty, StoreBytes),
                              "smem_st_roff_scaled");
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
  ParsedReg DataDst;  // Set only on the RTN arm.
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
    Value *RegOff = Ctx.B.CreateZExt(Op.src(OffSrcPos), Ctx.I64Ty,
                                      "smem_at_roff");
    if (Di.HasScaleOffset)
      RegOff = Ctx.B.CreateMul(RegOff, ConstantInt::get(Ctx.I64Ty, 4),
                               "smem_at_roff_scaled");
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
