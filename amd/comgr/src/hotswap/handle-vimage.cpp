//===- handle-vimage.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// VIMAGE TENSOR handler -- gfx1250-only tensor-descriptor memory ops.
//
// Covers the four `VIMAGE_TENSOR_Pseudo` instructions defined in
// `MIMGInstructions.td:2049-2113`:
//
//   * `tensor_load_to_lds_d2`    / `tensor_load_to_lds_d4`
//   * `tensor_store_from_lds_d2` / `tensor_store_from_lds_d4`
//
// Both `_d2` (up-to-2D) and `_d4` (up-to-4D) variants share a CanonicalOp
// (`TENSOR_LOAD_TO_LDS` / `TENSOR_STORE_FROM_LDS`); see the
// docstrings in `canonical-op.h` and the canonicalization entries in
// `opcode-map.cpp`. The form is recovered from `op.nSrcs()`: the
// pseudo's InOperandList has `vaddr0, vaddr1, r128, cpol` (4) for
// `_d2` and `vaddr0, vaddr1, vaddr2, vaddr3, r128, cpol` (6) for
// `_d4`; both `_d{2,4}_gfx1250` Reals inherit the same operand list
// (`MIMGInstructions.td:2087`), so this count is stable across the
// MC -> pseudo collapse `OpcodeMap::canonicalize` performs.
//
// === Same-target contract (gfx1250 -> gfx1250) ===
//
// When the compilation target is itself gfx1250 (TENSORcnt unit
// available, `ISAProfile::hasTensorOps`), the principled lift is a
// direct call to the matching LLVM intrinsic
// (`llvm.amdgcn.tensor.{load.to.lds,store.from.lds}`,
// IntrinsicsAMDGPU.td:4213). The intrinsic's signature mirrors the
// hardware operand bank exactly:
//
//   void int_amdgcn_tensor_load_to_lds(<4 x i32> grp0,
//                                       <8 x i32> grp1,
//                                       <4 x i32> grp2,
//                                       <4 x i32> grp3,
//                                       <8 x i32> grp4_reserved,
//                                       i32 cachepolicy)
//
// We marshal each SReg_128/SReg_256 source into the matching
// `<N x i32>` by reading consecutive dwords via `regs.loadSGPR32`
// and packing them into a vector with `CreateInsertElement`. Group 4
// is hardcoded to <8 x i32> zeroinitializer per the IntrinsicsAMDGPU
// docstring ("reserved for future targets, use zeroinitializer for
// now"); for the `_d2` form the unused groups 2 and 3 are also
// passed as <4 x i32> zeroinitializer (the MC `_d2` encoding pins
// `vaddr2`/`vaddr3` to the NULL SGPR sentinel via
// `MIMGInstructions.td:2099-2100`, which the disassembler does not
// surface as operands). The `r128` immediate is consumed by the
// hardware encoding and is not part of the intrinsic's argument
// vector -- see IntrinsicsAMDGPU.td:4197-4211.
//
// === Cross-target contract (gfx1250 -> gfx942 and earlier) ===
//
// On gfx942 (and every pre-gfx1250 target) there is no equivalent
// hardware unit: the TENSORcnt register and the `TENSOR_CNT` TSFlags
// bit live under `let SubtargetPredicate = isGFX125xOnly` in TableGen,
// and the matching LLVM intrinsics are themselves gated `isGFX125xOnly`
// so a cross-target intrinsic emit would fail at codegen on a non-
// gfx1250 backend.
//
// We provide functional emulation by linking a HIP-authored device
// runtime (`runtime/tdm.hip`) into the raised IR module. The handler
// emits a call to `hotswap_tdm_load_to_lds` / `hotswap_tdm_store_from_lds`
// with the same operand vectors the same-target intrinsic emit
// produces, plus the source wave size; the link merge happens in
// `raiseToIR` (see `tdm-runtime.h` and `raiser.cpp`). The helper
// stripes the descriptor's innermost X dimension across source-wave-local
// lanes and implements the full D# walk (4D/5D loops, OOB rules, padding,
// iteration, gather mode, atomic-barrier side effect).
//
// When the transpiler was built without hipcc, `tdmRuntimeAvailable()`
// is false and this handler keeps the pre-existing loud refusal path
// -- the bucket / format name / formatName(VIMAGE) summary stays
// stable for diagnostics.

#include "handlers.h"

#include "decoded-inst.h"
#include "isa-profile.h"
#include "parsed-reg.h"
#include "raise-context.h"
#include "raise-failure.h"
#include "reg-file.h"
#include "canonical-op.h"
#include "tdm-runtime.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Marshal `n` consecutive SGPR dwords starting at `base.BaseIdx` into
// an `<n x i32>` vector. `base.RegKind` MUST be `ParsedReg::SGPR` --
// callers that accept the SReg_128_XNULL `null` sentinel must
// short-circuit to a zero vector before reaching this helper.
// `n` is hardcoded by the caller from the operand class (4 for
// SReg_128 / D# group 0,2,3; 8 for SReg_256 / D# group 1) per the
// pseudo's InOperandList in MIMGInstructions.td:2073.
Value *marshalSgprGroup(RaiseContext &Ctx, ParsedReg Base, unsigned N,
                        const Twine &Name) {
  auto *VecTy = FixedVectorType::get(Ctx.I32Ty, N);
  Value *Vec = PoisonValue::get(VecTy);
  for (unsigned I = 0; I < N; ++I) {
    Value *Dword = Ctx.Regs.loadSGPR32(Ctx.B, Base.BaseIdx + static_cast<int>(I));
    Vec = Ctx.B.CreateInsertElement(Vec, Dword, I, Name);
  }
  return Vec;
}

// Build a `<n x i32> zeroinitializer` for groups the form does not
// supply (group 4 is always reserved; groups 2 and 3 are unused
// in the `_d2` form per `MIMGInstructions.td:2099-2100`).
Value *zeroVec(RaiseContext &Ctx, unsigned N) {
  auto *VecTy = FixedVectorType::get(Ctx.I32Ty, N);
  return ConstantAggregateZero::get(VecTy);
}

// Read an immediate operand and zero-extend it to i32. The intrinsic
// signature carries `cachepolicy` as i32 with `ImmArg<ArgIndex<5>>`
// so the value MUST be a constant -- `op.srcImm` returns the decoded
// integer directly, sidestepping any operand-read divergence path.
Value *cpolImm(RaiseContext &Ctx, OpResolver &Op, unsigned CpolIdx) {
  return ConstantInt::get(Ctx.I32Ty, static_cast<uint32_t>(Op.srcImm(CpolIdx)));
}

// Six-argument bundle matching the gfx1250 intrinsic operand bank.
// Same shape feeds the same-target intrinsic emit and the cross-target
// helper call (see `tdm-runtime.h`).
struct TDMArgs {
  Value *Grp0 = nullptr;
  Value *Grp1 = nullptr;
  Value *Grp2 = nullptr;
  Value *Grp3 = nullptr;
  Value *Grp4 = nullptr;
  Value *Cpol = nullptr;
};

// Marshal the SGPR-bank operand list of a `tensor_{load,store}_*_d{2,4}`
// pseudo into the six argument values both lowering paths take. On
// success, populates `out` and returns true. On any operand-shape
// rejection, populates `hr.Failure` and returns false.
bool marshalTDMArgs(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op,
                    HandlerResult &Hr, TDMArgs &Out) {
  // Recover the operand-shape variant. The pseudo's InOperandList
  // (MIMGInstructions.td:2073) has 4 sources for `_d2`
  // (vaddr0, vaddr1, r128, cpol) and 6 for `_d4` (vaddr0..vaddr3,
  // r128, cpol); both `_d{2,4}_gfx1250` Reals inherit the same
  // operand list at MIMGInstructions.td:2087. Anything else means
  // a future encoding variant landed in LLVM that we have not
  // audited -- refuse loudly so the drift surfaces immediately.
  const unsigned Nsrcs = Op.nSrcs();
  if (Nsrcs != 4 && Nsrcs != 6) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VIMAGE",
        Twine("unexpected source operand count ") + Twine(Nsrcs) +
            " for tensor op (expected 4 for _d2 or 6 for _d4)");
    return false;
  }
  const bool IsD2 = (Nsrcs == 4);

  // Vaddr operands are required to be real SGPR ranges. The
  // SReg_128_XNULL/SReg_256_XNULL operand classes nominally
  // permit the NULL sentinel, but a NULL D# pointer would mean
  // the kernel has no Tensor Descriptor to walk -- there is no
  // sensible lowering that preserves observed behaviour. We
  // intentionally do NOT cross-check `pr.width` against the
  // operand-class width: `computeRegWidth32` (raise-context.cpp:66)
  // walks the disassembler-supplied sub-reg chain, and AMDGPU's
  // SReg_*_XNULL tuple classes sometimes report a width smaller
  // than the operand's nominal dword count when the high lanes
  // alias an aggregate sub-reg index. The hardware encoding still
  // reads `n` consecutive SGPRs starting at `baseIdx` regardless
  // of how the tuple chain is named, so reading via baseIdx is
  // the source of truth -- it matches the decode of the encoded
  // 8-bit SGPR pointer field exactly.
  auto RequireSgpr = [&](ParsedReg Pr, const char *Role) -> bool {
    if (Pr.RegKind != ParsedReg::SGPR) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VIMAGE",
          Twine("tensor ") + Role + " must be a contiguous SGPR range "
                "(got non-SGPR operand kind)");
      return false;
    }
    return true;
  };

  ParsedReg Vaddr0 = Op.srcReg(0);
  ParsedReg Vaddr1 = Op.srcReg(1);
  if (!RequireSgpr(Vaddr0, "vaddr0/D# group 0") ||
      !RequireSgpr(Vaddr1, "vaddr1/D# group 1"))
    return false;

  Out.Grp0 = marshalSgprGroup(Ctx, Vaddr0, 4, "td_grp0");
  Out.Grp1 = marshalSgprGroup(Ctx, Vaddr1, 8, "td_grp1");
  if (IsD2) {
    Out.Grp2 = zeroVec(Ctx, 4);
    Out.Grp3 = zeroVec(Ctx, 4);
    Out.Cpol = cpolImm(Ctx, Op, 3);
  } else {
    ParsedReg Vaddr2 = Op.srcReg(2);
    ParsedReg Vaddr3 = Op.srcReg(3);
    if (!RequireSgpr(Vaddr2, "vaddr2/D# group 2") ||
        !RequireSgpr(Vaddr3, "vaddr3/D# group 3"))
      return false;
    Out.Grp2 = marshalSgprGroup(Ctx, Vaddr2, 4, "td_grp2");
    Out.Grp3 = marshalSgprGroup(Ctx, Vaddr3, 4, "td_grp3");
    Out.Cpol = cpolImm(Ctx, Op, 5);
  }
  Out.Grp4 = zeroVec(Ctx, 8); // reserved for future targets
  return true;
}

} // namespace

HandlerResult handleVIMAGE(RaiseContext &Ctx, const DecodedInst &Di,
                           OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  // Only the two TENSOR SemOps reach this handler; anything else in
  // the VIMAGE format family (e.g. future image_load/image_sample
  // ops if we ever lift them) would fall through with `handled =
  // false` so the main raiser loop reports the canonical
  // `UnsupportedOpcode [VIMAGE]` diagnostic. Intentionally do NOT
  // synthesise a generic refusal here -- that would mask new VIMAGE
  // members the kerneldex sweep surfaces in the future.
  if (Sop != CanonicalOp::TENSOR_LOAD_TO_LDS &&
      Sop != CanonicalOp::TENSOR_STORE_FROM_LDS) {
    return Hr;
  }

  TDMArgs Args;
  if (!marshalTDMArgs(Ctx, Di, Op, Hr, Args))
    return Hr;

  // Same-target gfx1250 -> gfx1250 intrinsic lift.
  if (Ctx.TargetIsa.HasTensorOps) {
    Intrinsic::ID Iid = (Sop == CanonicalOp::TENSOR_LOAD_TO_LDS)
                            ? Intrinsic::amdgcn_tensor_load_to_lds
                            : Intrinsic::amdgcn_tensor_store_from_lds;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Iid);
    Ctx.B.CreateCall(
        Fn, {Args.Grp0, Args.Grp1, Args.Grp2, Args.Grp3, Args.Grp4, Args.Cpol});
    Hr.Handled = true;
    return Hr;
  }

  // Cross-target (gfx1250 -> gfx942 and earlier) emulation via the
  // HIP-authored runtime helper link-merged into this module by
  // `linkTDMRuntime` in `raiser.cpp`. When the transpiler was built
  // without hipcc the embedded bitcode is empty and we keep the
  // pre-existing loud refusal so the no-hipcc build behaves exactly
  // as it did before this path landed.
  if (!tdmRuntimeAvailable()) {
    llvm::errs()
        << "transpiler: VIMAGE: " << Di.Mnemonic
        << " has no equivalent on the compilation target "
        << "(gfx1250 TENSORcnt unit; LLVM intrinsic "
        << (Sop == CanonicalOp::TENSOR_LOAD_TO_LDS
                ? "amdgcn.tensor.load.to.lds"
                : "amdgcn.tensor.store.from.lds")
        << " is gated isGFX125xOnly) and the TDM emulation runtime is "
           "unavailable (transpiler was built without hipcc).\n";
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VIMAGE",
        "gfx1250-only TENSOR cnt op; no equivalent on non-gfx1250 "
        "compilation target and TDM emulation runtime unavailable");
    return Hr;
  }

  const unsigned SourceWaveSize = Ctx.Isa.WaveSize;
  const unsigned TargetWaveSize = Ctx.TargetIsa.WaveSize;
  const bool SupportedWaveShape =
      SourceWaveSize == TargetWaveSize ||
      (SourceWaveSize == 32 && TargetWaveSize == 64);
  if (!SupportedWaveShape) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VIMAGE",
        Twine("TDM emulation supports only source-wave-local same-wave "
              "execution or wave32 source -> wave64 target cross-widening "
              "(got source wave ") +
            Twine(SourceWaveSize) + ", target wave " + Twine(TargetWaveSize) +
            ")");
    return Hr;
  }

  FunctionCallee Helper = (Sop == CanonicalOp::TENSOR_LOAD_TO_LDS)
                              ? declareTDMLoad(Ctx.M)
                              : declareTDMStore(Ctx.M);
  // The runtime helper's signature is the four D# groups plus the source
  // wave size. It deliberately does NOT take the intrinsic's trailing
  // `<8 x i32> grp4` because that group is reserved by the gfx1250
  // intrinsic contract, and it does not take `i32 cpol` because the
  // cross-target helper has no target cache-policy encoding to preserve.
  // The descriptor-visible side effects, including atomic-barrier updates,
  // live in the D# groups that are forwarded. `sourceWaveSize` keeps the
  // helper's descriptor readfirstlane / X stripe source-wave-local when
  // WaveNativeProjection packs two source wave32s into one target wave64.
  Ctx.emitUnderExec([&] {
    Ctx.B.CreateCall(Helper, {Args.Grp0, Args.Grp1, Args.Grp2, Args.Grp3,
                              Ctx.B.getInt32(SourceWaveSize)});
  });
  Hr.Handled = true;
  return Hr;
}

} // namespace COMGR::hotswap
