//===- handle-valu-vcmp.cpp - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-internal.h"

#include "opcode-map.h"
#include "canonical-op-attrs.h"
#include "canonical-op.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR::hotswap {

// SPE attribute registrations. V_CMPX is a compare-and-AND-into-EXEC;
// this handler routes the EXEC mutation through `regs.storeExec` after
// folding the ballot through `WaveProjection::ballotI1ToWidth` -- see
// the V_CMPX branch below. Audit before adding more entries here.
ArrayRef<CanonicalOpAttrSpec> getHandlerValuVcmpAttrs() {
  static constexpr CanonicalOpAttrSpec kAttrs[] = {
      {CanonicalOp::V_CMPX, {/*routesExecThroughStoreExec=*/true}},
  };
  return kAttrs;
}

HandlerResult handleValuVcmp(RaiseContext &Ctx, const DecodedInst &Di,
                               OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;
  switch (Sop) {
  case CanonicalOp::V_CMP:
  case CanonicalOp::V_CMPX:
    break;
  default:
    return Hr;
  }

  StringRef Mn(Di.Mnemonic);

  // The ~100 V_CMP_* and V_CMPX_* MC opcodes collapse onto two SemOps
  // (V_CMP, V_CMPX); the per-opcode metadata (ICmp/FCmp predicate,
  // element width, int/float kind) is looked up via `di.vcmp`,
  // populated at decode time by OpcodeMap. That keeps this handler
  // linear in the number of abstract shapes (2) rather than in the
  // number of AMDGPU opcodes.
  const VCmpMeta *M = Di.Vcmp;
  if (!M) {
    errs() << "transpiler: " << Mn
           << ": V_CMP/V_CMPX reached handler without VCmpMeta "
              "(OpcodeMap::build should have populated it)\n";
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VALU", "V_CMP/V_CMPX reached handler without VCmpMeta");
    return Hr;
  }

  // ---- v_cmp_class_f<bits> / v_cmpx_class_f<bits> ----
  // Special-cased before the generic predicate-compare dispatch
  // because the second operand is an i32 mask of FP classes, NOT a
  // value to compare against. Lifts to `llvm.amdgcn.class.f<bits>`,
  // which yields one i1 per active lane (the wave-mask plumbing is
  // shared with the predicate-compare path below).
  Value *Cmp = nullptr;
  if (M->IsClass) {
    Type *FTy = nullptr;
    Value *Src0 = nullptr;
    if (M->Bits == 16) {
      auto *I16Ty = Type::getInt16Ty(Ctx.C);
      FTy = Type::getHalfTy(Ctx.C);
      Value *Raw0 = Op.srcF(0);
      if (Raw0->getType() != FTy) {
        if (Raw0->getType() != I16Ty)
          Raw0 = Ctx.B.CreateTrunc(Raw0, I16Ty, "vclassf16_lo");
        Raw0 = Ctx.B.CreateBitCast(Raw0, FTy, "vclassf16");
      }
      Src0 = Raw0;
    } else if (M->Bits == 32) {
      FTy = Ctx.F32Ty;
      Value *Raw0 = Op.srcF(0);
      if (Raw0->getType() != FTy)
        Raw0 = Ctx.B.CreateBitCast(Raw0, FTy, "vclassf32");
      Src0 = Raw0;
    } else {
      FTy = Type::getDoubleTy(Ctx.C);
      Src0 = Ctx.B.CreateBitCast(Op.src64(0), FTy, "vclassf64");
    }
    Value *Mask = Op.src(1);
    Function *ClassFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_class, {FTy});
    Cmp = Ctx.B.CreateCall(ClassFn, {Src0, Mask}, "vclass");
    // fall through to the wave-mask write-back / EXEC-AND logic
    // below by reusing the same `cmp`-driven tail.
  }

  // Fetch operands at the correct width. For 64-bit integer compares we
  // read as i64; for 64-bit float compares we read as i64 and bitcast.
  // For 32-bit float compares we read as i32 and bitcast to f32. For
  // 16-bit float compares we read as i32, truncate to i16, and bitcast
  // to half -- required because srcF returns the raw 32-bit operand
  // (e.g. an inline integer immediate -1 = 0xFFFFFFFF for `v_cmpx_lt_
  // f16 vcc, -1, vN`) and CreateFCmp asserts on non-FP operand types.
  Value *S0 = nullptr, *S1 = nullptr;
  if (M->IsClass) {
    // Already lifted above; skip the predicate-operand fetch.
  } else if (M->IsFloat) {
    if (M->Bits == 64) {
      auto *F64Ty = Type::getDoubleTy(Ctx.C);
      S0 = Ctx.B.CreateBitCast(Op.src64(0), F64Ty);
      S1 = Ctx.B.CreateBitCast(Op.src64(1), F64Ty);
    } else if (M->Bits == 32) {
      S0 = Op.srcF(0);
      S1 = Op.srcF(1);
      if (S0->getType() != Ctx.F32Ty)
        S0 = Ctx.B.CreateBitCast(S0, Ctx.F32Ty);
      if (S1->getType() != Ctx.F32Ty)
        S1 = Ctx.B.CreateBitCast(S1, Ctx.F32Ty);
    } else {
      auto *I16Ty = Type::getInt16Ty(Ctx.C);
      auto *F16Ty = Type::getHalfTy(Ctx.C);
      S0 = Op.srcF(0);
      S1 = Op.srcF(1);
      if (S0->getType() != F16Ty) {
        if (S0->getType() != I16Ty)
          S0 = Ctx.B.CreateTrunc(S0, I16Ty, "vcmpf16_lo0");
        S0 = Ctx.B.CreateBitCast(S0, F16Ty, "vcmpf16_a");
      }
      if (S1->getType() != F16Ty) {
        if (S1->getType() != I16Ty)
          S1 = Ctx.B.CreateTrunc(S1, I16Ty, "vcmpf16_lo1");
        S1 = Ctx.B.CreateBitCast(S1, F16Ty, "vcmpf16_b");
      }
    }
  } else {
    if (M->Bits == 64) {
      S0 = Op.src64(0);
      S1 = Op.src64(1);
    } else if (M->Bits == 16) {
      // Integer V_CMP_*_I16/U16 operands live in the low half of the
      // 32-bit VGPR/inline-constant container.  Compare at i16 width so
      // signed predicates see 0xbfce as a negative bf16 bit-pattern, not as
      // the positive i32 value 0x0000bfce.
      auto *I16Ty = Type::getInt16Ty(Ctx.C);
      S0 = Ctx.B.CreateTrunc(Op.src(0), I16Ty, "vcmpi16_lo0");
      S1 = Ctx.B.CreateTrunc(Op.src(1), I16Ty, "vcmpi16_lo1");
    } else {
      S0 = Op.src(0);
      S1 = Op.src(1);
    }
  }
  if (!M->IsClass && (!S0 || !S1)) {
    errs() << "transpiler: " << Mn << ": missing operand\n";
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VALU", "V_CMP/V_CMPX missing operand");
    return Hr;
  }

  if (!M->IsClass)
    Cmp = M->IsFloat ? Ctx.B.CreateFCmp(M->Pred, S0, S1, "vcmpf")
                     : Ctx.B.CreateICmp(M->Pred, S0, S1, "vcmp");

  if (Sop == CanonicalOp::V_CMPX) {
    // Compare-and-exec: result ANDs into EXEC.
    //
    // The mask MUST be materialised as a wave-level ballot, not a
    // per-lane `sext i1`. `sext` on a divergent `cmp` produces a
    // divergent SSA value: each target lane writes its own private
    // all-ones-or-zero into the EXEC slot, and every subsequent read
    // (notably `emitLaneActiveBit`'s `lshr %exec, %lane_mod`) sees a
    // per-lane "EXEC" instead of the single wave-level mask the SPE
    // model requires. The backend then lowers the SPE diamond as a
    // divergent branch on a per-lane value, narrowing hardware EXEC
    // based on the wrong bit entirely -- which surfaces as stores
    // going missing on half the wave in cross-wave lifts (gfx1250
    // wave32 -> gfx942 wave64). Routing through `ballotI1ToWidth`
    // matches the VCC read path (`readVCCAsWaveMask`) and keeps EXEC
    // wave-uniform.
    //
    // Width choice. The ballot result feeds directly into the EXEC
    // alloca via AND, so we request it at the EXEC *storage* width
    // (`execTy`). Under modulo-replication `execTy` equals the
    // source wave-mask width and the projection truncates the
    // hardware ballot to match. Under wave-native cross-widening
    // (wave32 source -> wave64 target) `execTy` equals the full
    // hardware wave mask (i64) and no truncation occurs -- which is
    // what allows a data-dependent `v_cmpx` to preserve its per-
    // target-lane answer on lanes 32..63. See
    // `lit_tests/v_cmpx_ballot` for the pinned IR shape (MODREP)
    // and `lit_tests/v_cmpx_wave_native` for the wave-native shape.
    Value *Mask = Ctx.Projection.ballotI1ToWidth(Ctx.B, Cmp,
                                                  Ctx.Regs.ExecTy,
                                                  "cmpx_ballot");
    Value *CurExec = Ctx.Regs.loadExec(Ctx.B);
    Ctx.Regs.storeExec(Ctx.B, Ctx.B.CreateAnd(CurExec, Mask, "cmpx_exec"));
  } else {
    // Vanilla V_CMP: write to SGPR destination (e64 with sdst) or
    // VCC (e32, or e64 whose sdst is VCC).
    if (Di.NumDefs >= 1) {
      ParsedReg D = Op.dst();
      if (D.RegKind == ParsedReg::SGPR) {
        // Same ballot discipline as V_CMPX: the SGPR destination
        // carries a wave-level mask, not a per-lane predicate. `sext`
        // here would make every downstream consumer that reads the
        // SGPR as a wave mask (`s_and_b64`, `s_mov_b64 exec, …`,
        // `v_cndmask_b32`'s mask input via `readVCCAsWaveMask`) see
        // divergent SSA and silently miscompile.
        //
        // Width choice. The destination is a single SGPR (wave32
        // source) or an SGPR pair (wave64 source) -- i.e. *source*
        // wave-mask width, not EXEC storage width. Under modulo-
        // replication these match; under wave-native cross-
        // widening they diverge (execTy=i64 vs sourceWaveMaskTy=
        // i32), and the SGPR physically cannot hold the 64-bit
        // hardware ballot, so we ask the projection for the
        // narrower width explicitly. That takes the trunc-to-
        // source-width branch in
        // `WaveNativeProjection::ballotI1ToWidth`, a documented
        // residual lossy path whose in-BB correctness is restored
        // by the V_CMP -> V_CNDMASK per-lane-i1 shadow recorded
        // below (see `ctx.recordSgprWaveMaskI1`), and whose out-of-
        // BB / scalar-interleaved / other-consumer cases remain the
        // obstruction classifier's responsibility to refuse
        // (wave-size-obstruction.cpp).
        Type *SourceWidth =
            (Ctx.Projection.sourceWaveScopedLaneOps() && D.WidthInDwords >= 2)
                ? Ctx.I64Ty
                : Ctx.Projection.sourceWaveMaskTy();
        Value *Mask = Ctx.Projection.ballotI1ToWidth(
            Ctx.B, Cmp, SourceWidth, "vcmp_ballot");
        Ctx.writeRegExecWidth(D, Mask);

        // Cache the per-lane `i1` alongside the narrow wave-mask
        // store. The V_CNDMASK_B32 SGPR-source arm in
        // handle-valu-vop3p.cpp looks this up by baseIdx and
        // bypasses the lossy `extractLaneBitFromWaveMask` round-
        // trip when a consumer in the same BB reads the SGPR before
        // any intervening scalar write clobbers it (the latter
        // invalidates via `AllocaRegFile::onSgprWritten` ->
        // `ctx.invalidateSgprWaveMaskI1`). The low-level
        // `storeSGPR*` inside `writeRegExecWidth` above already
        // fired the invalidation hook for this baseIdx; this call
        // restores the fresh `i1` SSA value in the same step. See
        // sgpr-wave-mask-translation.md section 3.1 for the full
        // invariants.
        //
        // Covers BOTH predicate compares (the asin / libdevice-math
        // branch shape) AND class compares
        // (v_cmp_class_f{16,32,64}). The `cmp` value is the same
        // per-lane i1 shape in both arms of this handler -- `fcmp`
        // for the predicate-compare path, `llvm.amdgcn.class.f*`
        // for the class path -- so caching is sound either way.
        // Gating only the predicate arm would leave class compares
        // under cross-widening miscompiling through the lossy
        // extract fallback for no reason.
        //
        // `isPair` is derived from the destination's ParsedReg
        // width: wave64-source V_CMP_e64 writes an SGPR pair
        // (d.width == 2); wave32-source writes a single SGPR
        // (d.width == 1). The flag is consulted by
        // `ctx.invalidateSgprWaveMaskI1` to decide whether a
        // subsequent write to baseIdx+1 clobbers the pair's high
        // half and should invalidate this entry.
        Ctx.recordSgprWaveMaskI1(D.BaseIdx, Cmp,
                                 /*isPair=*/D.WidthInDwords >= 2);
      } else if (D.RegKind == ParsedReg::VCC_HI_SCRATCH ||
                 D.RegKind == ParsedReg::EXEC_HI_SCRATCH) {
        // Wave32-source vcc_hi / exec_hi are free general-purpose wave-mask
        // scalars (see ParsedReg::VCC_HI_SCRATCH). A V_CMP naming one as its
        // destination deposits the ballot into that scratch slot, like the SGPR
        // arm above, so a downstream consumer reads it as a wave mask via
        // extractLaneBitFromWaveMask. The per-lane shadow is not keyed for the
        // scratch slots, so this restores the wave-mask value (uniform
        // compares); per-lane compares into vcc_hi remain the obstruction
        // classifier's domain, as on the SGPR trunc path.
        Value *Mask = Ctx.Projection.ballotI1ToWidth(
            Ctx.B, Cmp, Ctx.Projection.sourceWaveMaskTy(), "vcmp_vcchi_ballot");
        Ctx.Regs.writeReg32(Ctx.B, D, Mask);
      } else {
        Ctx.Regs.storeVCC(Ctx.B, Cmp);
      }
    } else {
      Ctx.Regs.storeVCC(Ctx.B, Cmp);
    }
  }
  Hr.Handled = true;
  return Hr;
}

} // namespace COMGR::hotswap
