//===- c5-predicate-chain-classifier.cpp - Hotswap transpiler -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "c5-predicate-chain-classifier.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// ============================================================================
// Narrow-O1 classifier -- see hotswap/docs/modrep-predicate-chain.md sec. 5
// (O1) (sec. 5 O1 "narrow-O1, as landed" documents the narrowing rationale).
// Two-pass design:
//
//   Pass 1: forward-walk from each `@llvm.amdgcn.workitem.id.x()` call,
//           tagging every tid-reachable value with its masked-or-unmasked
//           state (a value reaches `unmaskedVisited` iff there is a walk
//           from some tid call that did not cross an `and V, K<=W_s-1`
//           mask; `maskedVisited` iff there is a walk that did).
//
//   Pass 2: scan every `icmp` in the function. Refuse iff the icmp has
//           BOTH (a) at least one operand in `unmaskedVisited` AND (b) at
//           least one operand that is a compile-time constant K in
//           `(0, W_s-1]`.
//
// Why the two passes -- and NOT the widened "unmasked + masked" rule.
// The two-pass structure catches `icmp tid, 15` and `icmp f(tid)_unmasked,
// 15` shapes where the icmp's tid-derived operand arrives via a
// propagator chain that does NOT include the small-K constant check as
// its sibling (a single forward pass with an `otherOperand(cmp, V)`
// test happened to catch these shapes too, but only because the walker
// visits both operands if both are tid-reachable -- the two-pass form
// makes the operand scan explicit and decoupled from the walk order).
//
// An earlier iteration of this file ALSO refused the cross-subtree shape
// `icmp f(tid)_unmasked, g(tid)_masked` on the theory that
// replica-0's `(f(L), g(L))` and replica-1's `(f(L+W_s), g(L+W_s))`
// must diverge because `g` is masked. **That theory is falsified** by
// the compare_correctness hotswap-path sweep (2026-04-21): baselines
// that were MATCH under MODREP pre-widen (`vecadd_f16`,
// `corpus_add_fp32`, `corpus_asin_fp32`, `canary_dpp_reduce_fp32`)
// all emit an `icmp sgt tid-unmasked, tid-masked` pattern (Triton's
// bounds-check idiom against a tid-derived end bound), and their
// runtime numerics show the icmp evaluates identically across
// replicas for the predicate families that actually arise (`sgt` /
// `ugt` / `slt` / `ult`). The divergence DOES occur for `eq` / `ne`
// -- e.g. `icmp eq tid, (tid&15)` is lane-position-scoped -- but the
// classifier cannot see the predicate-specific mathematics without
// per-predicate value-range reasoning, and the safe direction is to
// leave the baseline shapes un-refused. If a future recipe surfaces
// a genuine `eq`/`ne` cross-subtree miscompile, the rule gets
// tightened per-predicate rather than across-the-board.
// ============================================================================

// True iff `V` is a non-negative `ConstantInt` whose unsigned value lies
// strictly inside `(0, bound]`. We exclude zero because `icmp %v, 0` is
// a mask-extraction / null-check idiom independent of lane position, and
// clamping to `>= 0` keeps the LLVM "signed constant happens to fit in
// `bound`" case honest (the classifier exists to refuse specifically the
// `tid < 2^s` / `tid < W_s/2` idioms, all of which use positive constants).
bool isNonZeroConstantUleBound(const Value *V, uint64_t Bound) {
  const auto *Ci = dyn_cast<ConstantInt>(V);
  if (!Ci)
    return false;
  const APInt &Val = Ci->getValue();
  if (Val.isNegative())
    return false;
  if (Val.isZero())
    return false;
  return Val.ule(Bound);
}

// Return the "other" operand of a two-operand instruction. Used for
// `and %v, K` shape checks where we already know `%v` is one of the two
// operands and need to inspect the sibling to decide whether the AND
// is a source-wave mask.
Value *otherOperand(const Instruction *I, const Value *V) {
  if (I->getNumOperands() != 2)
    return nullptr;
  Value *A = I->getOperand(0);
  Value *B = I->getOperand(1);
  return (A == V) ? B : A;
}

// True iff `I` is an `and` that AND-masks `V` against a compile-time
// constant K with `K <= sourceWaveSize - 1`. Matches the sec. 5.6.2
// `wave_id` lift's `and X, 0x1F` mask, the SPE prelude's
// `lane_id & (execBits - 1)` mask, and the Triton-emitted
// `offs & (BLOCK_SIZE - 1)` mask when `BLOCK_SIZE <= W_s`.
bool isSourceWaveMaskAnd(const Instruction *I, const Value *V,
                         unsigned SourceWaveSize) {
  const auto *Bop = dyn_cast<BinaryOperator>(I);
  if (!Bop || Bop->getOpcode() != Instruction::And)
    return false;
  const Value *Other = otherOperand(I, V);
  if (!Other)
    return false;
  const auto *Ci = dyn_cast<ConstantInt>(Other);
  if (!Ci)
    return false;
  const APInt &Val = Ci->getValue();
  if (Val.isNegative())
    return false;
  return Val.ule(SourceWaveSize - 1);
}

// True iff `I` is a pure propagator -- an instruction whose result keeps
// `V`'s per-lane divergence. The walker pushes the result onto the
// worklist so downstream uses are inspected.
//
// Intentionally excludes `and` -- that is handled by the specialised
// mask check above and must not fall through to "continue walking the
// result as tid-derived-unmasked" when the `and` is the W_s-1 mask.
// A non-masking `and` (other operand is not a compile-time W_s-1
// constant) still propagates tid-divergence, so the BinaryOperator
// arm below keeps `Instruction::And` listed for that fallthrough.
//
// Also excludes `icmp` (handled separately in Pass 2 -- its result is
// `i1` and is not itself a tid-derived value for this class) and
// memory ops (sinks -- the value was consumed verbatim, no further
// tid-divergence carried by the memory op's own result).
//
// Intrinsic calls (#8 audit): numeric intrinsics like `@llvm.umin`,
// `@llvm.smax`, `@llvm.abs`, `@llvm.ctlz`, `@llvm.ctpop`,
// `@llvm.bitreverse`, `@llvm.fshl` preserve tid-divergence in the
// general case (`umin(tid, 31)` clamps lane-32 to 31 but lane-0 to 0,
// which is STILL divergent between replicas -- not a mask in the
// MODREP sense). Classified as propagators. No intrinsic we've
// observed in Triton / HIP output has the `(mod W_s)` semantics
// required to be classified as a mask; future additions go alongside
// `isSourceWaveMaskAnd` above, not here.
bool isPurePropagator(const Instruction *I) {
  if (isa<CastInst>(I))
    return true;
  if (isa<PHINode>(I))
    return true;
  if (isa<SelectInst>(I))
    return true;
  if (isa<GetElementPtrInst>(I))
    return true;
  if (isa<FreezeInst>(I))
    return true;
  if (isa<ExtractElementInst>(I) || isa<InsertElementInst>(I) ||
      isa<ShuffleVectorInst>(I) || isa<ExtractValueInst>(I) ||
      isa<InsertValueInst>(I))
    return true;
  if (const auto *Bop = dyn_cast<BinaryOperator>(I)) {
    switch (Bop->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::And:
      return true;
    default:
      return false;
    }
  }
  if (isa<UnaryOperator>(I))
    return true;
  if (const auto *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    // Min/max families: numerical clamps, divergence-preserving.
    case Intrinsic::smin:
    case Intrinsic::smax:
    case Intrinsic::umin:
    case Intrinsic::umax:
    // Absolute value / sign manipulation.
    case Intrinsic::abs:
    // Bit-count / bit-reverse / byte-swap: pure functions of the
    // argument's bit pattern; tid's divergence pattern flows
    // through unchanged.
    case Intrinsic::ctlz:
    case Intrinsic::cttz:
    case Intrinsic::ctpop:
    case Intrinsic::bitreverse:
    case Intrinsic::bswap:
    // Funnel shifts: combine two divergent inputs into a divergent
    // result.
    case Intrinsic::fshl:
    case Intrinsic::fshr:
    // Saturating arithmetic: same shape as regular arithmetic for
    // divergence-tracking purposes.
    case Intrinsic::sadd_sat:
    case Intrinsic::ssub_sat:
    case Intrinsic::uadd_sat:
    case Intrinsic::usub_sat:
      return true;
    default:
      return false;
    }
  }
  return false;
}

// Build the refusal-detail string for the diagnostic. Kept short and
// stable so lit fixtures can pin the refusal on substrings without
// tying themselves to exact wording.
std::string formatRefusalDetail(const ICmpInst *Cmp, unsigned SourceWaveSize,
                                const ConstantInt *SmallK) {
  std::string S;
  raw_string_ostream Os(S);
  Os << "`workitem.id.x()` reaches `icmp";
  switch (Cmp->getPredicate()) {
  case ICmpInst::ICMP_EQ:
    Os << " eq";
    break;
  case ICmpInst::ICMP_NE:
    Os << " ne";
    break;
  case ICmpInst::ICMP_ULT:
    Os << " ult";
    break;
  case ICmpInst::ICMP_ULE:
    Os << " ule";
    break;
  case ICmpInst::ICMP_UGT:
    Os << " ugt";
    break;
  case ICmpInst::ICMP_UGE:
    Os << " uge";
    break;
  case ICmpInst::ICMP_SLT:
    Os << " slt";
    break;
  case ICmpInst::ICMP_SLE:
    Os << " sle";
    break;
  case ICmpInst::ICMP_SGT:
    Os << " sgt";
    break;
  case ICmpInst::ICMP_SGE:
    Os << " sge";
    break;
  default:
    Os << " ?";
    break;
  }
  Os << "` against compile-time constant " << SmallK->getValue()
     << " which is within (0, W_s-1=" << (SourceWaveSize - 1) << "]. ";
  Os << "This is a lane-position-scoped predicate; under cross-widening ";
  Os << "modulo-replication, source wave 0's lane L and target replica-1's ";
  Os << "lane L+W_s would evaluate the predicate differently despite sharing ";
  Os << "the same source EXEC bit. Refusing per hotswap/docs/"
        "modrep-predicate-chain.md sec. 5 (narrow-O1 classifier).";
  return S;
}

bool modrepCanHaveActiveReplicaLane(unsigned SourceWaveSize,
                                    unsigned MaxFlatWorkgroupSize) {
  // Unknown metadata cannot prove the no-replica-lane case, so keep the
  // historical loud refusal. A known workgroup bound no larger than the
  // source wave means target lanes outside [0, W_s) are hardware-inactive
  // under MODREP and cannot observe a divergent `tid` predicate.
  return MaxFlatWorkgroupSize == 0 || MaxFlatWorkgroupSize > SourceWaveSize;
}

bool shouldRefuseC5(PredicateChainProjection Projection,
                    unsigned SourceWaveSize, unsigned TargetWaveSize,
                    unsigned MaxFlatWorkgroupSize, bool SuppressThreadLoopC5) {
  switch (Projection) {
  case PredicateChainProjection::ModuloReplication:
    return modrepCanHaveActiveReplicaLane(SourceWaveSize, MaxFlatWorkgroupSize);
  case PredicateChainProjection::WaveNative:
    return MaxFlatWorkgroupSize > 0 && MaxFlatWorkgroupSize < TargetWaveSize;
  case PredicateChainProjection::ThreadLoop:
    return !SuppressThreadLoopC5;
  }
  llvm_unreachable("unknown PredicateChainProjection");
}

bool isWaveNativePhantomRefusal(PredicateChainProjection Projection,
                                unsigned TargetWaveSize,
                                unsigned MaxFlatWorkgroupSize) {
  return Projection == PredicateChainProjection::WaveNative &&
         MaxFlatWorkgroupSize > 0 && MaxFlatWorkgroupSize < TargetWaveSize;
}

std::string formatSuppressionReason(PredicateChainProjection Projection,
                                    unsigned SourceWaveSize,
                                    unsigned TargetWaveSize,
                                    unsigned MaxFlatWorkgroupSize,
                                    bool SuppressThreadLoopC5) {
  switch (Projection) {
  case PredicateChainProjection::ModuloReplication:
    if (MaxFlatWorkgroupSize > 0 && MaxFlatWorkgroupSize <= SourceWaveSize)
      return formatv("selected ModuloReplicationProjection with "
                     "max_flat_workgroup_size={0} <= source wave size {1}; "
                     "the HSACO launch-bound metadata proves no active target "
                     "replica lanes exist, so upper target lanes remain "
                     "hardware-inactive and cannot observe the C5 predicate",
                     MaxFlatWorkgroupSize, SourceWaveSize)
          .str();
    break;
  case PredicateChainProjection::WaveNative:
    if (MaxFlatWorkgroupSize == 0)
      return "selected WaveNativeProjection with unknown "
             "max_flat_workgroup_size; no WaveNative phantom-lane evidence is "
             "available, so C5 is an attribution breadcrumb";
    if (MaxFlatWorkgroupSize >= TargetWaveSize)
      return formatv("selected WaveNativeProjection with "
                     "max_flat_workgroup_size={0} >= target wave size {1}; "
                     "every target lane has a source-lane mapping",
                     MaxFlatWorkgroupSize, TargetWaveSize)
          .str();
    break;
  case PredicateChainProjection::ThreadLoop:
    if (SuppressThreadLoopC5)
      return "selected ThreadLoopProjection via an analysis-triggered "
             "cross-widen retry; source-wave-scoped lane ops and banked "
             "predicate masks own the C5 boundary for this narrowed route";
    break;
  }
  return "";
}

} // namespace

PredicateChainClassifierReport classifyPredicateChain(
    Function &F, unsigned SourceWaveSize, unsigned TargetWaveSize,
    PredicateChainProjection Projection, unsigned MaxFlatWorkgroupSize,
    bool SuppressThreadLoopC5) {
  PredicateChainClassifierReport Report;

  // Direction gate: no predicate-chain risk at same-wave or narrowing,
  // and a zero/one-lane source wave is structurally incoherent.
  if (TargetWaveSize <= SourceWaveSize)
    return Report;
  if (SourceWaveSize < 2)
    return Report;

  const bool RefuseObservedC5 =
      shouldRefuseC5(Projection, SourceWaveSize, TargetWaveSize,
                     MaxFlatWorkgroupSize, SuppressThreadLoopC5);
  const bool WaveNativePhantomRefusal = isWaveNativePhantomRefusal(
      Projection, TargetWaveSize, MaxFlatWorkgroupSize);

  // ===== Pass 0: collect `@llvm.amdgcn.workitem.id.x()` call sites. =====
  SmallVector<CallInst *> Sites;
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    Function *Callee = CI->getCalledFunction();
    if (!Callee)
      continue;
    if (Callee->getIntrinsicID() == Intrinsic::amdgcn_workitem_id_x)
      Sites.push_back(CI);
  }
  Report.VisitedCalls = static_cast<unsigned>(Sites.size());

  if (Sites.empty())
    return Report;

  // ===== Pass 1: tag every tid-reachable value as unmasked vs masked. =====
  //
  // A value reaches `unmaskedVisited` iff there exists a walk from some
  // `workitem.id.x()` call to that value that did NOT pass through an
  // `and V, K<=W_s-1` mask. A value reaches `maskedVisited` iff there
  // exists a walk that DID pass through such a mask.
  //
  // Values that reach only `maskedVisited` evaluate identically across
  // MODREP replicas (the mask collapsed `tid` and `tid+W_s` onto the
  // same `[0, W_s)` residue). Values in `unmaskedVisited` retain
  // per-replica divergence and are the lane-position-sensitive subset.
  // A value can be in both sets (e.g. a phi that joins a masked arm and
  // an unmasked arm); Pass 2 treats it as unmasked for the conservative
  // refusal check.
  SmallPtrSet<Value *, 32> UnmaskedVisited;
  SmallPtrSet<Value *, 32> MaskedVisited;

  // Worklist entries: (Value, unmaskedPath). `unmaskedPath = true`
  // means this visit reached V without crossing a mask; `false` means
  // it crossed at least one mask.
  SmallVector<std::pair<Value *, bool>> Worklist;
  for (CallInst *Root : Sites)
    Worklist.push_back({Root, /*unmaskedPath=*/true});

  while (!Worklist.empty()) {
    auto [V, unmaskedPath] = Worklist.pop_back_val();
    auto &Set = unmaskedPath ? UnmaskedVisited : MaskedVisited;
    if (!Set.insert(V).second)
      continue;

    for (Use &U : V->uses()) {
      User *User = U.getUser();
      auto *I = dyn_cast<Instruction>(User);
      if (!I) {
        // Non-Instruction user on a tid-derived Value. ConstantExpr
        // cannot reference runtime Instructions (its operands must be
        // Constants), so this branch should be structurally unreachable.
        // If it ever does fire, that's an unexpected IR shape the
        // classifier can't reason about -- refuse (fail loud) per the
        // AGENTS.md "no silent fallback" discipline. This is deliberately
        // NOT projection-suppressible: the accepted single-source-wave
        // MODREP proof below applies to known C5 icmp sites, not unknown IR
        // user classes.
        Report.ObservedSites.push_back(
            "unexpected non-Instruction user of tid-derived Value; "
            "classifier cannot prove safety. "
            "See hotswap/docs/modrep-predicate-chain.md sec. 5.");
        Report.Refused = true;
        Report.WaveNativePhantomRefusal = false;
        if (Report.RefusalDetail.empty())
          Report.RefusalDetail = Report.ObservedSites.back();
        continue;
      }

      // Mask transition: if the `and` folds V onto `[0, W_s)`, the
      // AND result enters `maskedVisited`. Downstream uses of the
      // AND result cannot re-introduce per-replica divergence unless
      // they re-join an unmasked path (e.g. via a phi) -- which Pass 2
      // already handles by observing both sets.
      if (isSourceWaveMaskAnd(I, V, SourceWaveSize)) {
        Worklist.push_back({I, /*unmaskedPath=*/false});
        continue;
      }

      // `icmp` result is `i1` -- not a tid-derived value for this class.
      // Pass 2 audits icmps separately by scanning every icmp in F
      // and inspecting each operand against unmaskedVisited /
      // maskedVisited, so the walker doesn't need to propagate here.
      if (isa<ICmpInst>(I))
        continue;

      if (isPurePropagator(I)) {
        Worklist.push_back({I, unmaskedPath});
        continue;
      }

      // Unknown Instruction user (store, load, call, branch, return,
      // atomic, ...). These are sinks for the narrow-O1 class: the
      // value is consumed verbatim into the instruction, and the
      // instruction's result (if any) is not a lane-position-scoped
      // predicate value that the classifier should refuse on. If a
      // future iteration widens the class (e.g. to also refuse on
      // `store tid, ...` shapes), the new gate goes here. Stays
      // silent today to preserve the baseline-non-refusal contract
      // (`vecadd_f16` / `rope_fp32` store `tid`-derived values
      // verbatim into global memory -- that is not a C5 shape).
    }
  }

  // ===== Pass 2: scan every icmp for the C5 shape. =====
  //
  // The narrow-O1 shape: the icmp has at least one operand in
  // `unmaskedVisited` AND at least one operand that is a compile-time
  // constant K in `(0, W_s-1]`. `maskedVisited` is tracked in Pass 1
  // as a guard -- an operand that's ONLY in `maskedVisited` (not
  // `unmaskedVisited`) is already collapsed onto `[0, W_s)` and
  // cannot be the unmasked half of the refusal -- but it is NOT by
  // itself enough to refuse the icmp (see the file-header discussion
  // of why the cross-subtree `unmasked + masked` theory was
  // falsified).
  //
  // Kinds of icmps that are explicitly NOT refused by this rule:
  //   * `icmp tid, dynamic_kernarg` -- bounds-check shape (baseline
  //     `vecadd_f16` / `rope_fp32` / `corpus_add_fp32` pattern).
  //   * `icmp (tid&15), 15` -- masked operand + small-K. Masked side
  //     collapsed the divergence onto `[0, W_s)`.
  //   * `icmp tid, (tid&15)` -- unmasked + masked cross-subtree. In
  //     the general case (`ult` / `ugt` / `slt` / `sgt` / `ule` /
  //     etc.) the specific value ranges of `tid` and `tid&15` make
  //     the predicate evaluate identically across replicas. The
  //     `eq` / `ne` variants CAN diverge, but the classifier does
  //     not currently distinguish -- a future iteration can tighten
  //     per-predicate if a genuine corpus kernel surfaces the
  //     divergence.
  for (Instruction &I : instructions(F)) {
    auto *Cmp = dyn_cast<ICmpInst>(&I);
    if (!Cmp)
      continue;

    bool HasUnmaskedOp = false;
    const ConstantInt *SmallK = nullptr;

    for (Value *Op : Cmp->operands()) {
      if (UnmaskedVisited.count(Op))
        HasUnmaskedOp = true;
      if (!SmallK && isNonZeroConstantUleBound(Op, SourceWaveSize - 1))
        SmallK = cast<ConstantInt>(Op);
    }

    if (!HasUnmaskedOp || !SmallK)
      continue;

    std::string Detail = formatRefusalDetail(Cmp, SourceWaveSize, SmallK);
    Report.ObservedSites.push_back(Detail);

    // Refuse according to the selected projection's C5 obligation.
    //
    // WaveNative removes MODREP's replica-lane EXEC aliasing, but equality
    // predicates against lane-position constants are still load-bearing. The
    // GPT-OSS MXFP4 upcast kernel surfaced the concrete counterexample:
    // `icmp eq workitem.id.x-derived, 16` gates packed-sign selection and
    // sparsely flips signs under wave32->wave64 WaveNative. The file header
    // already identified eq/ne as the divergent predicate family; enforce that
    // distinction here instead of suppressing it as attribution-only.
    const bool WaveNativeEqualitySite =
        Projection == PredicateChainProjection::WaveNative && Cmp->isEquality();
    if (WaveNativeEqualitySite)
      Report.WaveNativeEqualityObserved = true;
    const bool SiteRefuses = RefuseObservedC5;

    // `!report.refused` avoids rewriting `refusalDetail` with later sites
    // so the diagnostic names the first failing icmp deterministically.
    if (SiteRefuses && !Report.Refused) {
      Report.Refused = true;
      Report.WaveNativePhantomRefusal = WaveNativePhantomRefusal;
      Report.WaveNativeEqualityRefusal = false;
      if (WaveNativePhantomRefusal) {
        // Prepend the phantom-lane explanation so the diagnostic
        // names the distinguishing evidence the operator needs:
        // "WaveNative would normally let this through; the
        // specific reason it refuses THIS kernel is the
        // sub-wave-width `max_flat_workgroup_size`".
        std::string Prefix;
        raw_string_ostream Os(Prefix);
        Os << "phantom-lane regime: HSACO "
              "`max_flat_workgroup_size`="
           << MaxFlatWorkgroupSize << " is below target wavefront width "
           << TargetWaveSize
           << ", so every launch activates lanes outside the "
              "source kernel's lane index space via "
              "`init_whole_wave`. Under WaveNativeProjection the "
              "MODREP-specific `replica-1 EXEC-share` trap the "
              "classifier normally catches does NOT apply, but "
              "phantom lanes are unmodelled by the "
              "per-source-lane EXEC model and the cross-widening "
              "safety argument collapses. ";
        Report.RefusalDetail = Prefix + Detail;
      } else {
        Report.RefusalDetail = std::move(Detail);
      }
    }
  }

  if (!Report.Refused && Report.WaveNativeEqualityObserved)
    Report.SuppressionReason =
        "selected WaveNativeProjection with target-width mask-shadow coverage "
        "for eq/ne C5 predicates; source-width SGPR stores are preserved while "
        "mask consumers use the shadow when proven valid";
  else if (!Report.Refused && !Report.ObservedSites.empty())
    Report.SuppressionReason =
        formatSuppressionReason(Projection, SourceWaveSize, TargetWaveSize,
                                MaxFlatWorkgroupSize, SuppressThreadLoopC5);

  return Report;
}

} // namespace COMGR::hotswap
