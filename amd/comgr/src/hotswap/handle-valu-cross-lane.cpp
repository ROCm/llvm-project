//===- handle-valu-cross-lane.cpp - Hotswap transpiler --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-internal.h"

#include "canonical-op.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::OpName
#include "SIDefines.h"                        // SISrcMods::OP_SEL_0
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR::hotswap {

// Cross-lane VALU primitives -- the subset of VALU opcodes whose result
// in lane L depends on values held by lane L' != L. Isolated from the
// rest of handleVALU because this is exactly the surface the cross-
// wave strategy (hotswap/docs/wave-size-translation.md §§5.3 and 7)
// keeps iterating on: every rewrite from the "wave-size-baked cross-
// lane" rewrite table lands in this file, not scattered through the
// VALU arithmetic sections.
//
// Each branch MUST use a genuine cross-lane LLVM intrinsic
// (`llvm.amdgcn.readlane`, `writelane`, `readfirstlane`, `mbcnt.{lo,
// hi}`, etc.). A "same-lane" stub that ignores the source-lane
// selector is a silent miscompile for any kernel that feeds divergent
// operands into the primitive. Several permlane variants here are
// known broken (see the pending-rewrite table in wave-size-
// translation.md §7); they stay same-lane for now but any new cross-
// lane CanonicalOp must be modelled correctly before landing.

// Shared `ds_bpermute`-based emulation of the `VOP_PERMLANE_SWAP`
// profile (see `llvm/lib/Target/AMDGPU/VOP1Instructions.td`) -- two
// tied VGPR pairs (vdst<->vdst_in, src0<->src0_out) whose values are
// swapped across the lane partner `L XOR partnerXorMask`.  Called
// from both the XOR-16 and XOR-32 arms below; the only things that
// differ between the two CanonicalOp cases are:
//
//   * `partnerXorMask` -- 16 for `v_permlane16_swap_b32` (XOR-16 pair
//     stays within each 32-lane half), 32 for
//     `v_permlane32_swap_b32` (XOR-32 pair spans both halves of a
//     wave64).
//   * `ssaPrefix`       -- `"pls16"` or `"pls32"`, stamped onto every
//     emitted SSA value's twine.  Lit fixtures pin on this prefix
//     (`lit_tests/{c2_permlane_swap,v_permlane32_swap_b32}/`), so it
//     IS a load-bearing contract; any future widener (XOR-64 on some
//     future ISA, say) would pick its own prefix.
//
// EXEC / fi / bc handling: the MCInst surfaces `fi` and `bound_ctrl`
// as named immediate operands ONLY in the e64 form (VOP3OpSel
// encoding); the e32 form (VOP1 encoding, which the GPT-OSS /
// AITER corpus exclusively emits) has no fi/bc operands and they
// default to 0.  The `ds_bpermute` emulation is observationally
// equivalent to the source's `fi=0, bc=0` semantics WHEN
// EXEC=all-active at the swap site -- which is the invariant
// butterfly-reduction kernels (Triton / AITER fmha reduction cores)
// maintain (the kernel emits divergent EXEC writes only at
// iteration boundaries, not inside the reduction core).  For
// partial-EXEC sites:
//
//   * fi=0 source semantics: inactive lanes' contribution is 0;
//     ds_bpermute returns the stale VGPR value instead.  Divergence
//     only on inactive lanes.
//   * bc=0 source semantics: out-of-range source lane -> return
//     %old.  For the XOR-16 swap every partner is in-range (XOR 16
//     stays within each 32-lane half); for the XOR-32 swap every
//     partner is in-range on wave64 (XOR 32 maps 0..31 <-> 32..63
//     within the single wave).  `bc` is irrelevant for both masks.
//
// The corpus patterns (e32 form, EXEC=full at the swap site) are
// bit-exact correct under this emulation; fi/bc are accepted
// without inspection and the EXEC=full assumption is documented
// here.  P4.b future-hardening (wave-size-translation.md §10): a
// "true fi=0 emulation" would zero inactive lanes' VGPR
// contribution before the bpermute via `select EXEC[L], src, 0`,
// at the cost of two extra selects per swap; a static alternative
// is a classifier check that proves EXEC=full at the swap site
// and refuses otherwise.  Today's corpus invariant makes both
// deferrable.
static HandlerResult
emitPermLaneSwapEmulation(RaiseContext &Ctx, const DecodedInst &Di,
                           OpResolver &Op, uint32_t PartnerXorMask,
                           const char *SsaPrefix) {
  HandlerResult Hr;

  // Operand-table contract (same for every `VOP_PERMLANE_SWAP`
  // profile instance): MCInst operand 0 is `vdst` (output, tied to
  // `vdst_in` input -- the disassembler elides the tied input), and
  // the `OpName::src0_out` named operand is the second output (tied
  // to `src0` input, same elision).  Logical inputs are therefore
  // `readReg32(vdst)` (the pre-instruction value of the tied vdst
  // slot) and `op.src(0)` (which `buildSrcMap` keeps as src0 after
  // the `vdst_in` elision).  Two outputs, two tied inputs, both
  // carried on VGPRs.
  int Src0OutIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                               AMDGPU::OpName::src0_out);
  if (Src0OutIdx < 0 ||
      static_cast<unsigned>(Src0OutIdx) >= Di.Inst.getNumOperands() ||
      !Di.Inst.getOperand(static_cast<unsigned>(Src0OutIdx)).isReg()) {
    std::string Msg = std::string(Di.Mnemonic) +
                      " missing OpName::src0_out register operand -- "
                      "operand-table mismatch (expected the "
                      "VOP_PERMLANE_SWAP profile's second-output "
                      "operand-table slot to be a register)";
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(Di, "VALU", Msg);
    return Hr;
  }
  ParsedReg VdstReg = Op.dst();
  // src0_out is the second output but is tied to src0: it must land in src0's
  // own VGPR and VGPR_MSB bank.  Parsing it via its own MCInst operand index
  // applies `CurrentVgprAdjust[Src0OutIdx]`, which `computeVGPRAdjust` sets to
  // the destination MSB for every def slot.  When `vdst` and `src0` sit in
  // different banks that misroutes the swapped result to the destination-bank
  // alias of src0, leaving src0's real bank stale.  Reuse src0's own
  // (correctly bank-adjusted) register for the second output instead.
  ParsedReg Src0OutReg = Op.srcReg(0);

  // Snapshot BOTH inputs up-front (read-before-write: `vdst_in`
  // aliases `vdst`, so writing vdstReg first would shadow this
  // read; mirroring the snapshot for `src0_in` keeps the pair
  // structural rather than relying on the rest of the handler's
  // implicit ordering).
  Value *VdstIn = Ctx.Regs.readReg32(Ctx.B, VdstReg);
  Value *Src0In = Op.src(0);

  // Partner lane: `L XOR partnerXorMask`.  Computed on the target
  // hardware lane id (mbcnt-derived), with per-BB memoisation via
  // `emitLaneIdx`.  Byte-address shift by 2 is the ds_bpermute
  // convention -- each lane's selector is the byte offset of the
  // source lane's LDS slot (LDS slot size = 4 bytes for a 32-bit
  // dword).
  Value *LaneId = Ctx.emitLaneIdx();
  Value *Partner = Ctx.B.CreateXor(
      LaneId, Ctx.B.getInt32(PartnerXorMask),
      Twine(SsaPrefix) + "_partner");
  Value *BpermIdx = Ctx.B.CreateShl(Partner, Ctx.B.getInt32(2),
                                     Twine(SsaPrefix) + "_addr");

  // Two emission shapes gated by source wave size, both emitting
  // TWO `ds_bpermute` calls sharing `bpermIdx`:
  //
  //   * WAVE32 source -- ASYMMETRIC per-lane select matching the
  //     MI400 Shader Programming Guide § V_PERMLANE16_SWAP_B32
  //     pragma (only two of the four 16-lane rows move).
  //
  //   * WAVE64 source -- SYMMETRIC cross-wired bpermute pair
  //     (every lane swaps with its `L XOR mask` partner), which
  //     is the lift shape that lived here before 2026-04-23.
  //     Retained conservatively: the asymmetric pragma above is
  //     gfx1250-specific (the only wave32 ISA that exposes
  //     `v_permlane16_swap_b32` + `v_permlane32_swap_b32`), and
  //     we don't currently have the gfx950 pragma in hand to
  //     confirm whether the wave64 flavour mirrors it or is
  //     genuinely symmetric.  The existing `c2_permlane_swap`
  //     and `v_permlane32_swap_b32` lit fixtures exercise this
  //     arm on gfx950 -> gfx942 and pin the symmetric shape; the
  //     graduated corpus did not regress under it before Session
  //     8.  If a future gfx950-source regression surfaces that
  //     points at this arm, confirm via `docs/manuals/` and
  //     switch the branch to the asymmetric emission.
  //
  // Emission outside `emitUnderExec`: all hardware lanes must
  // participate in the bpermute's LDS round-trip (fetch-invalid /
  // `OPF_EXEC_FI` per the MI400 V_PERMLANE16_SWAP_B32 op entry),
  // so we don't gate the compute under the source-active EXEC
  // here.  `writeReg32` below still wraps the final stores in
  // the target-side `emitUnderExec` so EXEC-inactive target lanes
  // keep their prior VGPR values, matching the
  // "OPF_WRMASK_NOT_EXEC" + asymmetric "if EXEC[lane]" write-side
  // flags in the op's pragma.  This is the same invariant that
  // held pre-Session-8 for the fi=0 / bc=0 assumption documented
  // in the top-of-function block comment; the asymmetric arm
  // does not widen the divergent-EXEC contract.
  Function *Bperm = Intrinsic::getOrInsertDeclaration(
      &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
  Value *NewVdst = nullptr;
  Value *NewSrc0Out = nullptr;
  if (Ctx.Isa.isWave32()) {
    // Asymmetric gfx1250 semantic of `v_permlane16_swap_b32`
    // (MI400 Shader Programming Guide § V_PERMLANE16_SWAP_B32
    // pragma, verbatim):
    //
    //   // Lanes 0:15 of src0 and lanes 16:31 of vdst swapped.
    //   // Lanes 16:31 of src0 and lanes 0:15 of vdst are unchanged.
    //   for lane in 0:15 do tmp[lane] = VGPR[lane][SRC0] endfor;
    //   for lane in 0:15 do
    //     if EXEC[lane]:    VGPR[lane][SRC0]  = VGPR[lane+16][VDST]
    //     if EXEC[lane+16]: VGPR[lane+16][VDST] = tmp[lane]
    //   endfor
    //
    // Only TWO of the four 16-lane "rows" move; the other two
    // retain their old values.  Per-lane:
    //
    //   new_vdst[L]     = (L_low) ?  vdst_in[L]             // UNCHANGED
    //                             :  old src0[L XOR mask]   // cross-wired
    //   new_src0_out[L] = (L_low) ?  old vdst[L XOR mask]   // cross-wired
    //                             :  src0_in[L]             // UNCHANGED
    //
    // where `L_low` is the low row of each partnered pair.  For
    // the XOR-16 variant (partnerXorMask=16) that's `L ∈ [0,15]
    // ∪ [32,47]` (i.e. `(L & 16) == 0`), which generalises
    // correctly to both MODREP wave32 replicas on the wave64
    // target (lanes 0..31 and 32..63).  The XOR-32 variant
    // cannot reach here (no wave32 ISA exposes
    // `v_permlane32_swap_b32` today); if one is added in the
    // future, the SAME pattern applies with `(L & 32) == 0`.
    //
    // Pre-Session-8 this arm emitted the symmetric cross-wire
    // (below) unconditionally -- over-swapping the "unchanged"
    // halves corrupted every `matmul_fp16` A-operand position
    // because vdst_in and src0_in carry distinct data at the
    // swap site (see § 12.4.7 of hotswap/docs/matrix-
    // translation.md for the Session-8 root-cause pin).  The
    // self-preserve idiom (`vdst_in == src0_in == seed`, Triton
    // `tl.sort` / `tl.topk`) masqueraded as working because the
    // per-lane select collapses to `seed` for the preserved
    // half anyway; the transitional `rewrite_permlane16_{xor3_
    // partner,swap_selfpreserve}` passes that papered over that
    // aliasing are deleted along with the symmetric emission.
    //
    // `isLaneLow` is computed via `lane AND partnerXorMask == 0`
    // rather than `lane < partnerXorMask` so the backend can
    // fold the AND into the subsequent select without a 32-bit
    // compare (and it extends trivially to the two MODREP
    // replicas above).
    Value *BpermSrc0 = Ctx.B.CreateCall(
        Bperm, {BpermIdx, Src0In},
        Twine(SsaPrefix) + "_bperm_src0"); // = old src0[L XOR mask]
    Value *BpermVdst = Ctx.B.CreateCall(
        Bperm, {BpermIdx, VdstIn},
        Twine(SsaPrefix) + "_bperm_vdst"); // = old vdst[L XOR mask]
    Value *HalfBit = Ctx.B.CreateAnd(
        LaneId, Ctx.B.getInt32(PartnerXorMask),
        Twine(SsaPrefix) + "_half_bit");
    Value *IsLaneLow = Ctx.B.CreateICmpEQ(
        HalfBit, Ctx.B.getInt32(0),
        Twine(SsaPrefix) + "_is_lane_low");
    NewVdst = Ctx.B.CreateSelect(
        IsLaneLow, VdstIn, BpermSrc0,
        Twine(SsaPrefix) + "_new_vdst");
    NewSrc0Out = Ctx.B.CreateSelect(
        IsLaneLow, BpermVdst, Src0In,
        Twine(SsaPrefix) + "_new_src0_out");
  } else {
    // Wave64 source (gfx950): pre-Session-8 symmetric lift.
    // Kept verbatim -- see the function-top branch comment above
    // for the gfx950-ISA-unconfirmed caveat and the two lit
    // fixtures that pin this shape.  Every lane's two output
    // VGPRs take its partner's tied-input value directly:
    //
    //   new_vdst[L]     = bperm(addr, src0_in)  = old src0[L XOR mask]
    //   new_src0_out[L] = bperm(addr, vdst_in)  = old vdst[L XOR mask]
    //
    // Unlike the wave32 arm we emit the bpermute results
    // straight into `writeReg32`; no per-lane select.
    NewVdst = Ctx.B.CreateCall(Bperm, {BpermIdx, Src0In},
                               Twine(SsaPrefix) + "_new_vdst");
    NewSrc0Out = Ctx.B.CreateCall(Bperm, {BpermIdx, VdstIn},
                                  Twine(SsaPrefix) + "_new_src0_out");
  }
  Ctx.writeReg32(VdstReg, NewVdst);
  Ctx.writeReg32(Src0OutReg, NewSrc0Out);
  Hr.Handled = true;
  return Hr;
}
HandlerResult handleValuCrossLane(RaiseContext &Ctx, const DecodedInst &Di,
                                    OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  switch (Sop) {

  // ---- v_permlane16_b32 / v_permlanex16_b32 ----
  // P2 lowering -- see the permlane16 / permlanex16 row of hotswap/
  // docs/wave-size-translation.md §5.3. Target constraint: `v_permlane16`
  // and `v_permlanex16` are RDNA/gfx10+ instructions and DO NOT exist
  // on CDNA (gfx9/gfx94x). Emitting `llvm.amdgcn.permlane16` or
  // `permlanex16` directly fails isel on gfx942 with "Cannot select:
  // intrinsic %llvm.amdgcn.permlanex16". We therefore emulate both
  // via `ds_bpermute_b32`, which IS available on every AMDGPU
  // generation with LDS (gfx8+), so this lowering is target-
  // independent -- it works for gfx1250 -> gfx942, gfx1250 -> gfx1250,
  // and any future target with ds_bpermute.
  //
  // MCInst operand layout (from VOP3_PERMLANE_Profile's InsVOP3OpSel):
  //
  //   [0] vdst (output)             [5] src2_modifiers (always 0)
  //   [1] src0_modifiers  <-- fi    [6] src2 (SSrc_b32)  = selector_2
  //   [2] src0 (VRegSrc_32) = val   [7] vdst_in (VGPR, tied) = %old
  //   [3] src1_modifiers  <-- bc    [8] op_sel (VOP3OpSel imm, unused
  //   [4] src1 (SSrc_b32)  = sel_1                                here)
  //
  // Selector encoding: src1 and src2 are each 32-bit scalar values
  // containing 8 × 4-bit per-lane selectors. src1 covers within-
  // group lanes 0..7, src2 covers within-group lanes 8..15. Each
  // 4-bit nibble selects a source lane within the 16-lane group.
  //
  // Per-lane emulation, L = `mbcnt`-derived absolute lane id (0..W_t):
  //
  //   group_base  = L & ~0xF           // 16, 32, 48 boundaries
  //   within      = L & 0xF            // 0..15
  //   within_lo   = within & 7         // 0..7 (nibble index)
  //   sel_word    = within < 8 ? src1 : src2
  //   nibble      = (sel_word >> (within_lo * 4)) & 0xF
  //
  //   permlane16  : src_group = group_base
  //   permlanex16 : src_group = group_base ^ 0x10  (swap adjacent groups)
  //
  //   src_lane_abs = src_group | nibble
  //   byte_addr    = src_lane_abs << 2
  //   result       = ds_bpermute(byte_addr, src0)
  //
  // Wave-width correctness under modulo-replication (hotswap/docs/
  // wave-size-translation.md §6's wave-size-obliviousness theorem):
  // the source gfx1250 kernel is wave32 so its selector values
  // encode a shuffle pattern over 2 × 16-lane groups. On wave64
  // target each modrep replica occupies 2 × 16-lane groups (R=2),
  // and the `group ^ 0x10` swap stays within a replica (0<->1 within
  // replica 0, 2<->3 within replica 1), so the modrep invariant is
  // preserved for permlanex16. permlane16 keeps every lane within
  // its own group, trivially within-replica.
  //
  // Handling of `fi` (fetch-invalid) and `bc` (bound_ctrl) -- the two
  // i1 immediates encoded via `opsel_i1timm` in PermlanePat
  // (`SISrcMods::OP_SEL_0` bit of src0_modifiers / src1_modifiers):
  //
  //   - `fi=1`: on an EXEC-inactive source lane, the kernel still
  //     fetches that lane's VGPR value (possibly stale). This is
  //     exactly how `llvm.amdgcn.ds.bpermute` behaves naturally
  //     (the LDS-backed path reads the VGPR alloca regardless of
  //     EXEC), so `fi=1` is supported directly.
  //   - `bc=0`: on an "out-of-range" source lane, the target lane
  //     retains %old. For permlane16 the 4-bit selector nibble is
  //     always in [0, 16) so the source lane is always in-group;
  //     `bc=0` is the only case the emulation needs to support.
  //     Under SPE, `writeReg32`'s `emitUnderExec` already retains
  //     prior VDST values on EXEC-masked target lanes, covering the
  //     "target lane inactive" direction of `bc=0`.
  //   - `fi=0` and `bc=1` diverge from the above in ways the
  //     emulation does not model. Every GPT-OSS / softmax /
  //     bitmatrix disassembly we have examined uses `op_sel:[1, 0]`
  //     (fi=1, bc=0); refusing the other combinations keeps the
  //     classifier-gate's "no silent miscompile" invariant intact
  //     rather than emitting ds_bpermute with fi=0 semantics it
  //     does not provide.
  //
  // Future optimisation: on targets that DO support native
  // permlane16 (gfx10+), emit the intrinsic directly for lower
  // latency. Left as a profitability refinement -- correctness-first
  // lands the ds_bpermute emulation.
  case CanonicalOp::V_PERMLANE16_B32:
  case CanonicalOp::V_PERMLANEX16_B32: {
    const bool IsPermlaneX16 = (Sop == CanonicalOp::V_PERMLANEX16_B32);
    const bool Fi = (Op.srcMod(0) & SISrcMods::OP_SEL_0) != 0;
    const bool Bc = (Op.srcMod(1) & SISrcMods::OP_SEL_0) != 0;
    if (!Fi || Bc) {
      // Empirically the GPT-OSS / softmax / bitmatrix corpora emit
      // `op_sel:[1, 0]` exclusively (fi=1, bc=0). Refuse any other
      // encoding loudly so a future corpus kernel's extended
      // fi/bc use surfaces during classifier verification rather
      // than producing an approximation silently. Re-narrowing this
      // gate is the right place to extend the emulation.
      std::string Detail;
      raw_string_ostream Os(Detail);
      Os << "permlane16 / permlanex16 emulation supports only "
            "op_sel:[1,0] (fi=1, bc=0); saw fi="
         << (Fi ? 1 : 0) << ", bc=" << (Bc ? 1 : 0);
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(Di, "VALU", Detail);
      return Hr;
    }
    Value *Src0 = Op.src(0);
    Value *Sel1 = Op.src(1);
    Value *Sel2 = Op.src(2);

    // Target-hardware lane id, wave-width-aware via emitLaneIdx, with
    // per-BB memoisation. Multiple permlane16 sites in the same BB
    // (e.g. butterfly reductions) reuse the single cached i32 instead
    // of re-emitting the mbcnt_lo / mbcnt_hi chain at each site --
    // LLVM's CSE would converge to the same end state, but the
    // pre-mem2reg IR stays smaller and lit-test-friendlier.
    Value *LaneId = Ctx.emitLaneIdx();

    // Group base (lane & ~0xF) and within-group index (lane & 0xF).
    Value *GroupBase = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(~0xF), "pl_group");
    Value *Within = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(0xF), "pl_within");

    // Pick the right 32-bit selector word based on within's high bit.
    Value *IsHiHalf = Ctx.B.CreateICmpUGE(Within, Ctx.B.getInt32(8), "pl_hi");
    Value *SelWord = Ctx.B.CreateSelect(IsHiHalf, Sel2, Sel1, "pl_sel");

    // Extract the 4-bit nibble at position (within & 7) * 4.
    Value *WithinLo = Ctx.B.CreateAnd(Within, Ctx.B.getInt32(7), "pl_lo");
    Value *ShiftAmt = Ctx.B.CreateShl(WithinLo, Ctx.B.getInt32(2), "pl_shift");
    Value *Shifted = Ctx.B.CreateLShr(SelWord, ShiftAmt, "pl_shifted");
    Value *Nibble = Ctx.B.CreateAnd(Shifted, Ctx.B.getInt32(0xF), "pl_nibble");

    // For permlanex16, XOR the group base by 0x10 to swap adjacent groups.
    Value *SrcGroup = IsPermlaneX16
        ? Ctx.B.CreateXor(GroupBase, Ctx.B.getInt32(0x10), "plx_group")
        : GroupBase;
    Value *SrcLaneAbs = Ctx.B.CreateOr(SrcGroup, Nibble, "pl_src_lane");
    Value *ByteAddr = Ctx.B.CreateShl(SrcLaneAbs, Ctx.B.getInt32(2), "pl_addr");

    // Convergent: emit the bpermute outside any emitUnderExec diamond
    // so all hardware lanes participate. writeReg32 below wraps the
    // store for EXEC masking.
    Function *Bperm = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
    Value *Result = Ctx.B.CreateCall(
        Bperm, {ByteAddr, Src0},
        IsPermlaneX16 ? "permlanex16_emu" : "permlane16_emu");
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_permlane64_b32 ----
  // KNOWN LIMITATION -- see the v_permlane64_b32 row in the
  // unrewritable table of hotswap/docs/wave-size-translation.md §7:
  // no wave32 analogue, so
  // the Phase 1.4.5 classifier refuses this op in any cross-wave
  // lift (it is taxonomised as FullWaveRotate / unrewritable). The
  // same-lane fallback here only runs in same-wave (wave64 -> wave64)
  // translation, where a gfx1250 binary would not contain the op
  // anyway (gfx942 and earlier do not emit it). Keeping the stub
  // prevents a silent raise failure on the theoretical case.
  case CanonicalOp::V_PERMLANE64_B32: {
    if (Di.NumDefs >= 1 && Di.NumSrcs >= 1)
      Ctx.writeReg32(Op.dst(), Op.src(0));
    Hr.Handled = true;
    return Hr;
  }

  // ---- gfx950 lane-swap: v_permlane16_swap_b32 ----
  //
  // P4 lowering -- see the permlane16_swap row of hotswap/docs/wave-
  // size-translation.md §5.3. Exchanges two VGPRs across
  // lanes 0..15 <-> 16..31 within each 32-lane group. Two defs
  // (vdst, src0_out) and two tied uses (vdst_in tied to vdst,
  // src0 tied to src0_out). Effect:
  //
  //   new_vdst[L]      = src0_in[L XOR 16]
  //   new_src0_out[L]  = vdst_in[L XOR 16]
  //
  // Target constraint mirrors P2: `v_permlane16_swap_b32` exists
  // natively only on gfx950 (CDNA4) and gfx12+ (HasPermlane16Swap
  // subtarget feature). gfx942 (CDNA3) and earlier wave64 targets
  // lack native isel for `llvm.amdgcn.permlane16.swap` -- upstream
  // LLVM's `test/CodeGen/AMDGPU/llvm.amdgcn.permlane16.swap.ll`
  // explicitly asserts the gfx942 isel failure ("LLVM ERROR: Cannot
  // select: intrinsic %llvm.amdgcn.permlane16.swap"). Emulate via
  // `ds_bpermute_b32`, available on every AMDGPU generation with
  // LDS (gfx8+), so this lowering is target-independent.
  //
  // Per-lane emulation, L = `mbcnt`-derived absolute lane id:
  //
  //   partner   = L XOR 16
  //   bperm_idx = partner << 2          // ds_bpermute byte address
  //   new_vdst       = ds_bpermute(bperm_idx, src0_in)
  //   new_src0_out   = ds_bpermute(bperm_idx, vdst_in)
  //
  // Wave-width correctness under modulo-replication: lane L XOR 16
  // is naturally per-32-lane-half-independent on wave64 (lane 0 <->
  // lane 16 in lower half, lane 32 <-> lane 48 in upper half -- both
  // halves swap internally). The wave32 source kernel's two-VGPR
  // exchange therefore lifts to a wave64 instruction that performs
  // the same exchange independently in each half, the textbook
  // modulo-replication match.
  //
  // Historical regression gate: a wave32 source kernel using
  // `v_permlane16_swap_b32_e32` via inline asm) and runs it on
  // gfx942 wave64 hardware, verifying per-lane outputs match the
  // expected XOR-16 partner pattern across all 64 lanes:
  //
  //   For every L in [0, 64): new_vdst[L]      == 1000 + (L XOR 16)
  //                           new_src0_out[L]  == (L XOR 16)
  //
  // (vdst_in seeded with L, src0_in with 1000+L.) The test verifies
  // BOTH the per-lane swap pattern AND per-32-lane-half
  // independence (lanes 32..63 produce the lower-half result + 32
  // for both VGPRs). A future change that breaks the XOR-16
  // partner, the byte-address shift, the convergence semantics, or
  // the per-half independence would fail this test.
  //
  // Native gfx942 isel for `llvm.amdgcn.permlane16.swap` would fail
  // (per upstream LLVM's permlane16.swap.ll ERR-SDAG assertion), so
  // we cannot directly compare emulation-vs-native on this target;
  // probing other targets (e.g. gfx950 which has the native
  // instruction) is left for hardware-availability work and is the
  // P4.b sub-item recorded in hotswap/docs/wave-size-translation.md
  // §10 (known gaps). The emulation
  // independently maps onto the published .td swap semantics
  // (VOP_PERMLANE_SWAP profile), so per-target hardware-vs-
  // emulation parity follows from emulation-correctness +
  // .td-semantics.
  //
  // Emission details (operand-table lookup, snapshot ordering,
  // convergence, EXEC/fi/bc handling, P4.b future-hardening) are
  // documented on the shared `emitPermLaneSwapEmulation` helper at
  // the top of this file.  Both arms below call the helper with
  // their mask-specific arguments; the only per-arm code here is
  // the XOR-32 precondition check on wave size.
  //
  // V_PERMLANE32_SWAP_B32 (the wider variant) is the wave64-native
  // XOR-32 sibling.  Source kernels that emit it come from the
  // gfx950-and-later wave64 ISAs where `FeaturePermlane32Swap`
  // is enabled (the GFX950Insts feature block in
  // llvm/lib/Target/AMDGPU/AMDGPU.td).  Same-wave lifts for those
  // sources (gfx950 -> gfx942) need an emulation because the gfx942
  // target does NOT enable `FeaturePermlane32Swap` -- gfx940 base
  // features do not include it, so the instruction is unavailable
  // natively on the compilation target.  The helper's
  // `ds_bpermute(lane_id XOR 32, src)` emulation covers this on
  // wave64 -> wave64; refusal is preserved for the narrowing
  // direction (wave64 source -> wave32 target, which has no 64-lane
  // neighbourhood to XOR against) and for the impossible wave32-
  // source case (no wave32 ISA enables the feature, so seeing it
  // in wave32 source bytes indicates either a corrupted disassembly
  // or a wave64 source mis-classified as wave32).
  //
  // Historical regression gates included XOR-16 emulation on gfx942
  // hardware with per-lane outputs checked against the expected
  // XOR-16 partner
  //     pattern across all 64 lanes.  A regression in the helper
  //     (wrong XOR mask, missing byte-address shift, cross-wiring
  //     error) fails this test before reaching a user.
  // Historical corpus coverage included AITER `fmha_v3_fwd/fwd_hd128_bf16*`
  // kernels whose reduction cores depend on `v_permlane32_swap_b32`; a
  // regression on the helper surfaces there as a lift failure. IR-shape
  // coverage lives in `lit_tests/c2_permlane_swap.s` (XOR-16) and
  // `lit_tests/v_permlane32_swap_b32.s` (XOR-32).
  case CanonicalOp::V_PERMLANE16_SWAP_B32:
    return emitPermLaneSwapEmulation(Ctx, Di, Op, /*partnerXorMask=*/16,
                                      /*ssaPrefix=*/"pls16");
  case CanonicalOp::V_PERMLANE32_SWAP_B32: {
    // Two precondition checks, both specific to the wider XOR-32
    // variant:
    //
    //   1. target wave32 -> `laneId ^ 32` wraps past the target
    //      wave; `ds_bpermute` cannot deliver a lane index >=
    //      target wave size.
    //   2. source wave32 -> no wave32 ISA enables
    //      `FeaturePermlane32Swap` (see the GFX950Insts block in
    //      `llvm/lib/Target/AMDGPU/AMDGPU.td`), so seeing the
    //      instruction in wave32 source bytes means corrupted
    //      disassembly or upstream wave-size mis-classification.
    //
    // Refusal in both cases preserves the "refuse when uncertain"
    // contract documented on the P4 pending row of
    // hotswap/docs/wave-size-translation.md §5.3; the wave64 ->
    // wave64 path below is the positive case this handler now
    // lifts.  The XOR-16 sibling has no equivalent precondition
    // because its partner stays within each 32-lane half
    // regardless of wave size.
    if (Ctx.TargetIsa.isWave32()) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VALU",
          "v_permlane32_swap_b32 lift refused: target is wave32, "
          "but the instruction's XOR-32 partner has no wave32 "
          "analogue (the partner index wraps past the target "
          "wave).  See the P4 permlane32_swap entry in the "
          "pending-rewrite table of "
          "hotswap/docs/wave-size-translation.md \u00a75.3.");
      return Hr;
    }
    if (Ctx.Isa.isWave32()) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VALU",
          "v_permlane32_swap_b32 in a wave32 source kernel -- no "
          "wave32 ISA enables FeaturePermlane32Swap (see "
          "llvm/lib/Target/AMDGPU/AMDGPU.td), so this is either a "
          "corrupted disassembly or a wave64 source mis-classified "
          "as wave32 upstream.  Refusing rather than silently "
          "emitting an XOR-32 partner that cannot exist in the "
          "source's wave topology.");
      return Hr;
    }
    return emitPermLaneSwapEmulation(Ctx, Di, Op, /*partnerXorMask=*/32,
                                      /*ssaPrefix=*/"pls32");
  }

  // ---- v_readfirstlane_b32 sDST, vSRC ----
  // Broadcast the value of vSRC from the lowest-numbered active source lane
  // (or lane 0 if EXEC==0) to sDST.  Same-wave lowering can use the native
  // intrinsic directly.  Under cross-widening, however, native readfirstlane
  // would pick one lane for the entire target wave64 and collapse the two
  // source-wave halves together.  Emulate the source operation with
  // `ds_bpermute`: select the source-width slice of the modeled EXEC mask for
  // the current target lane's source-wave half, find that slice's first set
  // bit, and fetch the corresponding lane's VGPR.
  //
  // This is an explicit semantic translation, not a readfirstlane allow-list:
  // downstream scalar-looking uses now consume an already-broadcast
  // source-wave value that may differ between the two packed source waves.
  case CanonicalOp::V_READFIRSTLANE_B32: {
    Value *Src = Ctx.B.CreateZExtOrTrunc(Op.src(0), Ctx.I32Ty, "rfl_src");
    Value *Val = nullptr;
    if (Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
      Value *LaneId = Ctx.emitLaneIdx();
      uint32_t SourceMask = Ctx.Isa.WaveSize - 1;
      Value *GroupBase = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(~SourceMask),
                                         "rfl_source_wave_base");

      Value *Exec = Ctx.Regs.loadExec(Ctx.B);
      Value *ShiftAmt = Ctx.B.CreateZExtOrTrunc(GroupBase, Exec->getType(),
                                                "rfl_exec_shift");
      Value *SourceExecWide = Ctx.B.CreateLShr(Exec, ShiftAmt,
                                               "rfl_exec_at_srcwave");
      Value *SourceExec = Ctx.B.CreateTrunc(SourceExecWide, Ctx.I32Ty,
                                            "rfl_exec");
      Function *Cttz = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::cttz, {Ctx.I32Ty});
      Value *FirstSet = Ctx.B.CreateCall(
          Cttz, {SourceExec, ConstantInt::getFalse(Ctx.I1Ty)}, "rfl_first_set");
      Value *ExecIsZero = Ctx.B.CreateICmpEQ(SourceExec, Ctx.B.getInt32(0),
                                             "rfl_exec_is_zero");
      Value *SourceLane = Ctx.B.CreateSelect(ExecIsZero, Ctx.B.getInt32(0),
                                             FirstSet, "rfl_source_lane");
      Value *TargetLane = Ctx.B.CreateOr(GroupBase, SourceLane,
                                         "rfl_target_lane");
      Value *Addr = Ctx.B.CreateShl(TargetLane, Ctx.B.getInt32(2),
                                    "rfl_bperm_addr");
      Function *Bperm = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
      Val = Ctx.B.CreateCall(Bperm, {Addr, Src}, "readfirstlane_srcwave");
    } else {
      Function *Rfl = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_readfirstlane, {Ctx.I32Ty});
      Val = Ctx.B.CreateCall(Rfl, {Src}, "readfirstlane");
    }
    Ctx.writeReg32(Op.dst(), Val);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_writelane_b32 ----
  // Write `val` into lane `lane` of vDst. Cross-lane: cannot be
  // emulated via per-thread private scratch nor via a single scalar
  // SSA value. `llvm.amdgcn.writelane(val, lane, old)` lowers to the
  // hardware primitive; the intrinsic returns the new per-lane scalar
  // (either `val` when lane_id==lane, else `old`), so the VGPR's
  // SSA slot carries the correct value for whichever lane we are.
  //
  // First-write pattern: if writelane is the first assignment to
  // vDst, non-selected lanes legitimately hold whatever vDst
  // contained before (hardware semantics). `readReg32` on the
  // never-stored alloca returns LLVM `undef`, which is the right
  // "unobservable" encoding -- any downstream use of those lanes
  // before they are written is itself undefined on hardware.
  case CanonicalOp::V_WRITELANE_B32: {
    ParsedReg Dst = Op.dst();
    Value *Val = Op.src(0);
    Value *Lane = Op.src(1);
    Lane = Ctx.B.CreateZExtOrTrunc(Lane, Ctx.I32Ty, "wrlane_idx");
    Value *OldVal = Ctx.Regs.readReg32(Ctx.B, Dst);
    Value *NewVal = nullptr;
    if (Ctx.Projection.sourceWaveScopedLaneOps()) {
      Value *LaneId = Ctx.emitLaneIdx();
      Value *SourceLane = Ctx.B.CreateAnd(
          LaneId, Ctx.B.getInt32(Ctx.Isa.WaveSize - 1), "wrlane_source_lane");
      Value *WantedLane = Ctx.B.CreateAnd(
          Lane, Ctx.B.getInt32(Ctx.Isa.WaveSize - 1), "wrlane_wanted_lane");
      Value *IsTargetLane = Ctx.B.CreateICmpEQ(SourceLane, WantedLane,
                                               "wrlane_is_target_lane");
      NewVal = Ctx.B.CreateSelect(IsTargetLane, Val, OldVal,
                                  "writelane_srcwave");
    } else {
      Function *Wl = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_writelane, {Ctx.I32Ty});
      NewVal = Ctx.B.CreateCall(Wl, {Val, Lane, OldVal}, "writelane");
    }
    Ctx.writeReg32(Dst, NewVal);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_readlane_b32 sDST, vSRC, lane ----
  // Read a specific lane of vSRC into an SGPR. Reverse of writelane;
  // cross-lane so must use the native intrinsic.
  case CanonicalOp::V_READLANE_B32: {
    ParsedReg SrcReg = Op.srcReg(0);
    Value *Lane = Op.src(1);
    Lane = Ctx.B.CreateZExtOrTrunc(Lane, Ctx.I32Ty, "rdlane_idx");
    Value *Src = Ctx.Regs.readReg32(Ctx.B, SrcReg);
    Value *Val = nullptr;
    if (Ctx.Projection.sourceWaveScopedLaneOps()) {
      Value *LaneId = Ctx.emitLaneIdx();
      uint32_t SourceMask = Ctx.Isa.WaveSize - 1;
      Value *GroupBase = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(~SourceMask),
                                         "rdlane_source_wave_base");
      Value *SourceLane = Ctx.B.CreateAnd(Lane, Ctx.B.getInt32(SourceMask),
                                          "rdlane_source_lane");
      Value *TargetLane = Ctx.B.CreateOr(GroupBase, SourceLane,
                                         "rdlane_target_lane");
      Value *Addr = Ctx.B.CreateShl(TargetLane, Ctx.B.getInt32(2),
                                    "rdlane_bperm_addr");
      Function *Bperm = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
      Val = Ctx.B.CreateCall(Bperm, {Addr, Src}, "readlane_srcwave");
    } else {
      Function *Rl = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_readlane, {Ctx.I32Ty});
      Val = Ctx.B.CreateCall(Rl, {Src, Lane}, "readlane");
    }
    Ctx.writeReg32(Op.dst(), Val);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_mbcnt_lo_u32_b32 / v_mbcnt_hi_u32_b32 ----
  // Count set bits in src0 below the current lane.  For same-wave lifts the
  // raw intrinsic is exact.  For wave32 source -> wave64 target, however,
  // raw target `mbcnt.lo` would return popcount(src0[0:31]) for target lanes
  // 32..63, while the source instruction's lane id restarts at 0 in the
  // second modeled source wave.  Recompute the source-wave-local low-half
  // count from `lane_id mod W_s` in that case.
  case CanonicalOp::V_MBCNT_LO_U32_B32: {
    Value *Result = nullptr;
    if (Ctx.Isa.isWave32() && Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
      Value *LaneId = Ctx.emitLaneIdx();
      Value *SourceLane = Ctx.B.CreateAnd(
          LaneId, Ctx.B.getInt32(Ctx.Isa.WaveSize - 1), "mbcnt_source_lane");
      Value *LaneBit = Ctx.B.CreateShl(Ctx.B.getInt32(1), SourceLane,
                                       "mbcnt_lane_bit");
      Value *BelowMask = Ctx.B.CreateSub(LaneBit, Ctx.B.getInt32(1),
                                         "mbcnt_below_mask");
      Value *Masked = Ctx.B.CreateAnd(Op.src(0), BelowMask, "mbcnt_masked");
      Function *Ctpop = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::ctpop, {Ctx.I32Ty});
      Result = Ctx.B.CreateAdd(
          Ctx.B.CreateCall(Ctpop, {Masked}, "mbcnt_pop"), Op.src(1),
          "mbcnt_lo_srcwave");
    } else {
      Function *Mbcnt = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_mbcnt_lo, {});
      Result = Ctx.B.CreateCall(Mbcnt, {Op.src(0), Op.src(1)}, "mbcnt_lo");
    }
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_MBCNT_HI_U32_B32: {
    Function *Mbcnt = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_mbcnt_hi, {});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateCall(Mbcnt, {Op.src(0), Op.src(1)},
                                    "mbcnt_hi"));
    Hr.Handled = true;
    return Hr;
  }

  default:
    break;
  }
  return Hr;
}

} // namespace COMGR::hotswap
