//===- handle-sopk.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amdgpu-mode-hwreg.h"
#include "handlers.h"
#include "pipeline.h" // isStrictMode()

#include "SIDefines.h" // AMDGPU::Hwreg::Id
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

using namespace llvm;

namespace COMGR::hotswap {

// HWREG (s_getreg / s_setreg) policy -- direction-aware.
//
// The simm16 encoding for these opcodes is
// `id[5:0] | offset[10:6] | size_minus_1[15:11]`; only the id field
// selects which hardware register is being touched. EXEC is not a HWREG
// id (confirmed in AMDGPU::Hwreg::Id), so s_setreg can never reach EXEC;
// the SPE abort-gate in raiser.cpp remains exhaustive for EXEC writers.
//
// Two axes of decision per id:
//
//  * READ policy (`HwregRead`):
//      Zero  -- produce 0. Used for ids whose observable value is either
//              a hardware timer / state we can't reproduce but whose 0
//              is the architectural default (MODE) or a diagnostic read
//              the kernel's compute does not meaningfully act on
//              (STATUS, HW_ID, perf counters, shader cycles, ALLOC
//              descriptors, TBA/TMA/trap-handler bases, etc.).
//      Abort -- refuse to lower. Used for ids whose value the kernel
//              *might* actually consume to make compute decisions we
//              cannot reproduce faithfully (FLAT_SCR_*, MEM_BASES,
//              XNACK_MASK). Zero would be a silent lie.
//
//  * WRITE policy (`HwregWrite`):
//      Drop     -- silently discard. Used for ids whose observable
//                 state the transpiler does not model downstream
//                 (diagnostic / status / perf), so dropping a write
//                 leaves no gap in semantics SPE lifts.
//      WarnDrop -- discard but emit a stderr warning. Used for ids
//                 where a write *could* change observable compute
//                 semantics in the general case but where silently
//                 aborting would break the current test corpus
//                 (compiler-emitted MODE writes for FP flags that
//                 downstream ops happen not to consume in
//                 practice). Symmetric with the cross-wave
//                 warn-only policy. Tightening to Abort requires
//                 either a same-wave regression corpus or a proof
//                 that the downstream FP ops are insensitive to the
//                 discarded bits.
//      Abort    -- refuse to lower. Used for ids whose value *does*
//                 feed state we lift and where dropping would
//                 change lifted semantics: FLAT aperture
//                 (FLAT_SCR_*, MEM_BASES), XNACK retry semantics,
//                 trap handler bases.
//
// Unknown ids (not enumerated below) default to (Abort, Abort): the
// principled fail-closed choice. A future ISA adding a new load-
// bearing register is trapped loudly instead of silently miscompiling.
enum class HwregRead { Zero, Abort };
enum class HwregWrite {
  // Discard silently.  For ids whose state the transpiler does not
  // lift (diagnostic / perf / status); dropping is a no-op.
  Drop,
  // Discard but stderr-warn.  Legacy policy for cases where the
  // write COULD matter but refusing would break existing tests.
  // Prefer `Preserve` for new uses so MODE-sensitive kernels
  // lift faithfully; prefer `Abort` for cases the lifter can't
  // faithfully reproduce.  Kept for `Drop`-but-louder opcodes we
  // don't want to land as pure `Drop` yet.
  WarnDrop,
  // Faithfully re-emit the write via `@llvm.amdgcn.s.setreg` with
  // the exact same simm16 + value.  The backend lowers this 1:1 to
  // `s_setreg_imm32_b32 <same simm16>, <same value>` on the target.
  // Correct when (a) the bit position is architecturally stable
  // across source and target ISAs, and (b) the bit's effect on
  // compute is cross-architecturally equivalent (not a "rounding-
  // mode-on-gfx1250 / clamp-on-gfx942" kind of repurpose).
  //
  // Why faithful lift is preferred over `WarnDrop` even when the
  // specific bit Triton writes is observationally harmless on
  // gfx942:
  //
  //   * The `WarnDrop` policy is a silent-fallback-shaped default
  //     that the project's "fail loud" rule (`AGENTS.md`) explicitly
  //     flags as latent-miscompile territory for any FP-mode-
  //     sensitive downstream compute.  The comment block this enum
  //     replaces acknowledged the risk ("Tightening to Abort is a
  //     follow-up"); `Preserve` closes that gap in the correct
  //     direction (ABI-faithful) rather than by refusal.
  //   * If the source kernel's MODE bit is equivalent across ISAs
  //     (the common case for bits 0..22 -- standard FP round /
  //     denormal / IEEE, unchanged since gfx8), `Preserve` produces
  //     byte-exact behaviour versus dropping.
  //   * If the bit is gfx-generation-specific (bit 23 FP16_OVFL, bit 25
  //     REPLAY_MODE, and other high MODE fields), `Preserve` writes to the
  //     bit position; the target hardware's interpretation is
  //     authoritative.  If a kernel later surfaces as WRONG
  //     because of a bit-23+ repurpose, that is a NEW data point
  //     that justifies per-bit gating rather than reverting to a
  //     silent drop.
  //
  // Not a fix for any currently-known miscompile.  The
  // `topk_forward_bf16` silent miscompile flagged in commit
  // `7507185094` WAS empirically checked under `Preserve`: output
  // unchanged (gfx942's MODE.REPLAY_MODE write is a no-op for Triton's softmax
  // path at this shape; the bug is elsewhere -- see the
  // `topk_forward_bisect_*` recipes landed alongside this change
  // for the ongoing triage).  `Preserve` is a principled-improvement
  // change, independent of that triage.
  Preserve,
  // Refuse to lower.  Used for writes we cannot faithfully
  // reproduce (FLAT aperture bases, trap handler, XNACK retry) or
  // for unknown HWREG ids.
  Abort,
};

struct HwregPolicy {
  HwregRead Read;
  HwregWrite Write;
};

static HwregPolicy classifyHwreg(unsigned Id) {
  using namespace AMDGPU::Hwreg;
  switch (Id) {
  // --- Load-bearing: both directions abort ---------------------------------
  // FLAT_SCR is the aperture base for FLAT_* instructions; MEM_BASES sets
  // the scratch/private apertures; XNACK_MASK governs per-lane page-fault
  // retry. Reads cannot be faithfully reproduced (our IR does not carry
  // aperture config), writes must not be dropped (they would corrupt
  // addressing for subsequent FLAT ops we *do* lift).
  case ID_FLAT_SCR_LO:
  case ID_FLAT_SCR_HI:
  case ID_MEM_BASES:
  case ID_XNACK_MASK:
    return {HwregRead::Abort, HwregWrite::Abort};

  // --- MODE: read zero, write PRESERVE via llvm.amdgcn.s.setreg -----------
  // Reading MODE as 0 returns the architectural default (IEEE, round-
  // to-nearest-even, no FP exception flags, no FTZ).  Writing MODE
  // mutates FP rounding / FTZ / clamp / IEEE / various graphics-
  // context bits.  HIP-compiled and Triton-compiled compute kernels
  // routinely emit `s_setreg_imm32_b32 mode(offset, size), imm` in
  // their prologue -- e.g. gfx1250 kernels set MODE.REPLAY_MODE
  // (`isModeReplayMultiGroupWrite` in amdgpu-mode-hwreg.h) before the
  // first SMEM load.  The
  // `WarnDrop` policy silently discarded these writes; `Preserve`
  // is the faithful lift path.  See the `HwregWrite::Preserve`
  // enum-variant comment for the full rationale and the
  // principled-vs-silent-fallback argument.
  case ID_MODE:
    return {HwregRead::Zero, HwregWrite::Preserve};

  // --- Trap handler bases: read-as-zero, write-abort -----------------------
  // TBA_LO/HI and TMA_LO/HI set the trap-handler base / trap-memory
  // address. SPE doesn't model trap handlers, so a dropped write would
  // lose information we have no other channel to surface. Aborting
  // forces us to address this explicitly when a real kernel writes
  // them; reads return zero because compiler-emitted probes exist and
  // their result does not feed compute.
  case ID_TBA_LO:
  case ID_TBA_HI:
  case ID_TMA_LO:
  case ID_TMA_HI:
    return {HwregRead::Zero, HwregWrite::Abort};

  // --- Diagnostic / read-dominant: read zero, drop writes ------------------
  // Status, HW_ID*, allocation descriptors (set by the runtime at kernel
  // launch, not by compute), instruction-buffer status, performance
  // counters, shader cycle counters, packer / scheduling hints,
  // privileged status / exception / XCC ids. None of these feed compute
  // SPE lifts; reading zero is a defensible default; writes (if any
  // compiler ever emits one) are lost state we do not track.
  case ID_STATUS:
  case ID_TRAPSTS:
  case ID_HW_ID:
  case ID_HW_ID1:
  case ID_HW_ID2:
  case ID_GPR_ALLOC:
  case ID_LDS_ALLOC:
  case ID_IB_STS:
  case ID_IB_STS2:
  case ID_POPS_PACKER:
  case ID_SCHED_MODE:
  case ID_PERF_SNAPSHOT_DATA_gfx11:
  case ID_SHADER_CYCLES:
  case ID_SHADER_CYCLES_HI:
  case ID_DVGPR_ALLOC_LO:
  case ID_DVGPR_ALLOC_HI:
    return {HwregRead::Zero, HwregWrite::Drop};

  // --- Unknown: fail-closed in both directions -----------------------------
  default:
    return {HwregRead::Abort, HwregWrite::Abort};
  }
}

// Extract the hwreg id from a s_setreg/s_getreg simm16 operand.
static unsigned extractHwregId(int64_t Simm16) {
  return static_cast<unsigned>(Simm16 & 0x3f);
}

static Value *getSimm16(RaiseContext &Ctx, const DecodedInst &Di,
                        unsigned OperandIdx, HandlerResult &Hr) {
  if (OperandIdx >= Di.Inst.getNumOperands() || !Di.isImm(OperandIdx)) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "SOPK", "expected signed 16-bit immediate operand");
    return nullptr;
  }
  int64_t Raw = Di.getImm(OperandIdx);
  if (Raw < -32768 || Raw > 65535) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "SOPK", "signed 16-bit immediate operand out of range");
    return nullptr;
  }
  uint32_t Bits = static_cast<uint32_t>(Raw) & 0xffffu;
  int32_t Simm16 = (Bits & 0x8000u)
                       ? static_cast<int32_t>(Bits) - 0x10000
                       : static_cast<int32_t>(Bits);
  return ConstantInt::get(Ctx.I32Ty, Simm16, true);
}

HandlerResult handleSOPK(RaiseContext &Ctx, const DecodedInst &Di,
                         OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  // SOPK operand layout: two shapes here.
  //
  //   * `S_MOVK_I32` uses SOPK_32 (outs=$sdst, ins=$simm16): MCInst
  //     has two operands (sdst, simm16), logical src(0) = simm16.
  //   * `S_ADDK_I32` / `S_MULK_I32` use SOPK_32TIE (outs=$sdst,
  //     ins=$src0,$simm16) with $src0 tied to $sdst. The AMDGPU
  //     disassembler keeps the tied $src0 as a distinct MCOperand
  //     rather than collapsing it, and `decode.cpp::buildSrcMap`
  //     intentionally keeps tied-def operands in srcMap (it's listed
  //     in `KKnownTiedIn` -- see the audit comment there for the
  //     rationale). Result: for SOPK_32TIE opcodes, srcMap has TWO
  //     entries -- `src(0)` is the tied SGPR (same physical reg as
  //     sdst, i.e. the prior-dst value already readable via
  //     `readReg32(op.dst())`), and `src(1)` is the simm16
  //     immediate. The simm16 is sign-extended by hardware; using
  //     op.src(1)'s zero-extended ConstantInt would turn e.g. 0xd000
  //     into +53248 instead of -12288.
  //
  //   Pre-fix bug: the SOPK_32TIE handlers below read `op.src(0)`
  //   thinking it was the simm16, but it actually returned the tied
  //   SGPR. So `s_addk_co_i32 s0, 0x400` lifted to
  //   `add i32 %s0, %s0` (doubling) instead of
  //   `add i32 %s0, 1024` (incrementing by K). For layer-norm's
  //   loop counter, that doubled 0 stays at 0 forever, hanging the
  //   kernel.  `s_mulk_i32` had the same latent bug (no kernel in
  //   the corpus exercised it).
  if (Sop == CanonicalOp::S_MOVK_I32) {
    Value *Imm = getSimm16(Ctx, Di, Op.srcIdx(0), Hr);
    if (!Imm) return Hr;
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Imm);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_MULK_I32) {
    Value *Dst = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    Value *Imm = getSimm16(Ctx, Di, Op.srcIdx(1), Hr);
    if (!Imm) return Hr;
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(),
                        Ctx.B.CreateMul(Dst, Imm, "mulk"));
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::S_ADDK_I32) {
    Value *Dst = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    Value *Imm = getSimm16(Ctx, Di, Op.srcIdx(1), Hr);
    if (!Imm) return Hr;
    Value *Res = Ctx.B.CreateAdd(Dst, Imm, "addk");
    Ctx.Regs.writeReg32(Ctx.B, Op.dst(), Res);
    // SCC holds the signed carry-out/overflow bit. Gfx12+ names this
    // spelling `s_addk_co_i32`; use the signed overflow intrinsic so
    // consumers of SCC observe the same flag as the source instruction.
    auto *Ov = Ctx.B.CreateIntrinsic(Intrinsic::sadd_with_overflow, {Ctx.I32Ty},
                                     {Dst, Imm});
    Ctx.Regs.storeSCC(Ctx.B, Ctx.B.CreateExtractValue(Ov, 1));
    Hr.SccHandled = true;
    Hr.Handled = true;
    return Hr;
  }
  // SOPK compares: s_cmpk_XX_i32 / s_cmpk_XX_u32.
  //
  // Operand layout (SOPK_SCC class, SOPInstructions.td):
  //   (outs)                          ; empty -- no def
  //   (ins SReg_32:$sdst,             ; operand 0: sdst as SOURCE
  //        {s,u}16imm:$simm16)        ; operand 1: the immediate
  //
  // Both ins operands are sources. With `getNumDefs() == 0`,
  // `buildSrcMap` (decode.cpp) keeps BOTH in srcMap:
  //     srcMap[0] = 0 (the `$sdst` source register)
  //     srcMap[1] = 1 (the `$simm16` immediate)
  //
  // Pre-fix bug: the handler read `op.src(0)` as the immediate,
  // but `op.src(0)` is `$sdst` -- same SGPR that `readReg32(op.dst())`
  // already returned. So `icmp eq %sdst, %imm` reduced to
  // `icmp eq %sdst, %sdst` = always true, writing a trivially-set
  // SCC irrespective of the comparison's actual truth. Gfx12+
  // dropped the `s_cmpk_*_i32/u32` mnemonics entirely (the
  // SOPK_Real defm caps at gfx11), so gfx1250-source lifts cannot
  // reach this arm and the bug only surfaces on gfx9xx-source
  // kernels (AITER corpus). No AITER kernel in the currently-
  // tested subset exercises it, which is why the bug survived.
  //
  // Fix: read the immediate from `op.src(1)`, matching the
  // documented SOPK_SCC layout above.
  if (Sop == CanonicalOp::S_CMPK_EQ_I32 || Sop == CanonicalOp::S_CMPK_EQ_U32 ||
      Sop == CanonicalOp::S_CMPK_LG_I32 || Sop == CanonicalOp::S_CMPK_LG_U32 ||
      (Sop >= CanonicalOp::S_CMPK_GE_I32 && Sop <= CanonicalOp::S_CMPK_LT_U32)) {
    Value *Sdst = Ctx.Regs.readReg32(Ctx.B, Op.dst());
    Value *Imm = Op.src(1);  // $simm16; op.src(0) aliases sdst
    Value *Cmp = nullptr;
    if (Sop == CanonicalOp::S_CMPK_EQ_I32 || Sop == CanonicalOp::S_CMPK_EQ_U32)
      Cmp = Ctx.B.CreateICmpEQ(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_LG_I32 || Sop == CanonicalOp::S_CMPK_LG_U32)
      Cmp = Ctx.B.CreateICmpNE(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_GT_I32)
      Cmp = Ctx.B.CreateICmpSGT(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_GE_I32)
      Cmp = Ctx.B.CreateICmpSGE(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_LT_I32)
      Cmp = Ctx.B.CreateICmpSLT(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_LE_I32)
      Cmp = Ctx.B.CreateICmpSLE(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_GT_U32)
      Cmp = Ctx.B.CreateICmpUGT(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_GE_U32)
      Cmp = Ctx.B.CreateICmpUGE(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_LT_U32)
      Cmp = Ctx.B.CreateICmpULT(Sdst, Imm, "scmpk");
    else if (Sop == CanonicalOp::S_CMPK_LE_U32)
      Cmp = Ctx.B.CreateICmpULE(Sdst, Imm, "scmpk");
    if (Cmp) {
      Ctx.Regs.storeSCC(Ctx.B, Cmp);
      Hr.SccHandled = true;
      Hr.Handled = true;
      return Hr;
    }
  }
  // s_getreg_b32 / s_setreg_*: read or write one field of a hardware
  // configuration register. Both encodings carry the HWREG selector as a
  // simm16 at MCInst operand index 1 (TableGen `ins` order is
  // `sdst, simm16` for GETREG/SETREG_B32 and `imm, simm16` for
  // SETREG_IMM32_B32). Policy is driven by `classifyHwreg` above --
  // direction-aware per id. See the classifier's docstring for the
  // rationale behind each entry.
  if (Sop == CanonicalOp::S_GETREG_B32 || Sop == CanonicalOp::S_SETREG_B32 ||
      Sop == CanonicalOp::S_SETREG_IMM32_B32) {
    const unsigned Simm16OpIdx = 1;
    if (Simm16OpIdx >= Di.numOps() || !Di.isImm(Simm16OpIdx)) {
      errs() << "transpiler: " << Di.Mnemonic
             << " has unexpected operand layout (simm16 not at op 1) -- "
                "refusing to model it silently.\n";
      return Hr;
    }
    unsigned HwregId = extractHwregId(Di.getImm(Simm16OpIdx));
    HwregPolicy Policy = classifyHwreg(HwregId);

    if (Sop == CanonicalOp::S_GETREG_B32) {
      if (Policy.Read == HwregRead::Abort) {
        errs() << "transpiler: " << Di.Mnemonic
               << " reads load-bearing or unknown HWREG id=" << HwregId
               << " -- refusing to lower. The transpiler does not carry "
                  "this register's value and returning zero would be a "
                  "silent lie that the kernel's compute may act on.\n";
        return Hr;
      }
      Ctx.Regs.writeReg32(Ctx.B, Op.dst(), ConstantInt::get(Ctx.I32Ty, 0));
      Hr.Handled = true;
      return Hr;
    }

    // S_SETREG path (both S_SETREG_B32 and S_SETREG_IMM32_B32).
    if (Policy.Write == HwregWrite::Abort) {
      errs() << "transpiler: " << Di.Mnemonic
             << " writes load-bearing or unknown HWREG id=" << HwregId
             << " -- refusing to lower. Dropping this write would silently "
                "change compute semantics (FLAT aperture, trap handler, "
                "XNACK retry, …) that subsequent lifted instructions "
                "rely on.\n";
      return Hr;
    }
    if (Policy.Write == HwregWrite::Preserve) {
      // Re-emit the write via `@llvm.amdgcn.s.setreg(i32 immarg
      // hwmode, i32 value)` so the target backend lowers it to
      // `s_setreg_imm32_b32 <same simm16>, <same value>` byte-for-
      // byte.  The simm16 IS an ImmArg on the intrinsic declaration,
      // so we must pass it as a `ConstantInt`.  The value operand
      // differs by opcode: S_SETREG_IMM32_B32 has the value as an
      // immediate at MCInst op 0 (`imm` in the TableGen `ins`
      // ordering); S_SETREG_B32 has the value in a scalar register
      // at MCInst op 0 (`sdst`), which we read through `op.src(0)`.
      int64_t Simm16 = Di.getImm(Simm16OpIdx);
      Value *ValArg = nullptr;
      if (Sop == CanonicalOp::S_SETREG_IMM32_B32) {
        // MCInst operand 0 is the i32 immediate value; the simm16 is
        // at op 1.  Read the immediate directly; DO NOT go through
        // op.src(0) because that reads SGPR contents, not an
        // immediate operand.
        if (Di.numOps() < 2 || !Di.isImm(0)) {
          errs() << "transpiler: " << Di.Mnemonic
                 << " has unexpected operand layout (value not "
                    "immediate at op 0) -- refusing to lower.\n";
          return Hr;
        }
        int64_t ImmVal = Di.getImm(0);
        ValArg = ConstantInt::get(Ctx.I32Ty, ImmVal);

        // On gfx1250 (1024-addressable-VGPRs targets), any
        // S_SETREG_IMM32_B32 targeting HW_REG_MODE writes the VGPR
        // most-significant-bit (MSB) selectors from imm32 bits [12:19],
        // regardless of the field offset/size. LLVM's
        // AMDGPULowerVGPREncoding pass folds the current MSB state into the
        // prologue MODE setreg the kernel already emits (e.g. imm 0x1001,
        // where bit 0 is the named field and the high byte carries the MSB
        // selectors). We must mirror that here, otherwise later VGPR operand
        // encodings that depend on a non-zero MSB resolve to the wrong
        // physical register. On older targets bits [12:19] are ordinary
        // FP-mode fields and must not be treated as VGPR_MSB.
        if (Ctx.Isa.Has1024AddressableVGPRs && HwregId == amdgpu::HwregIdMode) {
          // The MODE byte is (dst, src0, src1, src2); VgprMsBs tracks the
          // S_SET_VGPR_MSB layout (src0, src1, src2, dst). Right-rotate by
          // VgprMsbModeToSetregRotate to convert (inverse of the pass's
          // convertModeToSetregFormat, which rotates left).
          constexpr unsigned Rot = amdgpu::ModeReg::VgprMsbModeToSetregRotate;
          uint8_t ModeFmt = static_cast<uint8_t>(
              (ImmVal >> amdgpu::ModeReg::VgprMsbLowBit) &
              amdgpu::ModeReg::VgprMsbByteMask);
          Ctx.VgprMsBs = llvm::rotr<uint8_t>(ModeFmt, Rot);
        }
      } else {
        // S_SETREG_B32: value is an SGPR at MCInst op 0.  Reading
        // through op.src(0) returns the SSA i32 for that SGPR's
        // current value, which is exactly what we need to pass.
        ValArg = Op.src(0);
        if (ValArg->getType() != Ctx.I32Ty)
          ValArg = Ctx.B.CreateBitOrPointerCast(ValArg, Ctx.I32Ty);
        // On gfx1250, any MODE setreg captures VGPR_MSB from the operand's
        // bits [12:19] unconditionally. For S_SETREG_B32 the value is a
        // dynamic SGPR, so we cannot statically update Ctx.VgprMsBs. Refuse
        // writes whose field overlaps bits [12:19] to prevent silently
        // decoding subsequent VGPR operands with stale bank-selector state.
        if (Ctx.Isa.Has1024AddressableVGPRs && HwregId == amdgpu::HwregIdMode) {
          const amdgpu::SetregField Field = amdgpu::decodeSetregSimm16(Simm16);
          constexpr unsigned VgprMsbLo = amdgpu::ModeReg::VgprMsbLowBit;
          constexpr unsigned VgprMsbHi = VgprMsbLo + 8u - 1u;
          if (Field.Offset <= VgprMsbHi &&
              Field.Offset + Field.SizeBits - 1u >= VgprMsbLo) {
            errs() << "transpiler: " << Di.Mnemonic
                   << " writes MODE with field overlapping VGPR_MSB bits"
                      " [12:19] -- refusing; cannot statically track"
                      " SGPR-sourced VGPR_MSB change on gfx1250.\n";
            return Hr;
          }
        }
      }
      Function *SetregFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_s_setreg);
      Ctx.B.CreateCall(SetregFn,
                       {ConstantInt::get(Ctx.I32Ty, Simm16), ValArg});
      Hr.Handled = true;
      return Hr;
    }
    if (Policy.Write == HwregWrite::WarnDrop) {
      // Strict mode (`HSA_HOTSWAP_STRICT=1`, see pipeline.h) refuses
      // these writes structurally instead of warn-and-continue. The
      // refusal is the honest answer for any caller (e.g. the corpus
      // runner) that wants UNSUPPORTED verdicts in place of latent
      // wrong-result hazards. The non-strict path stays unchanged so
      // existing GPU tests that emit `s_setreg_imm32_b32 mode, imm`
      // and pass bit-exactly continue to pass.
      if (isStrictMode()) {
        errs() << "transpiler: " << Di.Mnemonic
               << " writes HWREG id=" << HwregId
               << " (MODE / FP-state-bearing register) -- refusing under "
                  "HSA_HOTSWAP_STRICT. Dropping the write would silently "
                  "change FP rounding / denormal / IEEE / FTZ semantics if "
                  "downstream compute consumes those bits.\n";
        Hr.Failure = RaiseFailure::strictUnsafeLowering(
            Di, "HWREG_MODE_write",
            "hotswap (strict): MODE-register write would be silently "
            "dropped; kernel may rely on FP rounding / denormal / IEEE / "
            "FTZ bits being changed");
        return Hr;
      }
      errs() << "transpiler: WARNING: " << Di.Mnemonic
             << " writes HWREG id=" << HwregId
             << " (MODE or similar FP-state-bearing register). Dropping "
                "the write to stay within the existing SPE lifting shape; "
                "this is observationally correct only when downstream "
                "compute is insensitive to the written bits. Tighten to "
                "Abort once the corpus allows it.\n";
    }
    Hr.Handled = true;
    return Hr;
  }
  return Hr;
}

} // namespace COMGR::hotswap
