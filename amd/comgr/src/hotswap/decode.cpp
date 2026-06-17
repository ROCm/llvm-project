//===- decode.cpp - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode.h"

#include "amdgpu-formats.h"
#include "decoded-inst.h"
#include "mc-state.h"
#include "opcode-map.h"
#include "canonical-op.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, VCC, SCC, ...
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Build the logical-source view of an MCInst. Walks `desc.operands()` and
// classifies each operand using TableGen-generated metadata only:
//
//   * Operand types carrying the AMDGPU-specific `OPERAND_INPUT_MODS`
//     tag are VOP3 source modifiers (neg/abs/opsel packed as an imm).
//     They attach to the next logical source via `modMap`.
//   * DPP/SDWA encodings carry a tied "old" input (fallback value for
//     inactive lanes, named `$old` or `$vdst_in` in TableGen). In our
//     all-lanes-active scalar model that slot is never read, so we skip
//     it. Not every tied-to-def operand is a fallback -- VOP2 MAC forms
//     (v_fmac_f32, v_mac_f32, v_dot2c_*) tie `$src2` to the dst and
//     atomics tie `$vdata_in`/`$sdst_in`/`$addr_in`; in those cases the
//     tied operand is a real accumulator/read-modify input and must stay
//     in srcMap. We therefore select on the named-operand id rather than
//     the TIED_TO bit alone.
//   * Everything else is a logical source recorded in MCInst order.
void buildSrcMap(DecodedInst &Di, const MCInstrDesc &Desc) {
  const MCInst &Inst = Di.Inst;
  unsigned Opc = Inst.getOpcode();
  int OldIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::old);
  int VdstInIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst_in);
  auto OpInfos = Desc.operands();
  unsigned PendingModIdx = UINT_MAX;
  for (unsigned I = Di.FirstSrcIdx; I < Inst.getNumOperands(); ++I) {
    if (I < OpInfos.size() &&
        OpInfos[I].OperandType == OPERAND_INPUT_MODS) {
      PendingModIdx = I;
      continue;
    }
    if (static_cast<int>(I) == OldIdx || static_cast<int>(I) == VdstInIdx) {
      PendingModIdx = UINT_MAX;
      continue;
    }
    if (Di.NumSrcs >= DecodedInst::KMaxSrcs)
      report_fatal_error("transpiler: DecodedInst::KMaxSrcs exceeded; "
                         "bump KMaxSrcs to match the widest LLVM operand "
                         "list");
    Di.SrcMap[Di.NumSrcs] = I;
    Di.ModMap[Di.NumSrcs] = PendingModIdx;
    Di.NumSrcs++;
    PendingModIdx = UINT_MAX;
  }
}

// Drift check A: every tied-to-def operand on this instruction must have
// an OpName we've explicitly classified. If LLVM introduces a new
// tied-input OpName we haven't audited (so we don't know whether to skip
// or keep it), stop and make a human decide. `KKnownTiedIn` is the
// exhaustive audit as of this commit. Two semantic categories:
//
//   skipped-as-fallback (DPP/SDWA inactive-lane value; never read in the
//                       all-lanes-active scalar model):
//     `old`, `vdst_in`.
//
//   kept-as-real-input (read-modify accumulator, atomic compare, or
//                      MAC-style third source; the instruction
//                      semantically reads the prior def value):
//     `sdst_in`, `vdata_in`, `addr_in`, `srcTiedDef`,
//     `src0`, `src1`, `src2`,
//     `src0X`, `src0Y`, `src2X`, `src2Y`,
//     `vsrc2X`, `vsrc2Y`.
//
// CAVEAT: this list reflects whether the *handler* should treat the
// tied operand as a real read (yes for `sdst_in`/`vdata_in`/etc.; no
// for `old`/`vdst_in`). It does NOT promise that the AMDGPU
// disassembler will materialise an MCOperand for that slot -- for
// SOP1 `sdst_in` (S_BITSET0/1_B{32,64}) and SOP1 `S_CMOV_B{32,64}`
// the disassembler collapses the tied slot and produces only
// `(sdst, src0)`, so `srcMap` won't contain an entry for the prior-
// dst read. Handlers in those cases must fetch the prior value
// directly via `ctx.Regs.readReg{32,64}(op.dst())`. For
// `vdata_in` / `addr_in` / `srcTiedDef` (atomics, MAC accumulators)
// the disassembler does emit a full MCOperand and the handler reads
// it through the normal `op.src(N)` path.
//
// srcN and VOPD variants all appear here because SOPK `S_ADDK_I32`
// ties `$src0`, SOP2 `sdst,sdst_in` variants may also surface `$src0`,
// VALU MAC forms tie `$src2`, and VOPD3 FMAC halves tie `$src2X` /
// `$src2Y` (plus potentially the separate VOPD3 third source).
void driftCheckTiedIn(const DecodedInst &Di, const MCInstrDesc &Desc) {
  static constexpr AMDGPU::OpName KKnownTiedIn[] = {
      AMDGPU::OpName::old,        AMDGPU::OpName::vdst_in,
      AMDGPU::OpName::sdst_in,    AMDGPU::OpName::vdata_in,
      AMDGPU::OpName::addr_in,    AMDGPU::OpName::srcTiedDef,
      AMDGPU::OpName::src0,       AMDGPU::OpName::src1,
      AMDGPU::OpName::src2,       AMDGPU::OpName::src0X,
      AMDGPU::OpName::src0Y,      AMDGPU::OpName::src2X,
      AMDGPU::OpName::src2Y,      AMDGPU::OpName::vsrc2X,
      AMDGPU::OpName::vsrc2Y,
  };
  const MCInst &Inst = Di.Inst;
  unsigned Opc = Inst.getOpcode();
  for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
    int Tied = Desc.getOperandConstraint(I, MCOI::TIED_TO);
    if (Tied < 0)
      continue;
    // Only flag operands tied to a def. Use-to-use ties exist in LLVM's
    // constraint system but are not relevant to the fallback/accumulator
    // distinction this check protects.
    if (static_cast<unsigned>(Tied) >= Desc.getNumDefs())
      continue;
    bool Known = false;
    for (AMDGPU::OpName N : KKnownTiedIn) {
      if (static_cast<int>(I) == AMDGPU::getNamedOperandIdx(Opc, N)) {
        Known = true;
        break;
      }
    }
    if (!Known) {
      std::string Msg;
      raw_string_ostream Os(Msg);
      Os << "transpiler: tied-to-def operand has an OpName not in the "
            "audited set -- classify explicitly (fallback to skip vs. real "
            "input to keep) before proceeding for " << Di.RawMnemonic
         << " (opcode=" << Opc << "): index=" << I
         << ", tiedTo=" << Tied
         << ", numDefs=" << Desc.getNumDefs()
         << ", numOps=" << Inst.getNumOperands();
      report_fatal_error(StringRef(Msg));
    }
  }
}

// Drift check B: for every opcode that exposes `srcN` / `srcN_modifiers`
// naming (VALU, VOPC, SOP1/SOP2, a handful of scalar forms), the first
// N entries of srcMap / modMap must agree with LLVM's named-operand
// table. Catches operand-layout drift for the large majority of opcodes
// -- but notably NOT for DS / MUBUF / FLAT / SMEM / image encodings,
// which don't use srcN naming; those formats are only protected by the
// walk's correctness and drift check A.
//
// Scaled MFMA instructions (ScaledMAIInst in TableGen) append
// src0_modifiers / src1_modifiers AFTER all source operands, not
// interleaved as in VOP3. The walk can't discover them because it only
// looks for OPERAND_INPUT_MODS *before* each source. We repair the
// modMap from LLVM's authoritative named-operand table here, but ONLY
// for MAI-format instructions so we don't silently mask future layout
// drift in other formats.
void driftCheckSrcN(DecodedInst &Di, const MCInstrDesc &Desc) {
  static constexpr AMDGPU::OpName KSrcNames[] = {
      AMDGPU::OpName::src0, AMDGPU::OpName::src1, AMDGPU::OpName::src2};
  static constexpr AMDGPU::OpName KModNames[] = {
      AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
      AMDGPU::OpName::src2_modifiers};

  auto ReportErr = [&](const Twine &Prefix, int Index, int Ours,
                       int Expected) -> void {
    std::string Msg;
    raw_string_ostream Os(Msg);
    Os << Prefix << " for " << Di.RawMnemonic
       << " (opcode=" << Di.Inst.getOpcode() << "): index=" << Index
       << ", srcMap/modMap=" << Ours << ", named=" << Expected
       << ", numSrcs=" << Di.NumSrcs
       << ", numDefs=" << Desc.getNumDefs()
       << ", numOps=" << Di.Inst.getNumOperands();
    report_fatal_error(StringRef(Msg));
  };

  unsigned Opc = Di.Inst.getOpcode();

  // MADMK/FMAMK exception: `v_fmamk_f32` and scalar `s_fmamk_f32`
  // place the 32-bit literal BETWEEN src0 and src1:
  //
  //   VOP_MADMK (VOP2Instructions.td): (src0, K-imm, src1)
  //   S_FMAMK_F32 (SOPInstructions.td): (src0, KImmFP32, src1)
  //
  // The natural positional walk in `buildSrcMap` produces
  // srcMap = [src0, K-imm, src1], which matches the corresponding handlers'
  // expectation that logical source order follows MC operand order.
  //
  // The strict `srcMap[k] == OpName::srcN` invariant breaks for
  // this layout because OpName::src1 lives at MCInst index 3, not
  // srcMap[1] = 2. The drift is INTENTIONAL: handlers index by
  // MCInst order, not OpName order, and MADMK is a known stable
  // form. Skip the strict srcN-position check at the AFFECTED
  // index (k=1, the src1 slot) for this signature; k=0 (src0)
  // still passes naturally and k=2 returns -1 (no src2) so the
  // outer loop breaks before reaching it.
  //
  // Detection: the opcode exposes both `OpName::imm` and
  // `OpName::src0` / `OpName::src1`, with the imm operand index
  // strictly between src0 and src1. (Compare to MADAK/FMAAK forms like
  // `v_fmaak_f32` / `s_fmaak_f32`, whose layouts are `(src0, src1, K-imm)` --
  // K trailing -- where the positional walk happens to coincide with OpName
  // order and the drift check passes naturally.)
  //
  // The modifier-map check below remains in force; MADMK has no
  // src{0,1,2}_modifiers operands, so the loop's modMap branch
  // simply finds expected=-1 and our=-1, agreement.
  int ImmIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::imm);
  int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  int Src1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1);
  bool IsMadmk = ImmIdx >= 0 && Src0Idx >= 0 && Src1Idx >= 0 &&
                 Src0Idx < ImmIdx && ImmIdx < Src1Idx;

  for (unsigned K = 0; K < 3; ++K) {
    int NamedSrc = AMDGPU::getNamedOperandIdx(Opc, KSrcNames[K]);
    if (NamedSrc < 0)
      break;
    int OurSrc = (K < Di.NumSrcs) ? static_cast<int>(Di.SrcMap[K]) : -1;
    // Skip ONLY the genuinely-affected index (k=1) for MADMK. k=0
    // (src0) still receives the strict check, so a hypothetical
    // future drift in src0's MCInst position is still caught even
    // for MADMK opcodes.
    bool SkipThis = IsMadmk && K == 1;
    if (!SkipThis && OurSrc != NamedSrc)
      ReportErr("transpiler: srcMap disagrees with OpName::srcN table",
                static_cast<int>(K), OurSrc, NamedSrc);
    int NamedMod = AMDGPU::getNamedOperandIdx(Opc, KModNames[K]);
    int OurMod =
        (Di.ModMap[K] == UINT_MAX) ? -1 : static_cast<int>(Di.ModMap[K]);
    int ExpectedMod = (NamedMod < 0) ? -1 : NamedMod;
    if (OurMod != ExpectedMod) {
      bool IsMai = Di.TsFlags & SIInstrFlags::IsMAI;
      if (IsMai && NamedMod >= 0 && OurMod == -1) {
        Di.ModMap[K] = static_cast<unsigned>(NamedMod);
      } else {
        ReportErr(
            "transpiler: modMap disagrees with OpName::srcN_modifiers table",
            static_cast<int>(K), OurMod, ExpectedMod);
      }
    }
  }
}

// Identify implicit defs of wave-mask / condition-flag registers via
// identity constants rather than register-name string matches. We
// normalise through `mc2PseudoReg` first, which strips subtarget
// suffixes (``_gfxNplus``) and converts aliases to their canonical
// pseudo-register id -- same pattern used by `parseReg`.
void classifyImplicitDefs(DecodedInst &Di, const MCInstrDesc &Desc) {
  for (MCPhysReg R : Desc.implicit_defs()) {
    llvm::MCRegister Reg = AMDGPU::mc2PseudoReg(R);
    switch (Reg) {
    case AMDGPU::SCC:
      Di.DefsScc = true;
      break;
    case AMDGPU::VCC:
    case AMDGPU::VCC_LO:
    case AMDGPU::VCC_HI:
      Di.DefsVcc = true;
      break;
    case AMDGPU::EXEC:
    case AMDGPU::EXEC_LO:
    case AMDGPU::EXEC_HI:
      Di.DefsExec = true;
      break;
    default:
      break;
    }
  }
}

// Decode the `scale_offset` bit out of the CPol operand once, so handlers
// can consume a typed boolean instead of string-searching the disassembled
// `fullText`. gfx12+ FLAT/GLOBAL forms carry the bit in `cpol`; earlier
// ISAs have no `cpol` operand and the flag is inherently absent
// (`hasScaleOffset` stays false).
void decodeScaleOffset(DecodedInst &Di) {
  const MCInst &Inst = Di.Inst;
  int CpolIdx =
      AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::cpol);
  if (CpolIdx < 0 ||
      static_cast<unsigned>(CpolIdx) >= Inst.getNumOperands())
    return;
  const MCOperand &Mop = Inst.getOperand(static_cast<unsigned>(CpolIdx));
  if (!Mop.isImm())
    return;
  int64_t Cpol = Mop.getImm();
  Di.HasScaleOffset = (Cpol & AMDGPU::CPol::SCAL) != 0;
}

// Decode DPP16 modifier operands (dpp_ctrl / row_mask / bank_mask /
// bound_ctrl) so the raiser can lift DPP-modified VALU ops through
// `llvm.amdgcn.update.dpp`. Sets `di.hasDpp = true` only when every
// DPP16 operand is present and immediate-typed.
//
// Preconditions:
//   - `di.tsFlags` is populated from the ORIGINAL (pre-canonicalisation)
//     MCInstrDesc -- the DPP bit here is the authoritative signal that
//     SOME DPP form is in play, but it does NOT distinguish DPP16 from
//     DPP8 (both `VOP_DPP8_Base` and VOP_DPP set `let DPP = 1`, see
//     VOPInstructions.td).
//
// DPP8 handling (present corpus: 0 instances, but architecturally
// possible): DPP8 encodes an 8-lane permutation as a single `OpName::
// dpp8` operand and has NO `dpp_ctrl` / `row_mask` / `bank_mask` /
// `bound_ctrl`. When we detect DPP8 (named operand `dpp8` exists) we
// leave `di.hasDpp` false; the classifier's DppCrossLane site will
// then mark the kernel as `rewriteImplemented = false` (pending P5
// extension to `llvm.amdgcn.mov.dpp8`), so the raiser refuses loudly
// rather than crashing on a partially-populated DPP modifier set.
//
// `fi` (fetch-inactive / fetch-invalid) is decoded when the DPP16
// operand exists. `llvm.amdgcn.update.dpp` does not take FI, so FI
// sites refuse before handler emission (via the cross-wave
// obstruction classifier) or at the DPP wrapper for same-wave raises.
// This keeps the ordinary DPP16 path representable while making the
// semantic gap explicit.
void decodeDppModifiers(DecodedInst &Di) {
  if (!(Di.TsFlags & SIInstrFlags::DPP))
    return;
  const MCInst &Inst = Di.Inst;
  const unsigned Opc = Inst.getOpcode();
  // Detect DPP8 form by presence of the `dpp8` named operand. If this
  // is a DPP8 instruction, leave `hasDpp` false -- see the header
  // comment for the classifier-refusal contract.
  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::dpp8) >= 0)
    return;
  auto ImmOpt = [&](AMDGPU::OpName Name) -> std::optional<int64_t> {
    int Idx = AMDGPU::getNamedOperandIdx(Opc, Name);
    if (Idx < 0 || static_cast<unsigned>(Idx) >= Inst.getNumOperands())
      return std::nullopt;
    const MCOperand &Mop = Inst.getOperand(static_cast<unsigned>(Idx));
    if (!Mop.isImm())
      return std::nullopt;
    return Mop.getImm();
  };
  auto Ctrl = ImmOpt(AMDGPU::OpName::dpp_ctrl);
  auto RowMask = ImmOpt(AMDGPU::OpName::row_mask);
  auto BankMask = ImmOpt(AMDGPU::OpName::bank_mask);
  auto BoundCtrl = ImmOpt(AMDGPU::OpName::bound_ctrl);
  auto Fi = ImmOpt(AMDGPU::OpName::fi);
  if (!Ctrl || !RowMask || !BankMask || !BoundCtrl) {
    // MCInstrDesc declared DPP and it is not a DPP8 variant, yet the
    // MCInst operand list is missing one of the four DPP16 modifier
    // fields. This is a decoder-vs-tblgen drift situation -- fail
    // loudly rather than emit IR with default (possibly wrong)
    // values. DPP8 was already filtered above, so we only reach here
    // on a genuinely unrecognised DPP form.
    std::string Msg;
    raw_string_ostream Os(Msg);
    Os << "decodeDppModifiers: TSFlags::DPP is set for '" << Di.RawMnemonic
       << "' (opcode=" << Opc
       << ") with no OpName::dpp8 operand, yet at least one of "
          "{dpp_ctrl, row_mask, bank_mask, bound_ctrl} is missing or "
          "not an immediate. LLVM likely added a new DPP variant "
          "whose operand layout this decoder does not yet recognise; "
          "extend decodeDppModifiers.";
    report_fatal_error(Os.str().c_str());
  }
  Di.HasDpp = true;
  Di.DppCtrl = static_cast<uint16_t>(*Ctrl & 0xFFFF);
  Di.DppRowMask = static_cast<uint8_t>(*RowMask & 0xF);
  Di.DppBankMask = static_cast<uint8_t>(*BankMask & 0xF);
  Di.DppBoundCtrl = (*BoundCtrl) != 0;
  Di.DppFi = Fi && *Fi != 0;
}

// Decode the 16-bit `OpName::offset` immediate of `ds_swizzle_b32`
// into `di.dsSwizzleImm` so the obstruction classifier and the DS
// handler share a single canonical extraction point. Mirrors the
// `decodeDppModifiers` pattern: decode-time field population, no
// per-call MCInst probing in downstream consumers.
//
// Only fires for `CanonicalOp::DS_SWIZZLE_B32`. For every other instruction
// `hasDsSwizzleImm` stays false and `dsSwizzleImm` is meaningless;
// consumers MUST gate on `hasDsSwizzleImm`.
//
// Soundness: refuses to populate the field if the operand is missing,
// non-immediate, or outside the unsigned 16-bit range. The classifier
// treats `!hasDsSwizzleImm` as "rewriteImplemented = false" so the
// kernel refuses loudly with a malformed-disassembly diagnostic
// rather than silently truncating a wider value to uint16_t (which
// could land in either the QUAD_PERM or BITMASK_PERM safe envelope
// and cause a silent miscompile).
void decodeDsSwizzleImm(DecodedInst &Di) {
  if (Di.CanonOp != CanonicalOp::DS_SWIZZLE_B32)
    return;
  const MCInst &Inst = Di.Inst;
  int Idx = AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                        AMDGPU::OpName::offset);
  if (Idx < 0 || static_cast<unsigned>(Idx) >= Inst.getNumOperands())
    return;
  const MCOperand &Mop = Inst.getOperand(static_cast<unsigned>(Idx));
  if (!Mop.isImm())
    return;
  int64_t Raw = Mop.getImm();
  if (Raw < 0 || Raw > 0xFFFF)
    return;
  Di.DsSwizzleImm = static_cast<uint16_t>(Raw);
  Di.HasDsSwizzleImm = true;
}

// Pull every branch-target offset out of a branch instruction's
// immediates and insert the resulting byte offsets into `blockStarts`.
// Signed 16-bit PC-relative offset * 4 bytes, relative to the
// instruction's successor (off + 4, not off + instSize -- matches the
// hardware encoding definition).

void failVopdDecode(const DecodedInst &Di, const Twine &Detail) {
  std::string Msg;
  raw_string_ostream Os(Msg);
  Os << "decodeVopd: " << Detail << " for '" << Di.RawMnemonic
     << "' (opcode=" << Di.Inst.getOpcode()
     << ", numOps=" << Di.Inst.getNumOperands() << ")";
  report_fatal_error(Os.str().c_str());
}

int findRegIndexInClass(const MCRegisterClass &RC, MCRegister Reg) {
  int Idx = 0;
  for (MCRegister R : RC) {
    if (R == Reg)
      return Idx;
    ++Idx;
  }
  return -1;
}

unsigned computeRegWidthDwords(const MCRegisterInfo &MRI, MCRegister Reg) {
  if (!Reg)
    return 1;
  unsigned Width = 1;
  for (unsigned Sub = AMDGPU::sub1; Sub < MRI.getNumSubRegIndices(); ++Sub) {
    MCRegister R = MRI.getSubReg(Reg, Sub);
    if (!R)
      break;
    ++Width;
  }
  return Width;
}

void classifyVopdRegSource(DecodedInst &Di, DecodedInst::VopdSource &Src,
                           const MCRegisterInfo &MRI, MCRegister Reg) {
  Src.Reg = Reg;
  Src.Width = computeRegWidthDwords(MRI, Reg);
  MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
  if (!Lane)
    Lane = Reg;
  Lane = AMDGPU::mc2PseudoReg(Lane);

  switch (Lane) {
  case AMDGPU::VCC_LO:
  case AMDGPU::VCC_HI:
    Src.SrcKind = DecodedInst::VopdSource::Kind::VCC;
    return;
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
    Src.SrcKind = DecodedInst::VopdSource::Kind::EXEC;
    Src.BaseIdx = (Lane == AMDGPU::EXEC_HI) ? 1 : 0;
    return;
  case AMDGPU::SCC:
    Src.SrcKind = DecodedInst::VopdSource::Kind::SCC;
    return;
  case AMDGPU::M0:
    Src.SrcKind = DecodedInst::VopdSource::Kind::M0;
    return;
  default:
    break;
  }

  unsigned Enc = MRI.getEncodingValue(Reg);
  unsigned HwIdx = Enc & AMDGPU::HWEncoding::REG_IDX_MASK;
  if (Enc & AMDGPU::HWEncoding::IS_AGPR) {
    Src.SrcKind = DecodedInst::VopdSource::Kind::AGPR;
    Src.BaseIdx = HwIdx;
    return;
  }
  if (Enc & AMDGPU::HWEncoding::IS_VGPR) {
    Src.SrcKind = DecodedInst::VopdSource::Kind::VGPR;
    Src.BaseIdx = HwIdx;
    return;
  }

  const MCRegisterClass &TTMP32 =
      MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  if (int Idx = findRegIndexInClass(TTMP32, Lane); Idx >= 0) {
    Src.SrcKind = DecodedInst::VopdSource::Kind::TTMP;
    Src.BaseIdx = Idx;
    return;
  }

  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Lane)) {
    Src.SrcKind = DecodedInst::VopdSource::Kind::SGPR;
    Src.BaseIdx = HwIdx;
    return;
  }

  failVopdDecode(Di, Twine("unsupported VOPD register source '") +
                         MRI.getName(Reg) + "'");
}

void decodeVopdSource(DecodedInst &Di, DecodedInst::VopdHalf &Half,
                      const AMDGPU::VOPD::ComponentInfo &Info,
                      unsigned CompSrcIdx, const MCRegisterInfo &MRI) {
  const MCInst &Inst = Di.Inst;
  unsigned McIdx = Info.getIndexOfSrcInMCOperands(CompSrcIdx, Di.IsVopd3);
  if (McIdx >= Inst.getNumOperands())
    failVopdDecode(Di, Twine("component source index out of MCInst range: src") +
                           Twine(CompSrcIdx) + " -> operand " + Twine(McIdx));

  if (static_cast<int>(McIdx) == Info.getBitOp3OperandIdx()) {
    const MCOperand &Mop = Inst.getOperand(McIdx);
    if (!Mop.isImm())
      failVopdDecode(Di, "bitop3 operand is not an immediate");
    int64_t Raw = Mop.getImm();
    if (Raw < 0 || Raw > 0xff)
      failVopdDecode(Di, Twine("bitop3 immediate out of range: ") + Twine(Raw));
    Half.HasBitOp3 = true;
    Half.BitOp3 = static_cast<uint8_t>(Raw);
    return;
  }

  if (Half.NumSrcs >= 3)
    failVopdDecode(Di, "component has more than three decoded sources");

  DecodedInst::VopdSource &Src = Half.Src[Half.NumSrcs++];
  Src.OperandIndex = McIdx;
  const MCOperand &Mop = Inst.getOperand(McIdx);
  if (Mop.isReg()) {
    classifyVopdRegSource(Di, Src, MRI, Mop.getReg());
  } else if (Mop.isImm()) {
    Src.SrcKind = DecodedInst::VopdSource::Kind::Imm;
    Src.Imm = Mop.getImm();
  } else {
    failVopdDecode(Di, Twine("component source operand is neither reg nor imm: ") +
                           Twine(McIdx));
  }

  if (Di.IsVopd3 && CompSrcIdx < Info.getCompVOPD3ModsNum()) {
    if (McIdx == 0)
      failVopdDecode(Di, "VOPD3 modifier cannot precede operand 0");
    unsigned ModIdx = McIdx - 1;
    if (ModIdx >= Inst.getNumOperands() || !Inst.getOperand(ModIdx).isImm())
      failVopdDecode(Di, Twine("VOPD3 modifier missing before source operand ") +
                             Twine(McIdx));
    int64_t Mods = Inst.getOperand(ModIdx).getImm();
    if (Mods < 0 || Mods > 0xff)
      failVopdDecode(Di, Twine("VOPD3 modifier out of range: ") + Twine(Mods));
    Src.Modifiers = static_cast<uint8_t>(Mods);
  }
}

void decodeVopdHalf(DecodedInst &Di, DecodedInst::VopdHalf &Half,
                    const AMDGPU::VOPD::ComponentInfo &Info,
                    unsigned ComponentOpcode, const OpcodeMap &OpcMap,
                    const MCRegisterInfo &MRI) {
  const MCInst &Inst = Di.Inst;
  Half.ComponentOpcode = ComponentOpcode;
  Half.CanonOp = OpcMap.lookup(ComponentOpcode);
  if (Half.CanonOp == CanonicalOp::Unknown)
    failVopdDecode(Di, Twine("unknown VOPD component opcode ") +
                           Twine(ComponentOpcode));
  Half.HasSrc2Acc = Info.hasSrc2Acc();
  Half.IsVoP3 = Info.isVOP3();

  unsigned DstIdx = Info.getIndexOfDstInMCOperands();
  if (DstIdx >= Inst.getNumOperands() || !Inst.getOperand(DstIdx).isReg())
    failVopdDecode(Di, Twine("component dst operand missing or not reg at ") +
                           Twine(DstIdx));
  Half.DstReg = Inst.getOperand(DstIdx).getReg();

  for (unsigned I = 0; I < Info.getCompParsedSrcOperandsNum(); ++I)
    decodeVopdSource(Di, Half, Info, I, MRI);

  int BitOpIdx = Info.getBitOp3OperandIdx();
  if (BitOpIdx < 0 &&
      (Half.CanonOp == CanonicalOp::V_AND_B32 || Half.CanonOp == CanonicalOp::V_OR_B32 ||
       Half.CanonOp == CanonicalOp::V_XOR_B32 ||
       Half.CanonOp == CanonicalOp::V_BITOP3_B32)) {
    // Some VOPD bitop2 forms expose the bitop3 immediate only on the paired
    // VOPD instruction, not on the canonical component pseudo (for example
    // `V_DUAL_LSHLREV_B32_e32_X_BITOP2_B32_e64_e96_gfx1250`). LLVM
    // canonicalizes the component to a simple bitwise CanonicalOp, but the paired
    // VOPD opcode name/layout still carries the authoritative BITOP2_B32
    // truth-table operand. Use the full instruction's generated named operand
    // rather than inferring anything from printed mnemonics.
    BitOpIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                          AMDGPU::OpName::bitop3);
  }
  if (!Half.HasBitOp3 && BitOpIdx >= 0) {
    if (static_cast<unsigned>(BitOpIdx) >= Inst.getNumOperands())
      failVopdDecode(Di, Twine("bitop3 operand index out of MCInst range: ") +
                             Twine(BitOpIdx));
    const MCOperand &Mop = Inst.getOperand(static_cast<unsigned>(BitOpIdx));
    if (!Mop.isImm())
      failVopdDecode(Di, "bitop3 operand is not an immediate");
    int64_t Raw = Mop.getImm();
    if (Raw < 0 || Raw > 0xff)
      failVopdDecode(Di, Twine("bitop3 immediate out of range: ") + Twine(Raw));
    Half.HasBitOp3 = true;
    Half.BitOp3 = static_cast<uint8_t>(Raw);
  }
}

void decodeVopd(DecodedInst &Di, const MCInstrInfo &MCII,
                const MCRegisterInfo &MRI, const OpcodeMap &OpcMap) {
  if (!AMDGPU::isVOPD(Di.Inst.getOpcode()))
    return;
  Di.HasVopd = true;
  Di.IsVopd3 = (Di.TsFlags & SIInstrFlags::VOPD3) != 0;

  auto [opX, opY] = AMDGPU::getVOPDComponents(Di.Inst.getOpcode());
  const MCInstrDesc &OpXDesc = MCII.get(opX);
  const MCInstrDesc &OpYDesc = MCII.get(opY);
  AMDGPU::VOPD::ComponentInfo XInfo(
      OpXDesc, AMDGPU::VOPD::ComponentKind::COMPONENT_X, Di.IsVopd3);
  AMDGPU::VOPD::ComponentInfo YInfo(OpYDesc, XInfo, Di.IsVopd3);

  decodeVopdHalf(Di, Di.Vopd[AMDGPU::VOPD::ComponentIndex::X], XInfo, opX,
                 OpcMap, MRI);
  decodeVopdHalf(Di, Di.Vopd[AMDGPU::VOPD::ComponentIndex::Y], YInfo, opY,
                 OpcMap, MRI);
}

void collectBranchTargets(const DecodedInst &Di, uint64_t Off,
                          uint64_t InstSize, uint64_t KernelStartOffset,
                          uint64_t DecodeLimit,
                          std::set<uint64_t> &BlockStarts) {
  const MCInst &Inst = Di.Inst;
  // s_add_pc_i64 carries a signed i64 PC-relative offset, not the SOPP form.
  if (Di.CanonOp == CanonicalOp::S_ADD_PC_I64) {
    std::optional<int64_t> ConstOpt = evalOperandAsConst(Inst, 0);
    if (!ConstOpt)
      report_fatal_error("transpiler: s_add_pc_i64 with non-constant source "
                         "(only immediate-literal and lit64 forms are supported)");
    int64_t Imm = *ConstOpt;
    if (InstSize > UINT64_MAX - Off)
      return;
    uint64_t Base = Off + InstSize;
    uint64_t Target = 0;
    if (Imm < 0) {
      uint64_t Back = llvm::AbsoluteValue(Imm);
      if (Back > Base)
        return;
      Target = Base - Back;
    } else {
      uint64_t Forward = static_cast<uint64_t>(Imm);
      if (Forward > UINT64_MAX - Base)
        return;
      Target = Base + Forward;
    }
    if (Target >= KernelStartOffset && Target < DecodeLimit)
      BlockStarts.insert(Target);
    return;
  }
  for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
    if (!Inst.getOperand(I).isImm())
      continue;
    int64_t Raw = Inst.getOperand(I).getImm();
    int64_t BrOff = static_cast<int64_t>(
        static_cast<int16_t>(static_cast<uint16_t>(Raw & 0xFFFF)));
    if (Off > UINT64_MAX - 4)
      continue;
    uint64_t Base = Off + 4;
    uint64_t Target = 0;
    if (BrOff < 0) {
      uint64_t Back = static_cast<uint64_t>(-BrOff) * 4;
      if (Back > Base)
        continue;
      Target = Base - Back;
    } else {
      uint64_t Forward = static_cast<uint64_t>(BrOff) * 4;
      if (Forward > UINT64_MAX - Base)
        continue;
      Target = Base + Forward;
    }
    if (Target >= KernelStartOffset && Target < DecodeLimit)
      BlockStarts.insert(Target);
  }
  if (Di.IsConditionalBranch && InstSize <= UINT64_MAX - Off) {
    uint64_t Fallthrough = Off + InstSize;
    if (Fallthrough >= KernelStartOffset && Fallthrough < DecodeLimit)
      BlockStarts.insert(Fallthrough);
  }
}

} // namespace

DecodeResult decodeKernel(const MCState &Mc,
                          const OpcodeMap &OpcMap,
                          ArrayRef<uint8_t> TextBytes,
                          uint64_t KernelOffset,
                          uint64_t KernelEndOffset,
                          std::optional<uint64_t> KernelStartOffset) {
  DecodeResult Out;
  Out.BlockStarts.insert(KernelOffset);
  uint64_t KernelStart = KernelStartOffset.value_or(KernelOffset);

  if (KernelOffset > 0)
    errs() << "transpiler: Starting disassembly at kernel offset 0x"
           << utohexstr(KernelOffset) << "\n";

  if (KernelOffset > TextBytes.size())
    report_fatal_error(
        "transpiler: kernel decode offset is outside .text contents");
  if (KernelStart > KernelOffset)
    report_fatal_error("transpiler: kernel decode start follows scan offset");
  if (KernelEndOffset != 0 && KernelEndOffset < KernelOffset)
    report_fatal_error("transpiler: kernel decode end precedes start");
  if (KernelEndOffset > TextBytes.size())
    report_fatal_error("transpiler: kernel decode end is outside .text contents");

  const uint64_t TotalSize =
      KernelEndOffset == 0 ? static_cast<uint64_t>(TextBytes.size())
                           : KernelEndOffset;
  uint64_t Off = KernelOffset;
  while (Off < TotalSize) {
    MCInst Inst;
    uint64_t InstSize = 0;
    auto Status = Mc.Disasm->getInstruction(Inst, InstSize,
                                            TextBytes.slice(Off,
                                                            TotalSize - Off),
                                            Off, nulls());
    if (Status != MCDisassembler::Success) {
      Off += 4;
      continue;
    }
    const MCInstrDesc &Desc = Mc.InstrInfo->get(Inst.getOpcode());
    DecodedInst Di;
    Di.RawMnemonic = getMnemonic(Mc, Inst);
    {
      std::string S;
      raw_string_ostream Os(S);
      Mc.Printer->printInst(&Inst, 0, "", *Mc.SubtargetInfo, Os);
      Di.FullText = StringRef(S).ltrim().str();
    }
    Di.Mnemonic = stripEncoding(StringRef(Di.RawMnemonic)).str();
    Di.Inst = Inst;
    Di.CanonOp = OpcMap.lookup(Inst.getOpcode());
    if (Di.CanonOp == CanonicalOp::V_CMP || Di.CanonOp == CanonicalOp::V_CMPX)
      Di.Vcmp = OpcMap.lookupVCmp(Inst.getOpcode());
    Di.NumDefs = Desc.getNumDefs();
    Di.IsBranch = Desc.isBranch();
    Di.IsConditionalBranch = Desc.isConditionalBranch();
    Di.Offset = Off;
    Di.Size = InstSize;
    Di.TsFlags = Desc.TSFlags;
    Di.FirstSrcIdx = Desc.getNumDefs();

    decodeScaleOffset(Di);
    decodeDppModifiers(Di);
    decodeDsSwizzleImm(Di);
    decodeVopd(Di, *Mc.InstrInfo, *Mc.RegInfo, OpcMap);
    buildSrcMap(Di, Desc);
    driftCheckTiedIn(Di, Desc);
    driftCheckSrcN(Di, Desc);
    classifyImplicitDefs(Di, Desc);

    if (Di.IsBranch)
      collectBranchTargets(Di, Off, InstSize, KernelStart, TotalSize,
                           Out.BlockStarts);

    bool IsEnd = (Di.CanonOp == CanonicalOp::S_ENDPGM);
    Out.Insts.push_back(std::move(Di));
    if (IsEnd) {
      // `s_endpgm` may appear mid-binary (early-return path); if there are
      // known block starts at later offsets, keep disassembling.
      uint64_t NextOff = Off + InstSize;
      auto It = Out.BlockStarts.upper_bound(Off);
      if (It != Out.BlockStarts.end() && *It < TotalSize) {
        Off = NextOff;
        continue;
      }
      break;
    }
    Off += InstSize;
  }

  return Out;
}

} // namespace COMGR::hotswap
