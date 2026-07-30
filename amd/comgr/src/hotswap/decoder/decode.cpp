//===- decode.cpp - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode.h"

#include "canonical-op.h"
#include "decoded-inst.h"
#include "mc-state.h"
#include "opcode-map.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, VCC, SCC, ...
#include "SIDefines.h"
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

#include <cassert>
#include <climits>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

constexpr unsigned KSoppBranchStrideBytes = 4;
constexpr int64_t KAddPcI64LiteralAlignmentBytes = 4;

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
Error buildSrcMap(DecodedInst &Di, const MCInstrDesc &Desc) {
  const MCInst &Inst = Di.Inst;
  unsigned Opc = Inst.getOpcode();
  int OldIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::old);
  int VdstInIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst_in);
  auto OpInfos = Desc.operands();
  unsigned PendingModIdx = UINT_MAX;
  for (unsigned I = Di.FirstSrcIdx; I < Inst.getNumOperands(); ++I) {
    if (I < OpInfos.size() &&
        OpInfos[I].OperandType == AMDGPU::OPERAND_INPUT_MODS) {
      PendingModIdx = I;
      continue;
    }
    if (static_cast<int>(I) == OldIdx || static_cast<int>(I) == VdstInIdx) {
      PendingModIdx = UINT_MAX;
      continue;
    }
    Di.SrcMap.push_back(I);
    Di.ModMap.push_back(PendingModIdx);
    PendingModIdx = UINT_MAX;
  }
  return Error::success();
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
Error driftCheckTiedIn(const MCState &Mc, const DecodedInst &Di,
                       const MCInstrDesc &Desc) {
  static constexpr AMDGPU::OpName KKnownTiedIn[] = {
      AMDGPU::OpName::old,     AMDGPU::OpName::vdst_in,
      AMDGPU::OpName::sdst_in, AMDGPU::OpName::vdata_in,
      AMDGPU::OpName::addr_in, AMDGPU::OpName::srcTiedDef,
      AMDGPU::OpName::src0,    AMDGPU::OpName::src1,
      AMDGPU::OpName::src2,    AMDGPU::OpName::src0X,
      AMDGPU::OpName::src0Y,   AMDGPU::OpName::src2X,
      AMDGPU::OpName::src2Y,   AMDGPU::OpName::vsrc2X,
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
            "input to keep) before proceeding for "
         << getMnemonic(Mc, Di.Inst) << " (opcode=" << Opc << "): index=" << I
         << ", tiedTo=" << Tied << ", numDefs=" << Desc.getNumDefs()
         << ", numOps=" << Inst.getNumOperands();
      return createStringError(Msg);
    }
  }
  return Error::success();
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
Error driftCheckSrcN(const MCState &Mc, DecodedInst &Di,
                     const MCInstrDesc &Desc) {
  static constexpr AMDGPU::OpName KSrcNames[] = {
      AMDGPU::OpName::src0, AMDGPU::OpName::src1, AMDGPU::OpName::src2};
  static constexpr AMDGPU::OpName KModNames[] = {
      AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
      AMDGPU::OpName::src2_modifiers};

  auto ReportErr = [&](const Twine &Prefix, int Index, int Ours,
                       int Expected) -> Error {
    std::string Msg;
    raw_string_ostream Os(Msg);
    Os << Prefix << " for " << getMnemonic(Mc, Di.Inst)
       << " (opcode=" << Di.Inst.getOpcode() << "): index=" << Index
       << ", srcMap/modMap=" << Ours << ", named=" << Expected
       << ", numSrcs=" << Di.SrcMap.size() << ", numDefs=" << Desc.getNumDefs()
       << ", numOps=" << Di.Inst.getNumOperands();
    return createStringError(Msg);
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

  // v_movreld_b32 / v_movrelsd_b32 are the "no true destination" VOP1 forms:
  // their profile sets HasDst=0 and hacks $vdst into the *inputs* at MC operand
  // 0 (see VOP_MOVREL in VOP1Instructions.td), so the positional source walk
  // records SrcMap[0] = that vdst-as-source while OpName::src0 lives at
  // index 1. That is a legitimate layout, not decoder drift -- like MADMK -- so
  // skip the strict srcN-position check at k=0 for it. Detect it structurally
  // from the operand tables (vdst present at operand 0 with zero declared defs)
  // rather than by mnemonic string. Note there is no MC-level TIED_TO here: the
  // tied vdst_in described in the ISA is added later by the custom inserter,
  // not in the MCInstrDesc we see at decode time. v_movrels_b32 has a genuine
  // def, so its walk matches OpName::src0 naturally and needs no exception. The
  // v_movrel* handler reads vdst/vsrc by named operand index (not SrcMap[0]),
  // so it does not depend on the layout skipped here; a data-dependent M0 has
  // no statically-known index and fails gracefully downstream (stubbed under
  // HSA_HOTSWAP_STUB_FAILED_KERNELS).
  int VdstIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
  bool IsMovrel = VdstIdx == 0 && Desc.getNumDefs() == 0;
  // Cross-check the structural signature against the mnemonic: every op that
  // matches (vdst-as-source, no declared defs) must be a v_movrel* form. If a
  // future opcode trips this signature without being a register-relative move,
  // we would silently skip a genuine srcN drift here -- catch that in asserts
  // builds rather than let it slip. (One-directional: v_movrels_b32 also starts
  // with the prefix but has a real def, so it is not expected to match above.)
  assert((!IsMovrel ||
          StringRef(getMnemonic(Mc, Di.Inst)).starts_with("v_movrel")) &&
         "vdst-at-0/no-defs signature matched a non-movrel opcode");

  for (unsigned K = 0; K < 3; ++K) {
    int NamedSrc = AMDGPU::getNamedOperandIdx(Opc, KSrcNames[K]);
    if (NamedSrc < 0)
      break;
    int OurSrc = (K < Di.SrcMap.size()) ? static_cast<int>(Di.SrcMap[K]) : -1;
    // Skip ONLY the genuinely-affected index (k=1) for MADMK. k=0
    // (src0) still receives the strict check, so a hypothetical
    // future drift in src0's MCInst position is still caught even
    // for MADMK opcodes.
    bool SkipThis = (IsMadmk && K == 1) || (IsMovrel && K == 0);
    if (!SkipThis && OurSrc != NamedSrc)
      return ReportErr("transpiler: srcMap disagrees with OpName::srcN table",
                       static_cast<int>(K), OurSrc, NamedSrc);
    int NamedMod = AMDGPU::getNamedOperandIdx(Opc, KModNames[K]);
    int OurMod =
        (Di.ModMap[K] == UINT_MAX) ? -1 : static_cast<int>(Di.ModMap[K]);
    int ExpectedMod = (NamedMod < 0) ? -1 : NamedMod;
    if (OurMod != ExpectedMod) {
      bool IsMai = Di.TargetSpecificFlags & SIInstrFlags::IsMAI;
      if (IsMai && NamedMod >= 0 && OurMod == -1) {
        Di.ModMap[K] = static_cast<unsigned>(NamedMod);
      } else {
        return ReportErr(
            "transpiler: modMap disagrees with OpName::srcN_modifiers table",
            static_cast<int>(K), OurMod, ExpectedMod);
      }
    }
  }
  return Error::success();
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
      Di.setDefsScc(true);
      break;
    case AMDGPU::VCC:
    case AMDGPU::VCC_LO:
    case AMDGPU::VCC_HI:
      Di.setDefsVcc(true);
      break;
    case AMDGPU::EXEC:
    case AMDGPU::EXEC_LO:
    case AMDGPU::EXEC_HI:
      Di.setDefsExec(true);
      break;
    default:
      break;
    }
  }
}

} // namespace

// Successor model over decoded block leaders. s_endpgm terminates the block
// with no successor; everything else falls through to the next linear block.
// Branch / set-pc successor edges land with control-flow support.
Expected<SmallVector<uint64_t>>
computeDecodedBlockSuccessors(const DecodedInst &LastInst,
                              std::optional<uint64_t> NextBlockOffset) {
  SmallVector<uint64_t> Result;
  if (LastInst.CanonOp == CanonicalOp::S_ENDPGM)
    return Result;
  if (NextBlockOffset)
    Result.push_back(*NextBlockOffset);
  return Result;
}

bool decodedInstEndsBlock(const DecodedInst &LastInst) {
  return LastInst.CanonOp == CanonicalOp::S_ENDPGM;
}

Expected<DecodeResult> decodeKernel(const MCState &Mc, const OpcodeMap &OpcMap,
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
    return createStringError(
        "transpiler: kernel decode offset is outside .text contents");
  if (KernelStart > KernelOffset)
    return createStringError(
        "transpiler: kernel decode start follows scan offset");
  if (KernelEndOffset != 0 && KernelEndOffset < KernelOffset)
    return createStringError("transpiler: kernel decode end precedes start");
  if (KernelEndOffset > TextBytes.size())
    return createStringError(
        "transpiler: kernel decode end is outside .text contents");

  const uint64_t TotalSize = KernelEndOffset == 0
                                 ? static_cast<uint64_t>(TextBytes.size())
                                 : KernelEndOffset;
  uint64_t Off = KernelOffset;
  while (Off < TotalSize) {
    MCInst Inst;
    uint64_t InstSize = 0;
    auto Status = Mc.Disasm->getInstruction(
        Inst, InstSize, TextBytes.slice(Off, TotalSize - Off), Off, nulls());
    if (Status != MCDisassembler::Success) {
      Off += 4;
      continue;
    }
    const MCInstrDesc &Desc = Mc.InstrInfo->get(Inst.getOpcode());
    DecodedInst Di;
    Di.Inst = Inst;
    Di.CanonOp = OpcMap.lookup(Inst.getOpcode());
    Di.NumDefs = Desc.getNumDefs();
    Di.Offset = Off;
    Di.setSizeInBytes(InstSize);
    Di.TargetSpecificFlags = Desc.TSFlags;
    Di.FirstSrcIdx = Desc.getNumDefs();

    if (Error E = buildSrcMap(Di, Desc))
      return E;
    if (Error E = driftCheckTiedIn(Mc, Di, Desc))
      return E;
    if (Error E = driftCheckSrcN(Mc, Di, Desc))
      return E;
    classifyImplicitDefs(Di, Desc);

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
