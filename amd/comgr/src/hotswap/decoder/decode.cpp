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

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <climits>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "hotswap-decode"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// AMDGPU instructions occupy whole 4-byte units; step past a word the
// disassembler could not decode.
constexpr uint64_t KInstAlignBytes = 4;

// getNamedOperandIdx returns -1 when the opcode has no such operand; index 0
// is valid. Normalise to an optional operand index so callers compare against
// unsigned MCInst operand indices without re-testing the sentinel or casting.
std::optional<unsigned> namedOperandIdx(unsigned Opc, AMDGPU::OpName Name) {
  int Idx = AMDGPU::getNamedOperandIdx(Opc, Name);
  return Idx >= 0 ? std::optional<unsigned>(Idx) : std::nullopt;
}

// Build the logical-source view of an MCInst: SrcMap lists the operand indices
// that are real sources and ModMap the source-modifier operand paired with each
// (UINT_MAX when none). A VOP3 source modifier (OPERAND_INPUT_MODS) precedes
// its source. The DPP/SDWA "old"/"vdst_in" operand is tied to the def but never
// read in our all-lanes-active model, so it is skipped; other tied inputs (MAC
// accumulators, atomic read-modify) are real sources and kept.
void buildSrcMap(DecodedInst &Di, const MCInstrDesc &Desc) {
  const MCInst &Inst = Di.Inst;
  unsigned Opc = Inst.getOpcode();
  std::optional<unsigned> OldIdx = namedOperandIdx(Opc, AMDGPU::OpName::old);
  std::optional<unsigned> VdstInIdx =
      namedOperandIdx(Opc, AMDGPU::OpName::vdst_in);
  auto OpInfos = Desc.operands();
  unsigned NumOps = Inst.getNumOperands();
  unsigned PendingModIdx = UINT_MAX;
  for (unsigned I = Di.FirstSrcIdx; I < NumOps; ++I) {
    if (I < OpInfos.size() &&
        OpInfos[I].OperandType == AMDGPU::OPERAND_INPUT_MODS) {
      PendingModIdx = I;
      continue;
    }
    if (OldIdx == I || VdstInIdx == I) {
      PendingModIdx = UINT_MAX;
      continue;
    }
    Di.SrcMap.push_back(I);
    Di.ModMap.push_back(PendingModIdx);
    PendingModIdx = UINT_MAX;
  }
}

// Assert that every operand tied to a def carries an OpName we have classified.
// A tied-to-def operand is either a DPP/SDWA inactive-lane fallback (skipped by
// buildSrcMap) or a real read-modify input (kept); the two are distinguished by
// OpName, so an unrecognised one means LLVM added a tied input this code does
// not account for.
void driftCheckTiedIn(const DecodedInst &Di, const MCInstrDesc &Desc) {
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
  unsigned Opc = Di.Inst.getOpcode();
  unsigned NumOps = Di.Inst.getNumOperands();
  for (unsigned I = 0; I < NumOps; ++I) {
    int Tied = Desc.getOperandConstraint(I, MCOI::TIED_TO);
    // Only defs matter here: use-to-use ties exist but are not fallbacks or
    // accumulators.
    if (Tied < 0 || static_cast<unsigned>(Tied) >= Desc.getNumDefs())
      continue;
    [[maybe_unused]] bool Known =
        llvm::any_of(KKnownTiedIn, [&](AMDGPU::OpName N) {
          return namedOperandIdx(Opc, N) == I;
        });
    assert(Known && "tied-to-def operand has an unclassified OpName");
  }
}

// Assert that the leading sources and their modifiers agree with LLVM's
// named-operand table, catching operand-layout changes for the many opcodes
// that use srcN naming (VALU, VOPC, SOP1/SOP2, ...). Scaled MFMA appends its
// source modifiers after the sources instead of interleaving them, so the
// positional walk in buildSrcMap misses them; those are repaired here from the
// named-operand table.
void driftCheckSrcN([[maybe_unused]] const MCState &Mc, DecodedInst &Di,
                    const MCInstrDesc &Desc) {
  static constexpr AMDGPU::OpName KSrcNames[] = {
      AMDGPU::OpName::src0, AMDGPU::OpName::src1, AMDGPU::OpName::src2};
  static constexpr AMDGPU::OpName KModNames[] = {
      AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
      AMDGPU::OpName::src2_modifiers};

  unsigned Opc = Di.Inst.getOpcode();

  // MADMK/FMAMK place the literal between src0 and src1, so buildSrcMap's
  // positional SrcMap[1] is the literal, not src1. Handlers index by MC operand
  // order, so skip the strict src1 position check for this known layout.
  std::optional<unsigned> ImmIdx = namedOperandIdx(Opc, AMDGPU::OpName::imm);
  std::optional<unsigned> Src0Idx = namedOperandIdx(Opc, AMDGPU::OpName::src0);
  std::optional<unsigned> Src1Idx = namedOperandIdx(Opc, AMDGPU::OpName::src1);
  bool IsMadmk =
      ImmIdx && Src0Idx && Src1Idx && *Src0Idx < *ImmIdx && *ImmIdx < *Src1Idx;

  // v_movrel{d,sd}_b32 have no real def and place $vdst at operand 0 as an
  // input, so SrcMap[0] is that vdst-as-source while src0 is at operand 1. Skip
  // the strict src0 position check for this layout; the handler reads by named
  // operand index, not SrcMap[0].
  bool IsMovrel = namedOperandIdx(Opc, AMDGPU::OpName::vdst) == 0u &&
                  Desc.getNumDefs() == 0;
  assert((!IsMovrel ||
          StringRef(getMnemonic(Mc, Di.Inst)).starts_with("v_movrel")) &&
         "vdst-at-0/no-defs signature matched a non-movrel opcode");

  for (unsigned K = 0; K < 3; ++K) {
    std::optional<unsigned> NamedSrc = namedOperandIdx(Opc, KSrcNames[K]);
    if (!NamedSrc)
      break;
    std::optional<unsigned> OurSrc = K < Di.SrcMap.size()
                                         ? std::optional<unsigned>(Di.SrcMap[K])
                                         : std::nullopt;
    bool SkipThis = (IsMadmk && K == 1) || (IsMovrel && K == 0);
    assert((SkipThis || OurSrc == NamedSrc) &&
           "srcMap disagrees with OpName::srcN table");

    std::optional<unsigned> NamedMod = namedOperandIdx(Opc, KModNames[K]);
    std::optional<unsigned> OurMod =
        Di.ModMap[K] == UINT_MAX ? std::nullopt
                                 : std::optional<unsigned>(Di.ModMap[K]);
    if (OurMod != NamedMod) {
      bool IsMai = Di.TargetSpecificFlags & SIInstrFlags::IsMAI;
      assert(IsMai && NamedMod && !OurMod &&
             "modMap disagrees with OpName::srcN_modifiers table");
      if (IsMai && NamedMod)
        Di.ModMap[K] = *NamedMod;
    }
  }
}

// Record implicit defs of SCC / VCC / EXEC, normalising subtarget-suffixed
// register ids to their canonical pseudo first.
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

// s_endpgm ends the block with no successor; any other block falls through to
// NextBlockOffset when a block follows.
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

DecodeResult decodeKernel(const MCState &Mc, const OpcodeMap &OpcMap,
                          ArrayRef<uint8_t> TextBytes, uint64_t KernelOffset,
                          std::optional<uint64_t> KernelEndOffset,
                          std::optional<uint64_t> KernelStartOffset) {
  DecodeResult Out;
  Out.BlockStarts.insert(KernelOffset);
  [[maybe_unused]] uint64_t KernelStart =
      KernelStartOffset.value_or(KernelOffset);

  LLVM_DEBUG(if (KernelOffset > 0) dbgs()
             << "hotswap: starting disassembly at kernel offset 0x"
             << utohexstr(KernelOffset) << "\n");

  assert(KernelOffset <= TextBytes.size() &&
         "kernel decode offset is outside .text");
  assert(KernelStart <= KernelOffset && "kernel decode start follows scan");
  assert((!KernelEndOffset || *KernelEndOffset >= KernelOffset) &&
         "kernel decode end precedes start");
  assert((!KernelEndOffset || *KernelEndOffset <= TextBytes.size()) &&
         "kernel decode end is outside .text");

  const uint64_t TotalSize = KernelEndOffset.value_or(TextBytes.size());
  uint64_t Off = KernelOffset;
  while (Off < TotalSize) {
    MCInst Inst;
    uint64_t InstSize = 0;
    auto Status = Mc.Disasm->getInstruction(
        Inst, InstSize, TextBytes.slice(Off, TotalSize - Off), Off, nulls());
    if (Status != MCDisassembler::Success) {
      Off += KInstAlignBytes;
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

    buildSrcMap(Di, Desc);
    driftCheckTiedIn(Di, Desc);
    driftCheckSrcN(Mc, Di, Desc);
    classifyImplicitDefs(Di, Desc);

    bool IsEnd = decodedInstEndsBlock(Di);
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
