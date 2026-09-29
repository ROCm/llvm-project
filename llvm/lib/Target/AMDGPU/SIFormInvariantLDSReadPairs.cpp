//===- SIFormInvariantLDSReadPairs.cpp - Pair invariant LDS reads ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Combine two invariant ds_read_b128 from the same address register, whose
/// results only feed one 256-bit REG_SEQUENCE, into a single
/// DS_READ_B256_INVARIANT_PSEUDO.
///
/// The register allocator can rematerialize a value defined by one
/// instruction, but not one assembled from several. Wide MFMA operands (e.g. an
/// 8-dword fragment kept live across a loop) are typically built from two
/// 128-bit LDS reads, so under register pressure they get spilled to scratch
/// and reloaded on the loop's critical path. With the pseudo, the allocator can
/// instead re-read the fragment from LDS. The pseudo is expanded back into the
/// two ds_read_b128 after register allocation.
///
/// Only loads marked invariant (!invariant.load) are combined: the transform
/// relies on the LDS contents not changing between the original read and any
/// rematerialized read.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "si-form-invariant-lds-read-pairs"

STATISTIC(NumFormed, "Number of invariant ds_read_b128 pairs combined");

static cl::opt<bool> EnableFormInvariantLDSReadPairs(
    "amdgpu-form-invariant-lds-read-pairs",
    cl::desc("Combine pairs of invariant ds_read_b128 that build one 256-bit "
             "value into a rematerializable pseudo"),
    cl::init(true), cl::Hidden);

namespace {

class SIFormInvariantLDSReadPairs {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  // Source of one dword of the REG_SEQUENCE result.
  struct DWordSrc {
    MachineInstr *Load = nullptr;
    unsigned DWord = 0;
  };

  bool isCandidateLoad(const MachineInstr &MI) const;
  unsigned dwordOffset(unsigned SubIdx) const {
    return SubIdx ? TRI->getSubRegIdxOffset(SubIdx) / 32 : 0;
  }
  unsigned dwordCount(Register Reg, unsigned SubIdx) const {
    return SubIdx ? TRI->getSubRegIdxSize(SubIdx) / 32
                  : TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) / 32;
  }
  bool tryCombine(MachineInstr &RegSeq);

public:
  bool run(MachineFunction &MF);
};

class SIFormInvariantLDSReadPairsLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIFormInvariantLDSReadPairsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return SIFormInvariantLDSReadPairs().run(MF);
  }

  StringRef getPassName() const override {
    return "SI Form Invariant LDS Read Pairs";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(SIFormInvariantLDSReadPairsLegacy, DEBUG_TYPE,
                "SI Form Invariant LDS Read Pairs", false, false)

char SIFormInvariantLDSReadPairsLegacy::ID = 0;

char &llvm::SIFormInvariantLDSReadPairsLegacyID =
    SIFormInvariantLDSReadPairsLegacy::ID;

FunctionPass *llvm::createSIFormInvariantLDSReadPairsLegacyPass() {
  return new SIFormInvariantLDSReadPairsLegacy();
}

bool SIFormInvariantLDSReadPairs::isCandidateLoad(
    const MachineInstr &MI) const {
  if (MI.getOpcode() != AMDGPU::DS_READ_B128_gfx9)
    return false;
  const MachineOperand *GDS = TII->getNamedOperand(MI, AMDGPU::OpName::gds);
  if (GDS && GDS->getImm())
    return false;
  const MachineOperand *Addr = TII->getNamedOperand(MI, AMDGPU::OpName::addr);
  if (!Addr || !Addr->getReg().isVirtual() || Addr->getSubReg())
    return false;
  if (!MI.getOperand(0).getReg().isVirtual() || MI.hasOrderedMemoryRef() ||
      !MI.hasOneMemOperand())
    return false;
  const MachineMemOperand *MMO = *MI.memoperands_begin();
  return MMO->isLoad() && !MMO->isStore() && MMO->isInvariant() &&
         !MMO->isVolatile() && MMO->getAddrSpace() == AMDGPUAS::LOCAL_ADDRESS;
}

bool SIFormInvariantLDSReadPairs::tryCombine(MachineInstr &RegSeq) {
  Register Dst = RegSeq.getOperand(0).getReg();
  if (!Dst.isVirtual() || RegSeq.getOperand(0).getSubReg() ||
      TRI->getRegSizeInBits(*MRI->getRegClass(Dst)) != 256)
    return false;

  DWordSrc DWords[8];
  SmallVector<MachineInstr *, 8> Copies;

  for (unsigned I = 1, E = RegSeq.getNumOperands(); I + 1 < E; I += 2) {
    const MachineOperand &Src = RegSeq.getOperand(I);
    unsigned DstSubIdx = RegSeq.getOperand(I + 1).getImm();
    if (!Src.isReg() || !Src.getReg().isVirtual())
      return false;

    Register SrcReg = Src.getReg();
    unsigned SrcSubIdx = Src.getSubReg();
    MachineInstr *Def = MRI->getUniqueVRegDef(SrcReg);
    if (!Def)
      return false;

    // Look through a single-use full COPY of a load subregister.
    if (Def->isCopy() && !SrcSubIdx) {
      const MachineOperand &CopySrc = Def->getOperand(1);
      if (!CopySrc.getReg().isVirtual() || !MRI->hasOneNonDBGUse(SrcReg) ||
          Def->getOperand(0).getSubReg())
        return false;
      Copies.push_back(Def);
      SrcReg = CopySrc.getReg();
      SrcSubIdx = CopySrc.getSubReg();
      Def = MRI->getUniqueVRegDef(SrcReg);
      if (!Def)
        return false;
    }
    if (!isCandidateLoad(*Def))
      return false;

    unsigned DstStart = dwordOffset(DstSubIdx);
    unsigned Count = TRI->getSubRegIdxSize(DstSubIdx) / 32;
    if (Count != dwordCount(SrcReg, SrcSubIdx) || DstStart + Count > 8)
      return false;
    unsigned SrcStart = dwordOffset(SrcSubIdx);
    for (unsigned D = 0; D < Count; ++D) {
      if (DWords[DstStart + D].Load)
        return false;
      DWords[DstStart + D] = {Def, SrcStart + D};
    }
  }

  // dwords 0-3 must be load A's dwords 0-3, dwords 4-7 load B's dwords 0-3.
  MachineInstr *Lo = DWords[0].Load;
  MachineInstr *Hi = DWords[4].Load;
  if (!Lo || !Hi || Lo == Hi || Lo->getParent() != Hi->getParent())
    return false;
  for (unsigned D = 0; D < 8; ++D) {
    MachineInstr *Expected = D < 4 ? Lo : Hi;
    if (DWords[D].Load != Expected || DWords[D].DWord != D % 4)
      return false;
  }

  Register Addr = TII->getNamedOperand(*Lo, AMDGPU::OpName::addr)->getReg();
  if (TII->getNamedOperand(*Hi, AMDGPU::OpName::addr)->getReg() != Addr)
    return false;

  // Every use of the loads (and of the looked-through copies) must be part of
  // this pattern, so the originals can be deleted.
  SmallPtrSet<const MachineInstr *, 12> Pattern(Copies.begin(), Copies.end());
  Pattern.insert(&RegSeq);
  for (MachineInstr *Load : {Lo, Hi})
    for (const MachineInstr &Use :
         MRI->use_instructions(Load->getOperand(0).getReg()))
      if (!Pattern.contains(&Use))
        return false;

  const MCInstrDesc &Desc = TII->get(AMDGPU::DS_READ_B256_INVARIANT_PSEUDO);
  if (!MRI->constrainRegClass(Dst, TII->getRegClass(Desc, 0)))
    return false;

  // Emit the pseudo after the later of the two loads.
  MachineBasicBlock &MBB = *Lo->getParent();
  MachineInstr *Later = Hi;
  for (MachineInstr &MI : MBB) {
    if (&MI == Lo) {
      Later = Hi;
      break;
    }
    if (&MI == Hi) {
      Later = Lo;
      break;
    }
  }

  int64_t OffLo = TII->getNamedOperand(*Lo, AMDGPU::OpName::offset)->getImm();
  int64_t OffHi = TII->getNamedOperand(*Hi, AMDGPU::OpName::offset)->getImm();
  MachineInstr *Pseudo = BuildMI(MBB, std::next(Later->getIterator()),
                                 Later->getDebugLoc(), Desc, Dst)
                             .addReg(Addr)
                             .addImm(OffLo)
                             .addImm(OffHi)
                             .addMemOperand(*Lo->memoperands_begin())
                             .addMemOperand(*Hi->memoperands_begin());
  (void)Pseudo;
  LLVM_DEBUG(dbgs() << "Formed: " << *Pseudo);

  MRI->clearKillFlags(Addr);
  RegSeq.eraseFromParent();
  for (MachineInstr *Copy : Copies)
    Copy->eraseFromParent();
  Lo->eraseFromParent();
  Hi->eraseFromParent();
  ++NumFormed;
  return true;
}

bool SIFormInvariantLDSReadPairs::run(MachineFunction &MF) {
  if (!EnableFormInvariantLDSReadPairs)
    return false;
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (ST.ldsRequiresM0Init())
    return false;
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  if (!MRI->isSSA())
    return false;

  SmallVector<MachineInstr *, 16> Candidates;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isRegSequence())
        Candidates.push_back(&MI);

  bool Changed = false;
  for (MachineInstr *RegSeq : Candidates)
    Changed |= tryCombine(*RegSeq);
  return Changed;
}
