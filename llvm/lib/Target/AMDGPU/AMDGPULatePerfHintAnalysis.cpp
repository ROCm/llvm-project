//===- AMDGPULatePerfHintAnalysis.cpp - analysis of functions memory traffic --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------------===//
//
/// \file
/// \brief Analyzes if a machine function is potentially memory bound and if a
/// kernel may benefit from perf hints such as MFMA Underfeed.
///
//===--------------------------------------------------------------------------===//

#include "AMDGPUMachineFunctionInfo.h"
#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-late-perf-hint"

static cl::opt<unsigned>
LatePerfHintMinMFMA("amdgpu-late-perf-hint-min-mfma",
                    cl::desc("Minimum MFMA count before late perf hint triggers"),
                    cl::init(8), cl::Hidden);

static cl::opt<unsigned>
LatePerfHintRatioNum("amdgpu-late-perf-hint-ratio-num",
                     cl::desc("Numerator for MFMA/DS_READ_B128 ratio threshold"),
                     cl::init(2), cl::Hidden);

static cl::opt<unsigned>
LatePerfHintRatioDen("amdgpu-late-perf-hint-ratio-den",
                     cl::desc("Denominator for MFMA/DS_READ_B128 ratio threshold"),
                     cl::init(1), cl::Hidden);

namespace {

class AMDGPULatePerfHintAnalysis : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULatePerfHintAnalysis() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU Late Perf Hint Analysis";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    const auto &ST = MF.getSubtarget<GCNSubtarget>();
    const SIInstrInfo *TII = ST.getInstrInfo();

    unsigned MFMACount = 0;
    unsigned DSReadB128Count = 0;

    LLVM_DEBUG(dbgs() << "Counting instructions...\n";);
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (TII->isMFMA(MI))
          ++MFMACount;
        if (MI.getOpcode() == AMDGPU::DS_READ_B128_gfx9)
          ++DSReadB128Count;
      }
    }

    // If there is enough MFMAs, we wager that this kernel will
    // spend more time on MFMA computations than other work.
    // If the ratio of MFMA to DS_READs is low, we can guess that
    // MFMAs will likely be underfed, waiting on DS_READs.
    const bool EnoughMFMA = MFMACount >= LatePerfHintMinMFMA;

    // Compare:
    //   MFMACount / max(1, DSReadB128Count) <= RatioNum / RatioDen
    // without using FP.
    const unsigned SafeDSReadCount = std::max(1u, DSReadB128Count);
    const bool LowRatio =
        MFMACount * LatePerfHintRatioDen <=
        SafeDSReadCount * LatePerfHintRatioNum;

    const bool Trigger = EnoughMFMA && LowRatio;

    LLVM_DEBUG(dbgs() << "LatePerfHint: " << MF.getName()
                      << " MFMA=" << MFMACount
                      << " DS_READ_B128=" << DSReadB128Count
                      << " trigger=" << Trigger << '\n');

    // Todo: Use different flag?
    auto &MFI = *MF.getInfo<AMDGPUMachineFunctionInfo>();
    MFI.MemoryBound = Trigger;

    return false; // analysis / hint only
  }
};

} // end anonymous namespace

char AMDGPULatePerfHintAnalysis::ID = 0;

INITIALIZE_PASS(AMDGPULatePerfHintAnalysis, DEBUG_TYPE,
                "AMDGPU Late Perf Hint Analysis", false, true)

char &llvm::AMDGPULatePerfHintAnalysisID = AMDGPULatePerfHintAnalysis::ID;

FunctionPass *createAMDGPULatePerfHintAnalysisPass() {
  return new AMDGPULatePerfHintAnalysis();
}
