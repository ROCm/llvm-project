//===- AMDGPUSim/AMDGPUSim.h - AMDGPU Static Simulator Library ---*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Main public header for the AMDGPU Static Simulator library.
///
/// This library provides a standalone GPU performance simulator that can be
/// used by:
/// - MachineFunction passes (via MachineInstrInfo adapter)
/// - MC layer passes (via MCInstInfo adapter)
/// - External tools (implementing custom SimInstInfo)
///
/// Usage:
/// \code
///   using namespace llvm::AMDGPUSim;
///
///   HWModel Model = createHWModel(GPUTarget::GFX1250);
///   MachineInstrInfo InstInfo(TII, TRI);
///   SimulatorConfig Cfg;
///   Cfg.Verbose = true;
///   Cfg.Log = &dbgs();
///
///   Simulator Sim(InstInfo, Model, Cfg);
///   for (auto &MI : MBB) {
///     SimInst SI = InstInfo.createSimInst(MI);
///     InstrSimInfo Info = Sim.simulateInst(SI);
///   }
///   Sim.reset();
/// \endcode
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_AMDGPUSIM_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_AMDGPUSIM_H

#include "HWModel.h"
#include "InstrInfo.h"
#include "SimInst.h"
#include "SimInstInfo.h"
#include "SimState.h"
#include "Simulator.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Factory Functions
//===----------------------------------------------------------------------===//

inline HWModel createHWModel(GPUTarget Target) {
  switch (Target) {
  case GPUTarget::GFX1250:
    return HWModel::gfx1250();
  }
  llvm_unreachable("Unknown GPUTarget");
}

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_AMDGPUSIM_H
