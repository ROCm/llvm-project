//===- AMDGPUSim/Simulator.h - Simulator Class ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the Simulator class that wraps the core simulation logic.
/// Owns GPUSimState and provides configuration for verbose logging and
/// scoreboard mode.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMULATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMULATOR_H

#include "HWModel.h"
#include "InstrInfo.h"
#include "SimInstInfo.h"
#include "SimState.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {
namespace AMDGPUSim {

/// Configuration for the Simulator.
struct SimulatorConfig {
  /// Output stream for verbose logging.
  /// If Log == nullptr, logging is suppressed.
  raw_ostream *Log = nullptr;

  /// Enable scoreboard-based RAW hazard tracking.
  bool EnableScoreboard = false;

  /// Enable Instruction Store (IS) cache modeling.
  bool EnableISCache = false;
};

/// Simulator wraps the core simulation logic and owns the simulation state.
///
/// Usage:
/// \code
///   MachineInstrInfo InstInfo(TII, TRI);
///   HWModel Model = createHWModel(GPUTarget::GFX1250);
///   SimulatorConfig Cfg;
///   Cfg.Verbose = true;
///   Cfg.Log = &dbgs();
///   Cfg.EnableScoreboard = true;
///
///   Simulator Sim(InstInfo, Model, Cfg);
///   for (auto &MI : MBB) {
///     SimInst SI = InstInfo.createSimInst(MI);
///     InstrSimInfo Info = Sim.simulateInst(SI);
///   }
///   Sim.reset(); // For next block/function
/// \endcode
class Simulator {
  class Impl;
  std::unique_ptr<Impl> PImpl;

public:
  /// Construct a Simulator with the given instruction info, hardware model,
  /// and configuration.
  Simulator(const SimInstInfo &II, const HWModel &M, SimulatorConfig C = {});

  /// Destructor (needed for unique_ptr to incomplete type).
  ~Simulator();

  /// Simulate a single instruction.
  /// @param Inst The SimInst to simulate
  /// @param Lookahead Optional array of upcoming instructions (for MSB_SET)
  /// @return Per-instruction simulation info
  InstrSimInfo simulateInst(const SimInst &Inst,
                            ArrayRef<SimInst> Lookahead = {});

  /// Reset simulation state for a new function/kernel/block.
  void reset();

  /// Advance the simulation state by N cycles.
  /// Used by the pass for IS cache post-hoc adjustment.
  void advanceCycles(unsigned N);

  /// Get read-only access to the simulation state.
  /// Used by the pass for false-wait analysis, WMMA metrics, etc.
  const GPUSimState &getState() const;

  /// Get the configuration.
  const SimulatorConfig &getConfig() const;

  /// Get the hardware model.
  const HWModel &getModel() const;
};

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMULATOR_H
