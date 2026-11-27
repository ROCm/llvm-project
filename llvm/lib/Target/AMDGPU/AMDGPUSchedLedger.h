//===- AMDGPUSchedLedger.h - Profile-guided scheduling ledger ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Scheduling DAG mutation driven by an external ledger of hints, plus a dump
/// of the scheduling DAG keyed by stable per-function instruction ids.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSCHEDLEDGER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSCHEDLEDGER_H

namespace llvm {

class MachineBasicBlock;
class MachineFunction;
class MachineInstr;
class ScheduleDAGMI;
class StringRef;

namespace AMDGPU {

/// Adds the ledger mutation to \p DAG when a ledger applies to \p MF or a DAG
/// dump was requested. Does nothing otherwise, so default codegen is unchanged.
void addSchedLedgerDAGMutation(ScheduleDAGMI *DAG, MachineFunction *MF);

/// When a DAG dump was requested, appends the register pressure of the region
/// holding \p FirstMI .. \p EndMI (exclusive, may be null) to
/// <dump dir>/<function>.pressure.jsonl: pressure before the stage, with the
/// stage's schedule, and what the scheduler kept (it may revert). The region
/// is named by its key, the smallest instruction id in it.
void reportRegionPressure(MachineFunction &MF, const MachineInstr *FirstMI,
                          const MachineInstr *EndMI, StringRef Stage,
                          unsigned VGPRBefore, unsigned VGPRAfter,
                          unsigned VGPRKept, unsigned SGPRBefore,
                          unsigned SGPRAfter, unsigned SGPRKept);

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSCHEDLEDGER_H
