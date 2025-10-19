//===-- AMDGPUHotBlockRegisterRenaming.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Reduces value density in hot basic blocks by remapping local values
/// from overused physical registers to free physical registers.
///
/// This gives the Post-RA scheduler more flexibility to reorder instructions
/// by reducing false dependencies created by register reuse.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUHOTBLOCKREGISTERRENAMING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUHOTBLOCKREGISTERRENAMING_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class AMDGPUHotBlockRegisterRenamingPass
    : public PassInfoMixin<AMDGPUHotBlockRegisterRenamingPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUHOTBLOCKREGISTERRENAMING_H
