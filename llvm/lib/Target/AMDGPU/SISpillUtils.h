//===- SISpillUtils.h - SI spill helper functions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H

namespace llvm {

class MachineFunction;
class MachineInstr;
class BitVector;

enum class SISpillKind {
  SGPRToVGPR,
  VGPRToAGPR
};

void updateDbgValueForSISpill(MachineFunction &MF, MachineInstr &MI,
                              const BitVector &SpillFIs, SISpillKind Kind);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H
