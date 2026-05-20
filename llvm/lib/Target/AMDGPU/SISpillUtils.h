//===- SISpillUtils.h - SI spill helper functions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfoMetadata.h"

namespace llvm {

class BitVector;

using MIVector = SmallVector<MachineInstr *>;

// Update DBG_VALUE and DBG_VALUE_LIST instructions so that they correctly
// reflect performed stack to VGPR spills.
// Examples:
//  DBG_VALUE  %stack.8, 0, !"next", !DIExpression(DIOpArg(0, ptr addrspace(5)),
//                                                 DIOpDeref(i32))
//    --->
//  DBG_VALUE  %249 : vgpr_32, 0, !"next", !DIExpression(DIOpArg(0, i32),
//                                                       DIOpConstant(i8 40),
//                                                       DIOpByteOffset(i32))
//
//
//  DBG_VALUE_LIST !"next", !DIExpression(DIOpArg(0, ptr addrspace(5)),
//                                        DIOpDeref(i32),
//                                        DIOpArg(1, ptr addrspace(5)),
//                                        DIOpDeref(i32),
//                                        DIOpAdd()),
//                 %stack.9, %stack.5
//    --->
//  DBG_VALUE_LIST !"next", !DIExpression(DIOpArg(0, i32),
//                                        DIOpConstant(i8 40),
//                                        DIOpByteOffset(i32),
//                                        DIOpArg(1, ptr addrspace(5)),
//                                        DIOpDeref(i32),
//                                        DIOpAdd()),
//                 %14 : vgpr_32, %stack.5
//
void updateDbgValueInstsForSpillFIs(MIVector &Insts, const BitVector &SpillFIs);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H
