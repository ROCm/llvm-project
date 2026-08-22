//===- OMPDeviceConstants.h - OpenMP device related constants ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines constans that will be used by both host and device
/// compilation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMPDEVICECONSTANTS_H
#define LLVM_FRONTEND_OPENMP_OMPDEVICECONSTANTS_H

namespace llvm {
namespace omp {

enum OMPTgtExecModeFlags : unsigned char {
  OMP_TGT_EXEC_MODE_BARE = 0,
  OMP_TGT_EXEC_MODE_GENERIC = 1 << 0,
  OMP_TGT_EXEC_MODE_SPMD = 1 << 1,
  OMP_TGT_EXEC_MODE_GENERIC_SPMD =
      OMP_TGT_EXEC_MODE_GENERIC | OMP_TGT_EXEC_MODE_SPMD,
  OMP_TGT_EXEC_MODE_SPMD_NO_LOOP = 1 << 2 | OMP_TGT_EXEC_MODE_SPMD,
  OMP_TGT_EXEC_MODE_SPMD_BIG_JUMP_LOOP = OMP_TGT_EXEC_MODE_SPMD_NO_LOOP | 1,
  OMP_TGT_EXEC_MODE_XTEAM_RED = 1 << 3
};

/// Whether \p Mode runs every thread in the kernel body, so that no generic
/// worker/dispatcher state machine is needed. Modes are not a bitmask that can
/// be tested for the SPMD bit, so they are enumerated explicitly.
constexpr bool isSPMDExecMode(OMPTgtExecModeFlags Mode) {
  switch (Mode) {
  case OMP_TGT_EXEC_MODE_SPMD:
  case OMP_TGT_EXEC_MODE_SPMD_NO_LOOP:
  case OMP_TGT_EXEC_MODE_SPMD_BIG_JUMP_LOOP:
  case OMP_TGT_EXEC_MODE_XTEAM_RED:
    return true;
  default:
    return false;
  }
}

} // end namespace omp
} // end namespace llvm

#endif // LLVM_FRONTEND_OPENMP_OMPDEVICECONSTANTS_H
