//===- OpenMPKernelTraffic.h - Per-kernel memory traffic --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OPENMPKERNELTRAFFIC_H
#define LLVM_TRANSFORMS_IPO_OPENMPKERNELTRAFFIC_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;

/// Record a per-iteration global memory traffic estimate in a
/// <kernel>_kernel_traffic global for every OpenMP offload kernel in \p M.
///
/// This must run after inlining and after address-space inference; see the
/// comment at the top of OpenMPKernelTraffic.cpp for why.
class OpenMPKernelTrafficPass
    : public OptionalPassInfoMixin<OpenMPKernelTrafficPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Mark the loop of every static OpenMP worksharing construct in \p M, so that
/// OpenMPKernelTrafficPass can tell a kernel's work loop from the loops the
/// device runtime brings along once it is inlined.
///
/// This must run while the calls into the runtime that bracket such a loop are
/// still there, i.e. before the device runtime is inlined.
class OpenMPWorkLoopMarkerPass
    : public OptionalPassInfoMixin<OpenMPWorkLoopMarkerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_OPENMPKERNELTRAFFIC_H
