//===- GenericProfiler.cpp - GenericProfiler implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "GenericProfiler.h"
#include "PluginInterface.h"
#include "Shared/Debug.h"

#include <cstdint>
#include <memory>

__attribute__((weak)) std::unique_ptr<llvm::offload::plugin::GenericProfilerTy>
getProfilerToAttach() {
  return std::make_unique<llvm::offload::plugin::GenericProfilerTy>();
}

namespace llvm {
namespace offload {
namespace plugin {

uint64_t GenericProfilerTy::getDeviceTimeStamp(GenericDeviceTy *D) {
  if (D)
    return D->getDeviceTimeStamp();
  return 0;
}
} // namespace plugin
} // namespace offload
} // namespace llvm
