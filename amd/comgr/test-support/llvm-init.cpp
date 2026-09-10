//===- llvm-init.cpp - AMDGPU registration for standalone test binaries ---===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of `COMGR::ensureLLVMInitialized` for the test binaries that link
// the hotswap OBJECT libraries without linking amd_comgr.
//
// amd_comgr bakes in its own copy of LLVM and hides every internal symbol, so a
// binary that linked it for this one function would register AMDGPU into that
// copy's `TargetRegistry` while its own LLVM kept an empty one. Compiling this
// translation unit into the binary lands the registration on the LLVM the
// binary is linked against.
//
//===----------------------------------------------------------------------===//

#include "comgr.h"

#include "llvm/Support/TargetSelect.h"

#include <mutex>

void COMGR::ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
  });
}
