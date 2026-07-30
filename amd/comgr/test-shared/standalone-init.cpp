//===- standalone-init.cpp - LLVM init for standalone hotswap binaries ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test/CLI-only translation unit. Provides a definition of
// `COMGR::ensureLLVMInitialized` for standalone binaries that link the hotswap
// OBJECT libraries (the transpile driver and the gtest unit tests) WITHOUT
// linking `amd_comgr.so`.
//
// `amd_comgr.so` statically bakes its own copy of LLVM (Support, Target,
// AMDGPU*) and hides every internal symbol via the export map. A test
// binary that linked the .so for this helper would register AMDGPU into
// the .so's `TargetRegistry` singleton, while the binary's own statically
// linked LLVM would still see an empty registry -- same code, two LLVM
// instances, zero shared globals. Compiling this TU directly into the
// standalone binary's link line keeps the init landing on the binary's
// own LLVM globals.
//
//===----------------------------------------------------------------------===//

#include "comgr.h"

#include "llvm/Support/TargetSelect.h"

#include <mutex>

void COMGR::ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, []() {
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
#ifdef COMGR_SPIRV_BACKEND_AVAILABLE
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
#endif
  });
}
