//===- standalone-init.cpp - LLVM init for standalone hotswap binaries ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test/CLI-only translation unit. Provides definitions of
// `COMGR::ensureLLVMInitialized` and `COMGR::parseTargetIdentifier` for
// standalone binaries that link the hotswap-transpiler OBJECT library
// (`raise_cli` and the gtest unit tests) WITHOUT linking `amd_comgr.so`.
//
// `amd_comgr.so` statically bakes its own copy of LLVM (Support, Target,
// AMDGPU*) and hides every internal symbol via the export map. A test
// binary that linked the .so for these helpers would register AMDGPU into
// the .so's `TargetRegistry` singleton, while the binary's own statically
// linked LLVM would still see an empty registry — same code, two LLVM
// instances, zero shared globals. Compiling this TU directly into the
// standalone binary's link line keeps the init landing on the binary's
// own LLVM globals.
//
// Body intentionally identical to `comgr.cpp`'s definitions; we are not
// extracting the production code into a shared TU because that adds a
// new file to `amd_comgr.so`'s source list (review-fragile) for a
// concern that only matters to test/CLI binaries. The duplication is
// localized to test infrastructure under `test-shared/`.
//
//===----------------------------------------------------------------------===//

#include "comgr.h"
#include "comgr-metadata.h"

#include "llvm/Support/TargetSelect.h"

#include <mutex>

using namespace llvm;

amd_comgr_status_t COMGR::parseTargetIdentifier(StringRef IdentStr,
                                                TargetIdentifier &Ident) {
  SmallVector<StringRef, 5> IsaNameComponents;
  IdentStr.split(IsaNameComponents, '-', 4);
  if (IsaNameComponents.size() != 5) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Ident.Arch = IsaNameComponents[0];
  Ident.Vendor = IsaNameComponents[1];
  Ident.OS = IsaNameComponents[2];
  Ident.Environ = IsaNameComponents[3];

  Ident.Features.clear();
  IsaNameComponents[4].split(Ident.Features, ':');

  Ident.Processor = Ident.Features[0];
  Ident.Features.erase(Ident.Features.begin());

  // TODO: Add a LIT test for this
  if (IdentStr == "spirv64-amd-amdhsa--amdgcnspirv" ||
      IdentStr == "spirv64-amd-amdhsa-unknown-amdgcnspirv") {
    if (!Ident.Features.empty())
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  size_t IsaIndex;
  amd_comgr_status_t Status = metadata::getIsaIndex(IdentStr, IsaIndex);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    return Status;
  }

  for (auto Feature : Ident.Features) {
    if (!metadata::isSupportedFeature(IsaIndex, Feature)) {
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

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
