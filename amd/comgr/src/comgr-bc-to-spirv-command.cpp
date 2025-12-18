//===- comgr-bc-to-spirv-command.cpp - BCToSPIRVCommand implementation ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CacheCommandAdaptor interface for the LLVM
/// Bitcode to SPIR-V conversion.
///
//===----------------------------------------------------------------------===//

#include "comgr-bc-to-spirv-command.h"

#ifndef COMGR_DISABLE_SPIRV
#include "comgr-diagnostic-handler.h"

#include <LLVMSPIRVLib.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>

#include <sstream>
#endif

namespace COMGR {
using namespace llvm;

Error BCToSPIRVCommand::writeExecuteOutput(StringRef CachedBuffer) {
  assert(OutputBuffer.empty());
  OutputBuffer.reserve(CachedBuffer.size());
  OutputBuffer.insert(OutputBuffer.end(), CachedBuffer.begin(),
                      CachedBuffer.end());
  return Error::success();
}

Expected<StringRef> BCToSPIRVCommand::readExecuteOutput() {
  return StringRef(OutputBuffer.data(), OutputBuffer.size());
}

amd_comgr_status_t BCToSPIRVCommand::execute(raw_ostream &LogS) {
#ifndef COMGR_DISABLE_SPIRV
  LLVMContext Context;
  Context.setDiagnosticHandler(
      std::make_unique<AMDGPUCompilerDiagnosticHandler>(LogS), true);

  // Llvm bc -> Llvm module
  auto MemBuf = MemoryBuffer::getMemBuffer(InputBuffer, "", false);
  Expected<std::unique_ptr<Module>> ModuleOrErr =
      parseBitcodeFile(MemBuf->getMemBufferRef(), Context);
  if (!ModuleOrErr)
    return AMD_COMGR_STATUS_ERROR;

  // Llvm module -> Spirv
  std::unique_ptr<Module> M = std::move(*ModuleOrErr);
  std::ostringstream OSS;
  std::string Err;
  SPIRV::TranslatorOpts Opts;
  Opts.enableAllExtensions();
  if (!writeSpirv(M.get(), Opts, OSS, Err)) {
    LogS << "Failed to translate LLVM IR to SPIR-V: " << Err << '\n';
    return AMD_COMGR_STATUS_ERROR;
  }

  std::string Result = OSS.str();
  OutputBuffer.assign(Result.begin(), Result.end());
  return AMD_COMGR_STATUS_SUCCESS;
#else
  return AMD_COMGR_STATUS_ERROR;
#endif
}

BCToSPIRVCommand::ActionClass BCToSPIRVCommand::getClass() const {
  // return an action class that is not allocated to distinguish it from any
  // clang action
  return clang::driver::Action::ActionClass::JobClassLast + 2;
}

void BCToSPIRVCommand::addOptionsIdentifier(HashAlgorithm &) const {
  // do nothing, there are no options
  return;
}

Error BCToSPIRVCommand::addInputIdentifier(HashAlgorithm &H) const {
  addString(H, InputBuffer);
  return Error::success();
}
} // namespace COMGR
