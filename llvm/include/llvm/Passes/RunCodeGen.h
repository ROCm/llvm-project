//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_RUN_CODEGEN_H
#define LLVM_PASSES_RUN_CODEGEN_H

#include "llvm/IR/Module.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class ModuleSummaryIndex;

namespace legacy {
class PassManager;
} // namespace legacy

/// Options for the target-independent code generation pipeline.
struct CodeGenPipelineConfig {
  /// Combined summary index to make available to the pipeline, or null if the
  /// pipeline should not have access to one.
  const ModuleSummaryIndex *SummaryIndex = nullptr;

  bool DisableVerify = true;
  bool DisableSimplifyLibCalls = false;

  /// Print the passes the New PM runs, as -debug-pass-manager does.
  bool DebugPassManager = false;

  /// Verify the IR after every pass.
  bool VerifyEach = false;

  /// Legacy-only extension point, invoked just before the pass manager is
  /// populated. Ignored by the New PM pipeline.
  std::function<void(legacy::PassManager &)> PreCodeGenPassesHook;

  /// File system to use, or null to fall back to the real file system. The
  /// fallback is resolved lazily because some clients cannot tolerate
  /// vfs::getRealFileSystem() being called at all.
  IntrusiveRefCntPtr<vfs::FileSystem> VFS = nullptr;
};

LLVM_ABI Error runCodeGenPipeline(TargetMachine &TM, Module &M,
                                  raw_pwrite_stream &OS,
                                  raw_pwrite_stream *DwoOS,
                                  CodeGenFileType CGFT,
                                  const CodeGenPipelineConfig &Config = {});

} // namespace llvm

#endif // LLVM_PASSES_RUN_CODEGEN_H
