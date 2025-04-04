/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#ifndef COMGR_CLANG_COMMAND_H
#define COMGR_CLANG_COMMAND_H

#include "comgr-cache-command.h"

#include <llvm/Support/VirtualFileSystem.h>

namespace clang {
class DiagnosticOptions;
namespace driver {
class Command;
} // namespace driver
} // namespace clang

namespace COMGR {
class ClangCommand final : public CachedCommandAdaptor {
public:
  using ExecuteFnTy = std::function<amd_comgr_status_t(
      clang::driver::Command &, llvm::raw_ostream &, clang::DiagnosticOptions &,
      llvm::vfs::FileSystem &)>;

private:
  clang::driver::Command &Command;
  clang::DiagnosticOptions &DiagOpts;
  llvm::vfs::FileSystem &VFS;
  ExecuteFnTy ExecuteImpl;

  // To avoid copies, store the output of execute, such that readExecuteOutput
  // can return a reference.
  std::unique_ptr<llvm::MemoryBuffer> Output;

public:
  ClangCommand(clang::driver::Command &Command,
               clang::DiagnosticOptions &DiagOpts, llvm::vfs::FileSystem &VFS,
               ExecuteFnTy &&ExecuteImpl);

  bool canCache() const override;
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  ~ClangCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
