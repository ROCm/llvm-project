//===- comgr-unbundle-command.h - UnbundleCommand implementation ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_BUNDLER_COMMAND_H
#define COMGR_BUNDLER_COMMAND_H

#include <comgr-cache-command.h>

#include <llvm/Support/MemoryBuffer.h>

namespace clang {
class OffloadBundlerConfig;
} // namespace clang

namespace COMGR {
class UnbundleCommand final : public CachedCommandAdaptor {
private:
  amd_comgr_data_kind_t Kind;
  const clang::OffloadBundlerConfig &Config;

  // To avoid copies, store the output of execute, such that readExecuteOutput
  // can return a reference.
  llvm::SmallString<64> OutputBuffer;

  // Owns the contents of each output, in the order of Config.OutputFileNames.
  // Empty if the outputs were written to files instead of being unbundled in
  // memory.
  llvm::SmallVector<std::unique_ptr<llvm::MemoryBuffer>, 4> OutputBuffers;

  /// Unbundle a compressed bundle into OutputBuffer without writing any file.
  /// Returns false if the input is not a compressed bundle whose contents are
  /// in the binary bundle format, in which case the caller must use the
  /// file-based OffloadBundler API.
  llvm::Expected<bool> executeInMemory();

public:
  UnbundleCommand(amd_comgr_data_kind_t Kind,
                  const clang::OffloadBundlerConfig &Config)
      : Kind(Kind), Config(Config) {}

  /// Return true if the outputs were unbundled in memory instead of being
  /// written to the files in Config.OutputFileNames.
  bool hasInMemoryOutputs() const { return !OutputBuffers.empty(); }

  /// Return the contents of the output at \p Index, transferring ownership to
  /// the caller. Only valid if hasInMemoryOutputs() is true.
  std::unique_ptr<llvm::MemoryBuffer> takeOutputBuffer(size_t Index) {
    return std::move(OutputBuffers[Index]);
  }

  bool canCache() const override;
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;
  /// Split a cached output into the contents of each output file, in the
  /// order of Config.OutputFileNames, without writing any file.
  llvm::Error splitCachedOutput(llvm::StringRef CachedBuffer,
                                llvm::SmallVectorImpl<llvm::StringRef> &Out);
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  ~UnbundleCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
