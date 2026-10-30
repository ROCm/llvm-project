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

namespace llvm::vfs {
class FileSystem;
}

namespace COMGR {
/// Adapt Clang's unbundler to COMGR's cache and data objects.
/// Bitcode/object bundles produce memory buffers; archives use output files.
class UnbundleCommand final : public CachedCommandAdaptor {
private:
  amd_comgr_data_kind_t Kind;
  const clang::OffloadBundlerConfig &Config;
  llvm::vfs::FileSystem &FS;

  // Owns the serialized cache entry returned by readExecuteOutput().
  llvm::SmallString<64> OutputBuffer;

  // Owns the contents of each output, in the order of Config.OutputFileNames.
  // Empty if the outputs were written to files instead of being unbundled in
  // memory.
  llvm::SmallVector<std::unique_ptr<llvm::MemoryBuffer>, 4> OutputBuffers;

  /// Collect Clang's VFS outputs in Config.OutputFileNames order. Duplicate
  /// names share storage; misaligned members are copied for COMGR consumers.
  /// OutputBuffers must be empty on entry.
  llvm::Error executeInMemory();

  /// Write OutputBuffers to Config.OutputFileNames if AMD_COMGR_SAVE_TEMPS is
  /// enabled. Retain the buffers and propagate any file-writing error.
  llvm::Error saveOutputTemps() const;

  /// Append views of cached members to \p Out in Config.OutputFileNames order.
  /// Accept aligned CUB1 and legacy size-prefixed entries; reject truncated
  /// framing or unused trailing bytes. CachedBuffer must outlive the views.
  /// On error, Out may contain partial results and must not be used.
  llvm::Error splitCachedOutput(llvm::StringRef CachedBuffer,
                                llvm::SmallVectorImpl<llvm::StringRef> &Out);

public:
  /// Borrow the configuration and input filesystem for one unbundling job.
  /// Config, FS, and any borrowed input storage must outlive this command.
  UnbundleCommand(amd_comgr_data_kind_t Kind,
                  const clang::OffloadBundlerConfig &Config,
                  llvm::vfs::FileSystem &FS)
      : Kind(Kind), Config(Config), FS(FS) {}

  /// Return true when results use OutputBuffers, including restored cache hits.
  /// Taking individual buffers does not change this flag.
  bool hasInMemoryOutputs() const { return !OutputBuffers.empty(); }

  /// Transfer the output at \p Index to the caller, retaining its backing
  /// storage independently of this command. Requires hasInMemoryOutputs(), an
  /// in-range Index, and a buffer that has not already been taken.
  std::unique_ptr<llvm::MemoryBuffer> takeOutputBuffer(size_t Index) {
    return std::move(OutputBuffers[Index]);
  }

  /// Allow caching only for nonempty sequences of compressed bundles whose
  /// headers each declare at most 64 MiB of uncompressed data. Archives and
  /// unreadable or malformed headers are excluded without decompressing input.
  bool canCache() const override;

  /// Restore cached outputs as copies, retaining no views into CachedBuffer.
  /// Diagnostic files are written only when AMD_COMGR_SAVE_TEMPS is enabled.
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;

  /// Restore cached members in Config.OutputFileNames order. If supplied,
  /// Owner must own CachedBuffer: aligned members retain it, while misaligned
  /// members are copied. A null Owner makes all members independent copies.
  /// Return framing errors or errors writing requested diagnostic files.
  llvm::Error restoreCachedOutput(llvm::StringRef CachedBuffer,
                                  std::shared_ptr<llvm::MemoryBuffer> Owner);

  /// Serialize memory outputs or output files using aligned CUB1 framing.
  /// The returned view is valid until the next call or command destruction.
  /// Call before transferring any buffers with takeOutputBuffer().
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;

  /// Replace prior results with memory outputs for bitcode/object bundles or
  /// files for archives. Log unbundling and diagnostic-file errors to LogS.
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  ~UnbundleCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  /// Hash every concatenated compressed header for device-only requests. Hash
  /// the full input when a host is requested: host fallback can return those
  /// original bytes, so different compressed encodings need distinct keys.
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
