//===- OffloadBundler.h - File Bundling and Unbundling ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines an offload bundling API that bundles different files
/// that relate with the same source code but different targets into a single
/// one. Also the implements the opposite functionality, i.e. unbundle files
/// previous created by this API.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_OFFLOADBUNDLER_H
#define LLVM_CLANG_DRIVER_OFFLOADBUNDLER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <llvm/Support/MemoryBuffer.h>
#include <string>

namespace llvm::vfs {
class FileSystem;
}

namespace clang {

class OffloadBundlerConfig {
public:
  OffloadBundlerConfig();

  bool AllowNoHost = false;
  bool AllowMissingBundles = false;
  bool CheckInputArchive = false;
  bool PrintExternalCommands = false;
  bool HipOpenmpCompatible = false;
  bool Compress = false;
  bool Verbose = false;
  llvm::compression::Format CompressionFormat;
  int CompressionLevel;
  uint16_t CompressedBundleVersion;

  unsigned BundleAlignment = 1;
  unsigned HostInputIndex = ~0u;

  std::string FilesType;
  std::string ObjcopyPath;

  llvm::SmallVector<std::string, 4> TargetNames;
  llvm::SmallVector<std::string, 4> InputFileNames;
  llvm::SmallVector<std::string, 4> OutputFileNames;
};

class OffloadBundler {
public:
  const OffloadBundlerConfig &BundlerConfig;

  // TODO: Add error checking from ClangOffloadBundler.cpp
  OffloadBundler(const OffloadBundlerConfig &BC) : BundlerConfig(BC) {}

  // List bundle IDs. Return true if an error was found.
  static llvm::Error
  ListBundleIDsInFile(llvm::StringRef InputFileName,
                      const OffloadBundlerConfig &BundlerConfig);

  llvm::Error BundleFiles();

  /// Unbundle the configured input, including concatenated bundles. Read via
  /// the borrowed FS when supplied, or the real filesystem/stdin otherwise.
  /// FS and WriteOutput need only remain valid for this synchronous call.
  ///
  /// With WriteOutput, pass each configured output filename and an owning
  /// buffer to the callback, not necessarily in OutputFileNames order. Copy the
  /// name if it must be retained; buffers may outlive this call. Binary members
  /// retain their backing image without copying and may be unaligned or lack
  /// a trailing NUL. Other formats copy through the stream-based reader.
  /// Missing-bundle and host fallback rules are the same as for file outputs.
  ///
  /// A callback error stops extraction and is returned to the caller; outputs
  /// already delivered are not rolled back. Without WriteOutput, outputs are
  /// written to the real filesystem even when FS is supplied.
  llvm::Error UnbundleFiles(
      llvm::vfs::FileSystem *FS = nullptr,
      llvm::function_ref<llvm::Error(llvm::StringRef,
                                     std::unique_ptr<llvm::MemoryBuffer>)>
          WriteOutput = nullptr);
  llvm::Error UnbundleArchive();
};

/// Obtain the offload kind, real machine triple, and an optional TargetID
/// out of the target information specified by the user.
/// Bundle Entry ID (or, Offload Target String) has following components:
///  * Offload Kind - Host, OpenMP, or HIP
///  * Triple - Standard LLVM Triple
///  * TargetID (Optional) - target ID, like gfx906:xnack+ or sm_30
struct OffloadTargetInfo {
  llvm::StringRef OffloadKind;
  llvm::Triple Triple;
  llvm::StringRef TargetID;

  const OffloadBundlerConfig &BundlerConfig;

  OffloadTargetInfo(const llvm::StringRef Target,
                    const OffloadBundlerConfig &BC);
  bool hasHostKind() const;
  bool isOffloadKindValid() const;
  bool isOffloadKindCompatible(const llvm::StringRef TargetOffloadKind) const;
  bool isTripleValid() const;
  bool operator==(const OffloadTargetInfo &Target) const;
  std::string str() const;
};

/// Check whether the bundle id is in the following format:
/// <kind>-<triple>[-<target id>[:target features]]
/// <triple> := <arch>-<vendor>-<os>-<env>
bool checkOffloadBundleID(const llvm::StringRef Str);
} // namespace clang

#endif // LLVM_CLANG_DRIVER_OFFLOADBUNDLER_H
