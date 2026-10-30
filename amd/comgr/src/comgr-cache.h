//===- comgr-cache.h - Comgr Cache implementation -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_CACHE_H
#define COMGR_CACHE_H

#include "amd_comgr.h"
#include "comgr-cache-command.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CachePruning.h>
#include <llvm/Support/MemoryBuffer.h>

#include <functional>
#include <memory>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR {
class CommandCache {
  std::string CacheDir;
  llvm::CachePruningPolicy Policy;
  llvm::raw_ostream &LogS;

  CommandCache(llvm::StringRef CacheDir, const llvm::CachePruningPolicy &Policy,
               llvm::raw_ostream &LogS);

  static std::optional<llvm::CachePruningPolicy>
  getPolicyFromEnv(llvm::raw_ostream &LogS);

public:
  static std::unique_ptr<CommandCache> get(llvm::raw_ostream &);

  ~CommandCache();
  void prune();

  /// An owning cache entry and a view of its serialized command output.
  /// Retain Entry while using Output or any views derived from it.
  struct CachedOutput {
    std::shared_ptr<llvm::MemoryBuffer> Entry;
    llvm::StringRef Output;
  };

  /// Run C, reusing cached output and replaying cached diagnostics to LogS when
  /// possible. On a miss, execute C and cache successful output if possible.
  ///
  /// With Hit, return the entry and output view without calling
  /// C.writeExecuteOutput(). The caller must restore the command's output and
  /// retry C if restoration fails. Hit must be empty on entry; it is unchanged
  /// on a miss or cache bypass.
  ///
  /// With a null Hit, restore through C.writeExecuteOutput(). Use this mode
  /// when retaining a cache mapping would prevent file removal, as on Windows.
  amd_comgr_status_t execute(CachedCommandAdaptor &C, llvm::raw_ostream &LogS,
                             CachedOutput *Hit = nullptr);
};
} // namespace COMGR

#endif
