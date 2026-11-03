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

  /// A cache hit's output kept in memory instead of written via
  /// C.writeExecuteOutput(). Output is a view into Entry; keep Entry alive
  /// for as long as Output (or any slice of it) is used.
  struct CachedOutput {
    std::shared_ptr<llvm::MemoryBuffer> Entry;
    llvm::StringRef Output;
  };

  /// Checks if Command C is cached; if so, replaces its output and logs its
  /// error stream. If Hit is set, a cache hit is returned there instead of
  /// via C.writeExecuteOutput(). Otherwise executes C directly.
  amd_comgr_status_t execute(CachedCommandAdaptor &C, llvm::raw_ostream &LogS,
                             CachedOutput *Hit = nullptr);
};
} // namespace COMGR

#endif
