//===- comgr-options-cache.h - LLVM cl::opt fingerprint cache -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// LLVM's `cl::opt` registry is process-global and unsynchronized, so
/// mutating it (via `clearLLVMOptions()` / `parseLLVMOptions()`) around every
/// compiler invocation forces full serialization of `executeCommand()`
/// callers. This file tracks the `-mllvm` option set that is currently
/// active in the registry as a fingerprint: callers requesting the same,
/// already-active set may run concurrently under a shared lock; a caller
/// requesting a different set is serialized against everyone else just long
/// enough to reset and reapply the registry, then proceeds under an
/// exclusive lock.
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_OPTIONS_CACHE_H
#define COMGR_OPTIONS_CACHE_H

#include <llvm/ADT/ArrayRef.h>

#include <array>
#include <cstdint>
#include <string>

namespace COMGR {

using OptionsFingerprint = std::array<uint8_t, 32>;

/// Compute the fingerprint of an ordered `-mllvm` argument list. Order
/// matters (LLVM's `cl::opt` parsing is last-wins for repeated options), so
/// the arguments are not sorted or deduplicated.
OptionsFingerprint
computeOptionsFingerprint(llvm::ArrayRef<std::string> MLLVMArgs);

/// RAII scope returned by `acquireOptionsScope()`. While held, LLVM's global
/// `cl::opt` registry is guaranteed to reflect the `-mllvm` arguments that
/// were passed to `acquireOptionsScope()`, and it is safe to run a compiler
/// invocation that reads it. Move-only.
class OptionsScopeGuard {
public:
  OptionsScopeGuard(OptionsScopeGuard &&Other) noexcept;
  OptionsScopeGuard &operator=(OptionsScopeGuard &&Other) noexcept;
  OptionsScopeGuard(const OptionsScopeGuard &) = delete;
  OptionsScopeGuard &operator=(const OptionsScopeGuard &) = delete;
  ~OptionsScopeGuard();

  /// True if this guard holds the options lock exclusively, i.e. this call
  /// was the one that reset and reapplied the `cl::opt` registry. False
  /// means the registry already matched and was left untouched, and the
  /// lock is held shared alongside any other same-fingerprint callers.
  bool isExclusive() const { return Exclusive; }

private:
  friend OptionsScopeGuard
  acquireOptionsScope(llvm::ArrayRef<std::string> MLLVMArgs);

  explicit OptionsScopeGuard(bool Exclusive);
  void release();

  bool Exclusive;
  bool Owns;
};

/// Acquire the right to run a compiler job whose `-mllvm` arguments are
/// `MLLVMArgs`, blocking as needed. See `OptionsScopeGuard` and
/// `isExclusive()` for what the caller may assume once this returns.
OptionsScopeGuard acquireOptionsScope(llvm::ArrayRef<std::string> MLLVMArgs);

/// Test-only: forget the tracked fingerprint so the next
/// `acquireOptionsScope()` call is treated as cold regardless of what the
/// `cl::opt` registry actually contains. Must not be called while any
/// `OptionsScopeGuard` is outstanding.
void resetOptionsCacheStateForTest();

} // namespace COMGR

#endif
