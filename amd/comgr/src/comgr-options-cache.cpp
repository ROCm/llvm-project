//===- comgr-options-cache.cpp - LLVM cl::opt fingerprint cache -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-options-cache.h"

#include "comgr-cache-command.h"
#include "comgr-compiler.h"
#include "comgr.h"

#include <optional>
#include <shared_mutex>
#include <vector>

namespace COMGR {
using namespace llvm;

namespace {
std::shared_mutex OptionsMutex;

// Guarded by OptionsMutex. The fingerprint of the `-mllvm` set that is
// currently applied to LLVM's global `cl::opt` registry. Starts at the
// fingerprint of an empty list, matching the "all options at default"
// state that a freshly-initialized process is in. `std::nullopt` means the
// registry's contents are unknown (e.g. a previous reapply attempt
// failed partway through) and must not be trusted by any fingerprint,
// including a subsequent request for an empty list.
std::optional<OptionsFingerprint> ActiveFingerprint =
    computeOptionsFingerprint({});
} // namespace

OptionsFingerprint computeOptionsFingerprint(ArrayRef<std::string> MLLVMArgs) {
  CachedCommandAdaptor::HashAlgorithm H;
  CachedCommandAdaptor::addUInt(H, MLLVMArgs.size());
  for (const std::string &Arg : MLLVMArgs) {
    CachedCommandAdaptor::addString(H, Arg);
  }
  return H.final();
}

OptionsScopeGuard::OptionsScopeGuard(bool Exclusive)
    : Exclusive(Exclusive), Owns(true) {}

OptionsScopeGuard::OptionsScopeGuard(OptionsScopeGuard &&Other) noexcept
    : Exclusive(Other.Exclusive), Owns(Other.Owns) {
  Other.Owns = false;
}

OptionsScopeGuard &
OptionsScopeGuard::operator=(OptionsScopeGuard &&Other) noexcept {
  if (this != &Other) {
    release();
    Exclusive = Other.Exclusive;
    Owns = Other.Owns;
    Other.Owns = false;
  }
  return *this;
}

void OptionsScopeGuard::release() {
  if (!Owns) {
    return;
  }
  if (Exclusive) {
    OptionsMutex.unlock();
  } else {
    OptionsMutex.unlock_shared();
  }
  Owns = false;
}

OptionsScopeGuard::~OptionsScopeGuard() { release(); }

OptionsScopeGuard acquireOptionsScope(ArrayRef<std::string> MLLVMArgs) {
  OptionsFingerprint FP = computeOptionsFingerprint(MLLVMArgs);

  OptionsMutex.lock_shared();
  if (ActiveFingerprint == FP) {
    return OptionsScopeGuard(/*Exclusive=*/false);
  }
  OptionsMutex.unlock_shared();

  OptionsMutex.lock();
  // Re-check: another thread may have already applied FP while we were
  // waiting for the exclusive lock.
  if (ActiveFingerprint != FP) {
    clearLLVMOptions();
    std::vector<std::string> Args(MLLVMArgs.begin(), MLLVMArgs.end());
    if (parseLLVMOptions(Args) == AMD_COMGR_STATUS_SUCCESS) {
      ActiveFingerprint = FP;
    } else {
      // Don't publish FP for a half-applied option set: reset to unknown
      // so the next caller -- regardless of what it asks for -- retries
      // the full reset-and-reapply path rather than trusting this one.
      ActiveFingerprint = std::nullopt;
    }
  }
  return OptionsScopeGuard(/*Exclusive=*/true);
}

void resetOptionsCacheStateForTest() {
  std::unique_lock<std::shared_mutex> Lock(OptionsMutex);
  ActiveFingerprint = computeOptionsFingerprint({});
}

} // namespace COMGR
