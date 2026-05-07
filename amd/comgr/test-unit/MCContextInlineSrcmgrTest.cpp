//===- MCContextInlineSrcmgrTest.cpp - inline-SourceMgr regression test --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression fence for the `InlineSrcMgr`-on-MCContext contract that
// prevents the abort `Either SourceMgr should be available UNREACHABLE`
// in `llvm/lib/MC/MCContext.cpp`.
//
// Background: the MCContext ctor leaves the inline SourceMgr null, so any
// MC-layer diagnostic reaching `MCContext::reportCommon` /
// `MCContext::diagnose` with a valid SMLoc trips the unreachable and the
// process dies on SIGABRT. `mc-state.cpp` works around that by attaching
// an inline SourceMgr immediately after constructing the MCContext.
//
// This test pins the invariant the workaround depends on: after
// `COMGR::hotswap::initMCState`, `state.Ctx->getInlineSourceManager()`
// returns non-null. A future edit that drops the `initInlineSourceManager`
// call from `mc-state.cpp` (or moves it before a `MCContext::reset()`
// that would silently clear it) fails this test instead of silently
// re-opening the abort hole.

#include "hotswap/mc-state.h"

#include "llvm/MC/MCContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <memory>
#include <mutex>

namespace {

// LLVM's AMDGPU target machinery has to be registered before any
// Target lookup in `mc-state.cpp` can succeed. Run once per test
// process -- `std::call_once` makes the init thread-safe against a
// gtest sharded runner, and the target registration is idempotent
// anyway.
void ensureAMDGPURegistered() {
  static std::once_flag Flag;
  std::call_once(Flag, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
  });
}

} // namespace

// initMCState must leave the MCContext with a non-null InlineSrcMgr.
// Any AMDGPU CPU name that `createMCSubtargetInfo` accepts works; gfx942
// matches the original repro target.
TEST(MCContextInlineSrcMgr, HotswapInitMCStateAttachesInlineSourceManager) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState State;
  llvm::Error InitErr = COMGR::hotswap::initMCState(State, "gfx942");
  ASSERT_FALSE(static_cast<bool>(InitErr))
      << "initMCState('gfx942') must succeed on an AMDGPU-enabled LLVM "
         "build (InitializeAllTargetMCs was just run above): "
      << llvm::toString(std::move(InitErr));

  ASSERT_NE(State.Ctx, nullptr) << "initMCState must construct an MCContext";

  EXPECT_NE(State.Ctx->getInlineSourceManager(), nullptr)
      << "state.Ctx->getInlineSourceManager() must return non-null after "
         "initMCState; if this fails, mc-state.cpp has lost its "
         "initInlineSourceManager() call -- restore it.";
}

// The same invariant must hold for a second MCState. Catches a regression
// that gates the inline SourceMgr attach on a `static bool once` flag
// (silent miscompile after the first call).
TEST(MCContextInlineSrcMgr, SecondMCStateAlsoGetsInlineSourceManager) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState First;
  llvm::Error FirstErr = COMGR::hotswap::initMCState(First, "gfx942");
  ASSERT_FALSE(static_cast<bool>(FirstErr))
      << llvm::toString(std::move(FirstErr));
  EXPECT_NE(First.Ctx->getInlineSourceManager(), nullptr);

  COMGR::hotswap::MCState Second;
  llvm::Error SecondErr = COMGR::hotswap::initMCState(Second, "gfx942");
  ASSERT_FALSE(static_cast<bool>(SecondErr))
      << llvm::toString(std::move(SecondErr));
  EXPECT_NE(Second.Ctx->getInlineSourceManager(), nullptr)
      << "Second initMCState on the same target must also produce an "
         "MCContext with a non-null InlineSrcMgr.";
}
