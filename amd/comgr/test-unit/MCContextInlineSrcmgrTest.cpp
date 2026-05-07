//===- mc_context_inline_srcmgr_test.cpp - mc_context_inline_srcmgr unit tests ---===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression fence for the `InlineSrcMgr`-on-MCContext contract that
// prevents the SIG6 `Either SourceMgr should be available UNREACHABLE`
// abort at `llvm/lib/MC/MCContext.cpp:1093` / `:1120`.
//
// Background
// ----------
// The hotswap legacy path and the hotswap raise path both construct
// MCContexts via
//   std::make_unique<MCContext>(Triple, MAI, MRI, STI)
// which defaults the 5th `SourceMgr *Mgr = nullptr` parameter.  When
// any MC-layer diagnostic reaches `MCContext::reportCommon` or
// `MCContext::diagnose` with a valid SMLoc, both `SrcMgr` and
// `InlineSrcMgr` are null → `llvm_unreachable("Either SourceMgr
// should be available")` fires and the process aborts on SIGABRT.
//
// Manifested as a systemic SIG6 on every Triton kernel run through
// `compare_correctness --lane=legacy` (fixed by attaching an inline
// SourceMgr at four sites — two post-ctor, two post-`MCContext::reset()`).
// The hotswap raise path got the same defensive init in the follow-up this
// test lives alongside.
//
// This test pins the INVARIANT the fix depends on: after
// `COMGR::hotswap::initMCState`, `state.Ctx->getInlineSourceManager()`
// returns a non-null SourceMgr pointer.  A future edit that
// removes the `initInlineSourceManager` call from `mc-state.cpp`
// (or reorders it before the ctor → post-`reset` idiom, which
// `reset()` would silently clear) will fail this test instead of
// silently re-opening the SIG6 hole.
//
// Scope note: this test covers ONLY the hotswap raise path's
// MCContext (`transpiler/mc-state.cpp`), because that's the one
// exposed through a public header we can #include from a gtest.
// The three legacy-path MCContexts (in `hotswap/hotswap.cpp` and
// `src/hotswap.cpp`) are internal to libhsa-runtime and
// carry no test hook; their correctness is pinned by the
// compare_correctness three-way-lane harness (the concrete
// symptom the fix targets) and by code review against this
// pattern.  If a future refactor exposes an MCState-like shim
// from those files, this test should be extended to cover them.

#include "hotswap/mc-state.h"

#include "llvm/MC/MCContext.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <memory>
#include <mutex>

namespace {

// LLVM's AMDGPU target machinery has to be registered before any
// Target lookup in `mc-state.cpp` can succeed.  Run once per test
// process — `std::call_once` makes the init thread-safe against a
// gtest sharded runner, and the target registration is idempotent
// anyway.
void ensureAMDGPURegistered() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
  });
}

} // namespace

// ---------------------------------------------------------------------------
// initMCState must leave the MCContext with a non-null InlineSrcMgr
// ---------------------------------------------------------------------------
//
// This is the contract `mc-state.cpp` establishes by attaching an inline
// SourceMgr to the MCContext immediately after construction.
// If a future edit removes the post-ctor `state.ctx->initInlineSourceManager()`
// call, this test turns red BEFORE a user hits the SIG6 in real
// compare_correctness runs.  The targetISA is chosen as gfx942 —
// the same target the fix was originally exercised against, and the
// one the hotswap legacy path uses for cross-widened runs.  A
// different `createMCSubtargetInfo`-accepted CPU name would work
// just as well; the invariant is a per-MCContext property, not a
// per-ISA one.
TEST(MCContextInlineSrcMgr, HotswapInitMCStateAttachesInlineSourceManager) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState state;
  ASSERT_TRUE(COMGR::hotswap::initMCState(state, "gfx942"))
      << "initMCState('gfx942') must succeed on an AMDGPU-enabled LLVM "
         "build (InitializeAllTargetMCs was just run above)";

  ASSERT_NE(state.Ctx, nullptr)
      << "initMCState must construct an MCContext — the fix we are "
         "pinning lives on that object";

  const llvm::SourceMgr *inline_src_mgr =
      state.Ctx->getInlineSourceManager();
  EXPECT_NE(inline_src_mgr, nullptr)
      << "state.ctx->getInlineSourceManager() must return non-null after "
         "initMCState — the post-ctor `initInlineSourceManager()` call "
         "attaches an inline SourceMgr so MC-layer diagnostics "
         "(MCContext::reportCommon / "
         "MCContext::diagnose) can format a valid SMLoc through the "
         "default diag handler instead of hitting the "
         "`llvm_unreachable(\"Either SourceMgr should be available\")` "
         "abort at llvm/lib/MC/MCContext.cpp:1093 / :1120.  If this "
         "expectation fails, `mc-state.cpp` has almost certainly lost "
         "its `state.ctx->initInlineSourceManager()` call — restore it "
         "immediately; the cost is one empty SourceMgr per MCState.";
}

// ---------------------------------------------------------------------------
// The same invariant must hold for a second MCState constructed later.
// ---------------------------------------------------------------------------
//
// MCContexts in both the hotswap and legacy paths are long-lived
// process globals / cached statics.  LLVM's AMDGPU backend has
// global state that historically doesn't survive repeated MCContext
// lifecycles (see the "retarget count > 0 skip" comment in
// `hotswap/hotswap.cpp`).  This test is mostly a smoke check that
// the second MCState also gets the inline SourceMgr — a regression
// that somehow attached an inline SourceMgr only on the first
// call (e.g. a `static bool once` guard) would be caught here.
TEST(MCContextInlineSrcMgr, SecondMCStateAlsoGetsInlineSourceManager) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState first;
  ASSERT_TRUE(COMGR::hotswap::initMCState(first, "gfx942"));
  EXPECT_NE(first.Ctx->getInlineSourceManager(), nullptr);

  COMGR::hotswap::MCState second;
  ASSERT_TRUE(COMGR::hotswap::initMCState(second, "gfx942"));
  EXPECT_NE(second.Ctx->getInlineSourceManager(), nullptr)
      << "Second initMCState on the same target must also produce an "
         "MCContext with a non-null InlineSrcMgr — a one-shot init "
         "gate on the shim would silently regress any caller after "
         "the first.";
}
