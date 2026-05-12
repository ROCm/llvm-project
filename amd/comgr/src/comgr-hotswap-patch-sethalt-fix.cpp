//===- comgr-hotswap-patch-sethalt-fix.cpp - in-shader sethalt fix ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applySethaltFixPatch. Defensive hotswap
/// pass for the GFX1250 A0 LD_SCALE+WMMA clause-break bug as triggered
/// by the in-shader `s_sethalt` instruction.
///
/// On GFX1250 A0, the SQ has an FSM bug where a wave halt issued just
/// before LD_SCALE breaks the implicit LD_SCALE+WMMA clause: an
/// asynchronous clause-break event during the halt allows other waves
/// to interleave between this wave's LD_SCALE and its WMMA, which
/// either picks up a leaked scale (corrupting another wave's WMMA) or
/// loses its own scale (this wave's WMMA runs unscaled).
///
/// "Software running on A0 cannot use SQ_CMD.HALT operations" covers
/// two halt sources:
///   1. `s_sethalt` -- in-shader instruction. Reachable from compiler
///      code objects; this pass handles it.
///   2. SQ_CMD.HALT (SETHALT) -- external register write from the host
///      (debugger breakpoints, rocprofiler PC sampling, kernel CWSR
///      paths). NOT reachable from COMGR; needs ROCdbgapi /
///      rocprofiler / runtime changes to skip on GFX1250 A0. Out of
///      scope for this pass.
///
/// LLVM almost never emits `s_sethalt` in production code -- it appears
/// only via the `int_amdgcn_s_sethalt` intrinsic, which is largely
/// confined to debug builds. This pass exists for defense-in-depth
/// against any input ELF (compiler-emitted or hand-written) that
/// contains the instruction, and would otherwise hit the SQ FSM bug
/// when the kernel also uses VOP3PX2.
///
/// Strategy: detect every `s_sethalt` (mnemonic match against the MC
/// printer output) and rewrite to `s_nop 0` in-place. Same 4 bytes, no
/// relocation, no length change. Replacement bytes come from
/// LS.SNopBytes (pre-encoded at initLLVM() time -- same source the
/// splitter uses for trampoline padding). The shader proceeds without
/// the halt; if the halt was for an in-shader debug breakpoint, that's
/// the correct trade-off on A0 (the debugger should switch to
/// trap-based breakpoints, which is the source-2 fix the debugger team
/// owns).
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

constexpr StringLiteral SSethaltMnemonic = "s_sethalt";

uint32_t applySethaltFixPatchImpl(PatchContext &Ctx) {
  if (Ctx.LS.SNopBytes.size() != MinInstSize) {
    log() << "hotswap: error: sethalt-fix: LS.SNopBytes is not "
          << MinInstSize << " bytes (got " << Ctx.LS.SNopBytes.size() << ")\n";
    return 0;
  }

  uint32_t Patched = 0;
  for (const InternalDecodedInst &DI : Ctx.Decoded) {
    if (DI.Mnemonic != SSethaltMnemonic)
      continue;
    if (DI.Size != MinInstSize)
      continue;
    if (DI.Offset + DI.Size > Ctx.TextSize)
      continue;

    std::memcpy(Ctx.Text + DI.Offset, Ctx.LS.SNopBytes.data(), MinInstSize);

    log() << "hotswap: sethalt-fix: neutralized s_sethalt at offset 0x"
          << utohexstr(DI.Offset) << "\n";
    ++Patched;
  }

  return Patched;
}

} // namespace

void registerSethaltFixPatch(HotswapPatchVTable &VT) {
  VT.applySethaltFixPatch = &applySethaltFixPatchImpl;
}

} // namespace hotswap
} // namespace COMGR
