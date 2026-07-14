//===- handle-valu-internal.h - Hotswap transpiler ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLE_VALU_INTERNAL_H
#define HOTSWAP_TRANSPILER_HANDLE_VALU_INTERNAL_H

#include "raise-context.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"

#include <cassert>

// Sub-handlers that together make up `handleVALU`. Each returns
// `{handled=true}` when it recognised and lowered the CanonicalOp, or an
// unhandled HandlerResult when the CanonicalOp is out of its scope (so the
// top-level router can try the next sub-handler).
//
// Sub-handlers are private to the handle_valu.* translation units;
// they are not exposed to the format dispatcher in raiser.cpp.

namespace COMGR::hotswap {

// Cross-lane primitives: V_READFIRSTLANE_B32, V_READLANE_B32,
// V_WRITELANE_B32, V_MBCNT_{LO,HI}_U32_B32, V_PERMLANE{16,X16,64}_B32,
// V_PERMLANE{16,32}_SWAP_B32. Isolated because the cross-wave
// strategy (hotswap/docs/wave-size-translation.md sec. sec. 5.3 and 7) keeps
// iterating on exactly this surface.
llvm::Expected<HandlerResult>
handleValuCrossLane(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op);

// "Small ops": type conversions, F16 arith, 16-bit shifts / min /
// max, byte pack, V_BFREV_B32, V_NOT_B32, F32 single-src
// transcendentals. See `handle-valu-small-ops.cpp` for the exact list.
llvm::Expected<HandlerResult>
handleValuSmallOps(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op);

// Vector compares (V_CMP / V_CMPX collapsed onto two SemOps with
// VCmpMeta side-table lookup) including cross-wave projection of the
// ballot result back to source-EXEC width.
llvm::Expected<HandlerResult>
handleValuVcmp(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op);

// VOP3P packed ops (V_PK_*_F32, V_PK_MOV_B32), WMMA (V_WMMA_F32_*),
// v_fma_mix_f32, and v_cndmask_b32.
llvm::Expected<HandlerResult>
handleValuVoP3P(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op);

// Read the per-lane wave-mask condition operand (src2) shared by the
// V_CNDMASK_B32 (VOP3P handler) and V_CNDMASK_B16 (true16 handler) lifts.
// Handles the SGPR fresh-compare / shadow fallback, wave32 vcc_hi/exec_hi
// scratch, and default-VCC routing; always returns a non-null per-lane `i1`.
llvm::Value *raiseCndmaskWaveCondition(RaiseContext &Ctx, const DecodedInst &Di,
                                       OpResolver &Op);

// Emit unsigned i16 multiply-add for either `i16` or `<N x i16>` operands.
// `A`, `B`, and `C` must have the same type; `WideTy` and `ClampMax` must be
// the matching scalar/vector i32 shape used for the exact widened sum.
inline llvm::Value *emitU16Mad(RaiseContext &Ctx, llvm::Value *A,
                               llvm::Value *B, llvm::Value *C, bool Clamp,
                               llvm::Type *WideTy, llvm::Constant *ClampMax,
                               llvm::StringRef Name) {
  assert(A->getType() == B->getType() && A->getType() == C->getType() &&
         "u16 MAD operands must have the same type");
  assert(A->getType()->getScalarType()->isIntegerTy(16) &&
         "u16 MAD operands must be i16 lanes");
  assert(WideTy->getScalarType()->isIntegerTy(32) &&
         "u16 MAD widened type must be i32 lanes");
  assert(ClampMax->getType() == WideTy &&
         "u16 MAD clamp constant must match the widened type");

  if (!Clamp) {
    return Ctx.B.CreateAdd(Ctx.B.CreateMul(A, B, llvm::Twine(Name) + "_mul"), C,
                           Name);
  }

  llvm::Value *WideA =
      Ctx.B.CreateZExt(A, WideTy, llvm::Twine(Name) + "_a_wide");
  llvm::Value *WideB =
      Ctx.B.CreateZExt(B, WideTy, llvm::Twine(Name) + "_b_wide");
  llvm::Value *WideC =
      Ctx.B.CreateZExt(C, WideTy, llvm::Twine(Name) + "_c_wide");
  llvm::Value *Wide = Ctx.B.CreateAdd(
      Ctx.B.CreateMul(WideA, WideB, llvm::Twine(Name) + "_mul_wide"), WideC,
      llvm::Twine(Name) + "_wide");
  llvm::Function *UminFn = llvm::Intrinsic::getOrInsertDeclaration(
      &Ctx.M, llvm::Intrinsic::umin, {WideTy});
  llvm::Value *Sat =
      Ctx.B.CreateCall(UminFn, {Wide, ClampMax}, llvm::Twine(Name) + "_clamp");
  return Ctx.B.CreateTrunc(Sat, A->getType(), Name);
}

} // namespace COMGR::hotswap

#endif
