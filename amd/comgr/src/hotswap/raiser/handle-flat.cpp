//===- handle-flat.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-mc-tables.h"
#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/raiser/flat-addr.h"
#include "hotswap/raiser/op-resolver.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::hotswap {

// A dword global access is aligned to its own width.
static constexpr Align DwordGlobalAccessAlignment = Align::Constant<4>();

static Error emitGlobalLoad(RaiseContext &Ctx, const DecodedInst &Di,
                            OpResolver &Op) {
  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();

  Expected<Value *> Address =
      emitGlobalAddress(Ctx, Di, DwordGlobalAccessAlignment);
  if (!Address)
    return Address.takeError();

  // An inactive lane holds an unconstrained address, so the load itself is
  // predicated and not only the register write it feeds.
  Ctx.registers().writeReg32UnderExec(*Destination, [&] {
    return Ctx.B.CreateAlignedLoad(Ctx.B.getInt32Ty(), *Address,
                                   DwordGlobalAccessAlignment, "global_load");
  });
  return Error::success();
}

static Error emitGlobalStore(RaiseContext &Ctx, const DecodedInst &Di) {
  int DataIndex = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                     AMDGPU::OpName::vdata);
  assert(DataIndex >= 0 && "global store is missing its data operand");
  Expected<Value *> Data = Ctx.registers().readOp32(Di, DataIndex);
  if (!Data)
    return Data.takeError();

  Expected<Value *> Address =
      emitGlobalAddress(Ctx, Di, DwordGlobalAccessAlignment);
  if (!Address)
    return Address.takeError();

  // A store by an inactive lane must not reach memory at all, so the whole
  // access is predicated on the lane bit of EXEC.
  Ctx.registers().emitUnderExec([&] {
    Ctx.B.CreateAlignedStore(*Data, *Address, DwordGlobalAccessAlignment);
  });
  return Error::success();
}

Error handleFLAT(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::GLOBAL_LOAD_B32:
    return emitGlobalLoad(Ctx, Di, Op);
  case CanonicalOp::GLOBAL_STORE_B32:
    return emitGlobalStore(Ctx, Di);
  default:
    return unsupported(Ctx, Di, "unsupported flat memory operation");
  }
}

} // namespace COMGR::hotswap
