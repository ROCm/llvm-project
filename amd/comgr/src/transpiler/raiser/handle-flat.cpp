//===- handle-flat.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/raiser/flat-addr.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::transpiler {

// A dword global access is aligned to its own width.
static constexpr Align DwordGlobalAccessAlignment = Align::Constant<4>();

static Error emitGlobalLoad(RaiseContext &Ctx, const DecodedInst &Di,
                            OperandResolver &Op) {
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
  int DataIndex = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
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

Error handleFLAT(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::GLOBAL_LOAD_B32:
    return emitGlobalLoad(Ctx, Di, Op);
  case CanonicalOp::GLOBAL_STORE_B32:
    return emitGlobalStore(Ctx, Di);
  default:
    return unsupported(Ctx, Di, "unsupported flat memory operation");
  }
}

} // namespace COMGR::transpiler
