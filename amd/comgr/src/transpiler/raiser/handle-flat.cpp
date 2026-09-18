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
#include "transpiler/raiser/reg-file.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

#include <optional>

using namespace llvm;

namespace COMGR::transpiler {

// Ordinary GLOBAL dword tuples guarantee only dword alignment.
static constexpr Align GlobalAccessAlignment = Align::Constant<4>();

/// Return the transfer width in dwords, or no value for other operations.
static std::optional<unsigned>
globalAccessWidthInDwords(CanonicalOp Operation) {
  switch (Operation) {
  case CanonicalOp::GLOBAL_LOAD_B32:
  case CanonicalOp::GLOBAL_STORE_B32:
    return 1;
  case CanonicalOp::GLOBAL_LOAD_B64:
  case CanonicalOp::GLOBAL_STORE_B64:
    return 2;
  case CanonicalOp::GLOBAL_LOAD_B96:
  case CanonicalOp::GLOBAL_STORE_B96:
    return 3;
  case CanonicalOp::GLOBAL_LOAD_B128:
  case CanonicalOp::GLOBAL_STORE_B128:
    return 4;
  default:
    return std::nullopt;
  }
}

/// Return the LLVM IR type for a transfer of the given dword width.
static Type *globalAccessType(IRBuilder<> &B, unsigned WidthInDwords) {
  assert(WidthInDwords >= 1 && WidthInDwords <= 4 &&
         "unsupported GLOBAL access width");
  if (WidthInDwords == 1)
    return B.getInt32Ty();
  if (WidthInDwords == 2)
    return B.getInt64Ty();
  return FixedVectorType::get(B.getInt32Ty(), WidthInDwords);
}

/// Parse and validate the named GLOBAL data register tuple.
static Expected<ParsedReg>
globalDataReg(RaiseContext &Ctx, const DecodedInst &Di, AMDGPU::OpName Name,
              unsigned WidthInDwords, StringRef Role) {
  int Index = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  assert(Index >= 0 && static_cast<unsigned>(Index) < Di.numOperands() &&
         "global memory operation is missing its data operand");
  assert(Di.isReg(Index) && "global memory data operand must be a register");

  Expected<ParsedReg> Reg = Ctx.registers().parseReg(Di, Index);
  if (!Reg)
    return Reg.takeError();
  if (Reg->RegKind != ParsedReg::VGPR && Reg->RegKind != ParsedReg::AGPR)
    return unsupported(
        Ctx, Di, Twine("global memory ") + Role + " must be a VGPR or AGPR");
  assert(Reg->BaseIdx && "global memory data register has no base index");
  assert(Reg->WidthInDwords == WidthInDwords &&
         "global memory data register width does not match the opcode");
  return *Reg;
}

/// Emit a GLOBAL load of the given dword width under EXEC.
static Error emitGlobalLoad(RaiseContext &Ctx, const DecodedInst &Di,
                            unsigned WidthInDwords) {
  Expected<ParsedReg> Destination = globalDataReg(Ctx, Di, AMDGPU::OpName::vdst,
                                                  WidthInDwords, "destination");
  if (!Destination)
    return Destination.takeError();

  Expected<Value *> Address = emitGlobalAddress(Ctx, Di, GlobalAccessAlignment);
  if (!Address)
    return Address.takeError();

  // An inactive lane holds an unconstrained address, so the load itself is
  // predicated and not only the register write it feeds.
  Ctx.registers().emitUnderExec([&] {
    Value *Loaded =
        Ctx.B.CreateAlignedLoad(globalAccessType(Ctx.B, WidthInDwords),
                                *Address, GlobalAccessAlignment, "global_load");
    Ctx.registers().regFile().writeRegVec(Ctx.B, *Destination, Loaded);
  });
  return Error::success();
}

/// Emit a GLOBAL store of the given dword width under EXEC.
static Error emitGlobalStore(RaiseContext &Ctx, const DecodedInst &Di,
                             unsigned WidthInDwords) {
  Expected<ParsedReg> DataReg =
      globalDataReg(Ctx, Di, AMDGPU::OpName::vdata, WidthInDwords, "source");
  if (!DataReg)
    return DataReg.takeError();
  Value *Data = Ctx.registers().regFile().readRegVec(
      Ctx.B, *DataReg, globalAccessType(Ctx.B, WidthInDwords));

  Expected<Value *> Address = emitGlobalAddress(Ctx, Di, GlobalAccessAlignment);
  if (!Address)
    return Address.takeError();

  // A store by an inactive lane must not reach memory at all, so the whole
  // access is predicated on the lane bit of EXEC.
  Ctx.registers().emitUnderExec(
      [&] { Ctx.B.CreateAlignedStore(Data, *Address, GlobalAccessAlignment); });
  return Error::success();
}

Error handleVGLOBAL(RaiseContext &Ctx, const DecodedInst &Di,
                    OperandResolver &) {
  std::optional<unsigned> WidthInDwords = globalAccessWidthInDwords(Di.CanonOp);
  if (!WidthInDwords)
    return unsupported(Ctx, Di, "unsupported flat memory operation");

  switch (Di.CanonOp) {
  case CanonicalOp::GLOBAL_LOAD_B32:
  case CanonicalOp::GLOBAL_LOAD_B64:
  case CanonicalOp::GLOBAL_LOAD_B96:
  case CanonicalOp::GLOBAL_LOAD_B128:
    return emitGlobalLoad(Ctx, Di, *WidthInDwords);
  case CanonicalOp::GLOBAL_STORE_B32:
  case CanonicalOp::GLOBAL_STORE_B64:
  case CanonicalOp::GLOBAL_STORE_B96:
  case CanonicalOp::GLOBAL_STORE_B128:
    return emitGlobalStore(Ctx, Di, *WidthInDwords);
  default:
    llvm_unreachable("classified global memory operation has no handler");
  }
}

} // namespace COMGR::transpiler
