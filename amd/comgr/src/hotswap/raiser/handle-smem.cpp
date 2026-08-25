//===- handle-smem.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/amdgpu-mc-tables.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>

using namespace llvm;

namespace COMGR::hotswap {

// Return the index assigned to a TableGen-named operand, if present.
static std::optional<unsigned> namedOperandIndex(const DecodedInst &Di,
                                                 AMDGPU::OpName Name) {
  const int Index =
      COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  return Index < 0 ? std::nullopt : std::optional(static_cast<unsigned>(Index));
}

// Return the index of an operand every mapped scalar load must carry.
static unsigned requiredNamedOperandIndex(const MCState &MC,
                                          const DecodedInst &Di,
                                          AMDGPU::OpName Name,
                                          StringRef OperandName) {
  const std::optional<unsigned> Index = namedOperandIndex(Di, Name);
  if (!Index)
    report_fatal_error(Twine("transpiler: instruction '") +
                       strippedMnemonic(MC, Di.Inst) + "' (MC opcode " +
                       Twine(Di.Inst.getOpcode()) +
                       ") is missing required operand '" + OperandName +
                       "' (OpName " + Twine(static_cast<unsigned>(Name)) + ")");
  return *Index;
}

// Return the data width for a supported non-buffer scalar load.
static std::optional<unsigned> scalarLoadWidthInDwords(CanonicalOp Operation) {
  switch (Operation) {
  case CanonicalOp::S_LOAD_B32:
    return 1;
  case CanonicalOp::S_LOAD_B64:
    return 2;
  case CanonicalOp::S_LOAD_B128:
    return 4;
  default:
    return std::nullopt;
  }
}

Error handleSMEM(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &) {
  const auto Failure = [&](const Twine &Detail) {
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags), Detail);
  };

  const std::optional<unsigned> LoadWidthInDwords =
      scalarLoadWidthInDwords(Di.CanonOp);
  if (!LoadWidthInDwords)
    return Failure("unsupported scalar memory operation");

  const unsigned DestinationIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::sdst, "sdst");
  const unsigned BaseIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::sbase, "sbase");
  const unsigned CachePolicyIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::cpol, "cpol");
  const std::optional<unsigned> OffsetIndex =
      namedOperandIndex(Di, AMDGPU::OpName::offset);
  const std::optional<unsigned> ScalarOffsetIndex =
      namedOperandIndex(Di, AMDGPU::OpName::soffset);
  if (DestinationIndex >= Di.numOperands() || BaseIndex >= Di.numOperands() ||
      CachePolicyIndex >= Di.numOperands() ||
      (OffsetIndex && *OffsetIndex >= Di.numOperands()) ||
      (ScalarOffsetIndex && *ScalarOffsetIndex >= Di.numOperands()))
    return Failure("scalar load operand list is incomplete");
  if (!Di.isReg(DestinationIndex))
    return Failure("scalar load destination must be a register");
  if (!Di.isReg(BaseIndex))
    return Failure("scalar load base must be a register");
  if (ScalarOffsetIndex || !OffsetIndex || !Di.isImm(*OffsetIndex))
    return Failure("only immediate scalar load offsets are supported");
  if (!Di.isImm(CachePolicyIndex))
    return Failure("scalar load cache policy must be an immediate");
  if (Di.getImm(CachePolicyIndex) != 0)
    return Failure("non-default scalar load modifiers are not supported");

  const int64_t ImmediateOffset = Di.getImm(*OffsetIndex);
  if (ImmediateOffset < 0)
    return Failure("negative scalar load offsets are not supported");

  Expected<ParsedReg> Destination =
      Ctx.registers().parseReg(Di, DestinationIndex);
  if (!Destination)
    return Destination.takeError();
  if (Destination->RegKind != ParsedReg::SGPR || !Destination->BaseIdx)
    return Failure("scalar load requires an SGPR destination");
  if (Destination->WidthInDwords != *LoadWidthInDwords)
    return Failure("scalar load destination width does not match opcode");

  Expected<ParsedReg> Base = Ctx.registers().parseReg(Di, BaseIndex);
  if (!Base)
    return Base.takeError();
  if (Base->RegKind != ParsedReg::SGPR || !Base->BaseIdx ||
      Base->WidthInDwords != 2)
    return Failure("scalar load requires an SGPR-pair base");

  Expected<Value *> BaseValue = Ctx.registers().readOp64(Di, BaseIndex);
  if (!BaseValue)
    return BaseValue.takeError();

  Type *I64Ty = Ctx.B.getInt64Ty();
  // SMEM aligns each address component before adding them.
  Value *AlignedBase = Ctx.B.CreateAnd(
      *BaseValue, ConstantInt::get(I64Ty, ~uint64_t(3)), "smem_base");
  const uint64_t Offset = static_cast<uint64_t>(ImmediateOffset) & ~uint64_t(3);
  Value *Address = Ctx.B.CreateAdd(AlignedBase, ConstantInt::get(I64Ty, Offset),
                                   "smem_addr");
  PointerType *PointerTy =
      PointerType::get(Ctx.B.getContext(), AMDGPUAS::GLOBAL_ADDRESS);
  Value *Pointer = Ctx.B.CreateIntToPtr(Address, PointerTy, "smem_ptr");

  Type *LoadType = Ctx.B.getInt32Ty();
  if (*LoadWidthInDwords != 1)
    LoadType = FixedVectorType::get(Ctx.B.getInt32Ty(), *LoadWidthInDwords);
  Value *Loaded =
      Ctx.B.CreateAlignedLoad(LoadType, Pointer, Align(4), "smem_load");
  if (*LoadWidthInDwords == 1)
    Ctx.registers().writeReg32(*Destination, Loaded);
  else
    Ctx.registers().writeRegVec(*Destination, Loaded);
  return Error::success();
}

} // namespace COMGR::hotswap
