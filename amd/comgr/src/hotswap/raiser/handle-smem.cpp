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
#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/raiser/operand-resolver.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace llvm;

namespace COMGR::hotswap {

// S_LOAD_B32, S_LOAD_B64, and S_LOAD_B128 ignore the low two bits of both
// address components.
static constexpr Align DwordSmemAddressAlignment = Align::Constant<4>();

// Report decoded operands that contradict the generated instruction metadata.
[[noreturn]] static void invalidOperandLayout(const MCState &MC,
                                              const DecodedInst &Di,
                                              StringRef Detail) {
  std::string Message =
      formatv("transpiler: instruction '{0}' (MC opcode {1}, format {2}, "
              "offset 0x{3:x}) has invalid operand layout: {4}",
              strippedMnemonic(MC, Di.Inst), Di.Inst.getOpcode(),
              formatName(Di.TargetSpecificFlags), Di.Offset, Detail)
          .str();
  report_fatal_error(StringRef(Message));
}

// Return the index assigned to a TableGen-named operand, if present.
static std::optional<unsigned> namedOperandIndex(const MCState &MC,
                                                 const DecodedInst &Di,
                                                 AMDGPU::OpName Name,
                                                 StringRef OperandName) {
  int Index = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Index < 0)
    return std::nullopt;
  if (static_cast<unsigned>(Index) >= Di.numOperands())
    invalidOperandLayout(
        MC, Di,
        formatv("operand '{0}' has index {1}, but the instruction has {2} "
                "operands",
                OperandName, Index, Di.numOperands())
            .str());
  return static_cast<unsigned>(Index);
}

// Return the index of an operand every mapped scalar load must carry.
static unsigned requiredNamedOperandIndex(const MCState &MC,
                                          const DecodedInst &Di,
                                          AMDGPU::OpName Name,
                                          StringRef OperandName) {
  std::optional<unsigned> Index = namedOperandIndex(MC, Di, Name, OperandName);
  if (!Index)
    invalidOperandLayout(MC, Di,
                         formatv("missing required operand '{0}' (OpName {1})",
                                 OperandName, static_cast<unsigned>(Name))
                             .str());
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

Error handleSMEM(RaiseContext &Ctx, const DecodedInst &Di, OperandResolver &) {
  std::optional<unsigned> LoadWidthInDwords =
      scalarLoadWidthInDwords(Di.CanonOp);
  if (!LoadWidthInDwords)
    return unsupported(Ctx, Di, "unsupported scalar memory operation");

  unsigned DestinationIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::sdst, "sdst");
  unsigned BaseIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::sbase, "sbase");
  unsigned CachePolicyIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::cpol, "cpol");
  std::optional<unsigned> OffsetIndex =
      namedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::offset, "offset");
  std::optional<unsigned> ScalarOffsetIndex =
      namedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::soffset, "soffset");
  if (!Di.isReg(DestinationIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'sdst' is not a register");
  if (!Di.isReg(BaseIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'sbase' is not a register");
  if (ScalarOffsetIndex)
    return unsupported(Ctx, Di,
                       "only immediate scalar load offsets are supported");
  if (!OffsetIndex)
    invalidOperandLayout(Ctx.MC, Di,
                         "immediate scalar load has no 'offset' operand");
  if (!Di.isImm(*OffsetIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'offset' is not an immediate");
  if (!Di.isImm(CachePolicyIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'cpol' is not an immediate");
  if (Di.getImm(CachePolicyIndex) != 0)
    return unsupported(Ctx, Di,
                       "non-default scalar load modifiers are not supported");

  int64_t ImmediateOffset = Di.getImm(*OffsetIndex);
  if (ImmediateOffset < 0)
    return unsupported(Ctx, Di,
                       "negative scalar load offsets are not supported");

  Expected<ParsedReg> Destination =
      Ctx.registers().parseReg(Di, DestinationIndex);
  if (!Destination)
    return Destination.takeError();
  if (Destination->RegKind != ParsedReg::SGPR)
    return unsupported(Ctx, Di, "scalar load requires an SGPR destination");
  if (!Destination->BaseIdx)
    invalidOperandLayout(Ctx.MC, Di,
                         "SGPR destination has no base register index");
  if (Destination->WidthInDwords != *LoadWidthInDwords)
    invalidOperandLayout(Ctx.MC, Di,
                         "destination width does not match the load opcode");

  Expected<ParsedReg> Base = Ctx.registers().parseReg(Di, BaseIndex);
  if (!Base)
    return Base.takeError();
  if (Base->RegKind != ParsedReg::SGPR)
    return unsupported(Ctx, Di, "scalar load requires an SGPR-pair base");
  if (!Base->BaseIdx)
    invalidOperandLayout(Ctx.MC, Di, "SGPR base has no base register index");
  if (Base->WidthInDwords != 2)
    invalidOperandLayout(Ctx.MC, Di, "scalar load base is not two dwords");

  Expected<Value *> BaseValue = Ctx.registers().readOp64(Di, BaseIndex);
  if (!BaseValue)
    return BaseValue.takeError();

  Type *I64Ty = Ctx.B.getInt64Ty();
  uint64_t AddressMask =
      maskTrailingZeros<uint64_t>(Log2(DwordSmemAddressAlignment));
  Value *AlignedBase = Ctx.B.CreateAnd(
      *BaseValue, ConstantInt::get(I64Ty, AddressMask), "smem_base");
  uint64_t Offset = alignDown(static_cast<uint64_t>(ImmediateOffset),
                              DwordSmemAddressAlignment.value());
  Value *Address = Ctx.B.CreateAdd(AlignedBase, ConstantInt::get(I64Ty, Offset),
                                   "smem_addr");
  PointerType *PointerTy =
      PointerType::get(Ctx.B.getContext(), AMDGPUAS::GLOBAL_ADDRESS);
  Value *Pointer = Ctx.B.CreateIntToPtr(Address, PointerTy, "smem_ptr");

  Type *LoadType = Ctx.B.getInt32Ty();
  if (*LoadWidthInDwords == 2)
    LoadType = Ctx.B.getInt64Ty();
  else if (*LoadWidthInDwords == 4)
    LoadType = FixedVectorType::get(Ctx.B.getInt32Ty(), *LoadWidthInDwords);
  Value *Loaded = Ctx.B.CreateAlignedLoad(
      LoadType, Pointer, DwordSmemAddressAlignment, "smem_load");
  if (*LoadWidthInDwords == 1) {
    Ctx.registers().writeReg32(*Destination, Loaded);
  } else if (*LoadWidthInDwords == 2) {
    Ctx.registers().writeReg64(*Destination, Loaded);
  } else {
    // B128 is represented as four i32 values written across an SGPR tuple.
    Ctx.registers().writeRegVec(*Destination, Loaded);
  }
  return Error::success();
}

} // namespace COMGR::hotswap
