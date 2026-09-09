//===- handle-smem.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "SIDefines.h"

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

// Supported S_LOAD_B* instructions ignore bits [1:0] of every address
// component.
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
static std::optional<unsigned> scalarLoadWidthInDwords(CanonicalInst Inst) {
  switch (Inst.Op) {
  case CanonicalOp::S_LOAD_B32:
  case CanonicalOp::S_LOAD_B64:
  case CanonicalOp::S_LOAD_B96:
  case CanonicalOp::S_LOAD_B128:
  case CanonicalOp::S_LOAD_B256:
  case CanonicalOp::S_LOAD_B512:
    break;
  default:
    return std::nullopt;
  }
  return canonicalTypeBitWidth(Inst.Type) / 32;
}

Error handleSMEM(RaiseContext &Ctx, const DecodedInst &Di, OperandResolver &) {
  std::optional<unsigned> LoadWidthInDwords = scalarLoadWidthInDwords(Di.Canon);
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
  if (!Di.isImm(CachePolicyIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'cpol' is not an immediate");
  // SCALE_OFFSET is encoded in the cache-policy field but changes the address:
  // it makes the SGPR offset an element index scaled by the load size. The
  // scale is handled below; other cache-policy modifiers remain unsupported.
  int64_t CachePolicy = Di.getImm(CachePolicyIndex);
  bool ScaleScalarOffset = (CachePolicy & AMDGPU::CPol::SCAL) != 0;
  if (CachePolicy & ~static_cast<int64_t>(AMDGPU::CPol::SCAL))
    return unsupported(Ctx, Di,
                       "scalar load cache-policy modifiers other than "
                       "SCALE_OFFSET are not supported");

  // The immediate and SGPR offsets are separate operands that add together.
  // An encoding may carry either or both, but not neither.
  if (!OffsetIndex && !ScalarOffsetIndex)
    invalidOperandLayout(Ctx.MC, Di, "scalar load has no offset operand");
  int64_t ImmediateOffset = 0;
  if (OffsetIndex) {
    if (!Di.isImm(*OffsetIndex))
      invalidOperandLayout(Ctx.MC, Di, "operand 'offset' is not an immediate");
    ImmediateOffset = Di.getImm(*OffsetIndex);
    if (ImmediateOffset < 0)
      return unsupported(Ctx, Di,
                         "negative scalar load offsets are not supported");
  }

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

  Value *ScalarOffsetValue = nullptr;
  if (ScalarOffsetIndex) {
    Expected<Value *> Read = Ctx.registers().readOp32(Di, *ScalarOffsetIndex);
    if (!Read)
      return Read.takeError();
    ScalarOffsetValue = *Read;
  }

  Type *I64Ty = Ctx.B.getInt64Ty();
  uint64_t AddressMask =
      maskTrailingZeros<uint64_t>(Log2(DwordSmemAddressAlignment));
  Value *AlignedBase = Ctx.B.CreateAnd(
      *BaseValue, ConstantInt::get(I64Ty, AddressMask), "smem_base");
  uint64_t Offset = alignDown(static_cast<uint64_t>(ImmediateOffset),
                              DwordSmemAddressAlignment.value());
  Value *Address = Ctx.B.CreateAdd(AlignedBase, ConstantInt::get(I64Ty, Offset),
                                   "smem_addr");
  if (ScalarOffsetValue) {
    // SCALE_OFFSET treats the 32-bit SGPR value as an element index. Widen it
    // first, then scale it by the load size.
    Value *WideScalarOffset =
        Ctx.B.CreateZExt(ScalarOffsetValue, I64Ty, "smem_soff");
    if (ScaleScalarOffset) {
      uint64_t ScaleFactor =
          *LoadWidthInDwords * DwordSmemAddressAlignment.value();
      WideScalarOffset = Ctx.B.CreateMul(WideScalarOffset,
                                         ConstantInt::get(I64Ty, ScaleFactor),
                                         "smem_soff_scaled");
    }
    // The ISA aligns the scaled offset, not the raw register, so the mask
    // follows the multiply.
    Value *AlignedScalarOffset =
        Ctx.B.CreateAnd(WideScalarOffset, ConstantInt::get(I64Ty, AddressMask),
                        "smem_soff_dword");
    Address = Ctx.B.CreateAdd(Address, AlignedScalarOffset, "smem_addr_soff");
  }
  PointerType *PointerTy =
      PointerType::get(Ctx.B.getContext(), AMDGPUAS::GLOBAL_ADDRESS);
  Value *Pointer = Ctx.B.CreateIntToPtr(Address, PointerTy, "smem_ptr");

  Type *LoadType = Ctx.B.getInt32Ty();
  if (*LoadWidthInDwords == 2)
    LoadType = Ctx.B.getInt64Ty();
  else if (*LoadWidthInDwords > 2)
    LoadType = FixedVectorType::get(Ctx.B.getInt32Ty(), *LoadWidthInDwords);
  Value *Loaded = Ctx.B.CreateAlignedLoad(
      LoadType, Pointer, DwordSmemAddressAlignment, "smem_load");
  if (*LoadWidthInDwords == 1) {
    Ctx.registers().writeReg32(*Destination, Loaded);
  } else if (*LoadWidthInDwords == 2) {
    Ctx.registers().writeReg64(*Destination, Loaded);
  } else {
    // Every wider load is a vector of i32 written across an SGPR tuple.
    Ctx.registers().writeRegVec(*Destination, Loaded);
  }
  return Error::success();
}

} // namespace COMGR::hotswap
