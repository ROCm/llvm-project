//===- flat-addr.cpp - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/flat-addr.h"

#include "SIDefines.h"

#include "hotswap/decoder/amdgpu-mc-tables.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/raiser/handlers.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <optional>

using namespace llvm;

namespace COMGR::hotswap {

// Index of the operand named `Name`, or nothing when the instruction has no
// operand of that name. The GLOBAL addressing forms differ in exactly which
// of these operands the encoding carries.
static std::optional<unsigned> globalOperandIndex(const DecodedInst &Di,
                                                  AMDGPU::OpName Name) {
  int Index = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Index < 0)
    return std::nullopt;
  assert(static_cast<unsigned>(Index) < Di.numOperands() &&
         "named operand index is past the end of the instruction");
  return static_cast<unsigned>(Index);
}

// Index of an operand every GLOBAL memory instruction carries.
static unsigned requiredGlobalOperandIndex(const DecodedInst &Di,
                                           AMDGPU::OpName Name) {
  std::optional<unsigned> Index = globalOperandIndex(Di, Name);
  assert(Index && "GLOBAL memory instruction is missing a required operand");
  return *Index;
}

Expected<Value *> emitGlobalAddress(RaiseContext &Ctx, const DecodedInst &Di,
                                    Align AccessAlign) {
  unsigned CachePolicyIndex =
      requiredGlobalOperandIndex(Di, AMDGPU::OpName::cpol);
  assert(Di.isImm(CachePolicyIndex) && "operand 'cpol' is not an immediate");
  int64_t CachePolicy = Di.getImm(CachePolicyIndex);
  // Address scaling shares the cache-policy field but is an addressing mode,
  // and one that changes what the per-lane offset below means.
  if (CachePolicy & AMDGPU::CPol::SCAL) {
    return unsupported(Ctx, Di,
                       "scaling the per-lane offset by the access size is not "
                       "modeled");
  }
  if (CachePolicy != 0)
    return unsupported(Ctx, Di, "non-default cache policy is not modeled");

  // MC surfaces the immediate offset as the raw encoded field, so sign-extend
  // it from the width the source ISA encodes it in.
  unsigned OffsetIndex = requiredGlobalOperandIndex(Di, AMDGPU::OpName::offset);
  assert(Di.isImm(OffsetIndex) && "operand 'offset' is not an immediate");
  int64_t Offset = SignExtend64(static_cast<uint64_t>(Di.getImm(OffsetIndex)),
                                Ctx.Projection.sourceIsa().flatOffsetBits());
  // The access is modeled as naturally aligned, which holds for the base
  // address of a well-formed program but is the caller's assumption to make.
  // An immediate offset that does not preserve it would make that model a lie.
  if (!isAligned(AccessAlign, static_cast<uint64_t>(Offset)))
    return unsupported(Ctx, Di,
                       "immediate offset does not preserve the alignment of "
                       "the access");

  unsigned LaneAddressIndex =
      requiredGlobalOperandIndex(Di, AMDGPU::OpName::vaddr);
  std::optional<unsigned> ScalarBaseIndex =
      globalOperandIndex(Di, AMDGPU::OpName::saddr);

  Value *Address = nullptr;
  if (ScalarBaseIndex) {
    Expected<Value *> ScalarBase =
        Ctx.registers().readOp64(Di, *ScalarBaseIndex);
    if (!ScalarBase)
      return ScalarBase.takeError();
    Expected<Value *> LaneOffset =
        Ctx.registers().readOp32(Di, LaneAddressIndex);
    if (!LaneOffset)
      return LaneOffset.takeError();
    Type *I64Ty = Ctx.B.getInt64Ty();
    Value *WideLaneOffset =
        Ctx.Projection.sourceIsa().hasSignedGlobalLaneOffset()
            ? Ctx.B.CreateSExt(*LaneOffset, I64Ty, "global_lane_offset")
            : Ctx.B.CreateZExt(*LaneOffset, I64Ty, "global_lane_offset");
    Address = Ctx.B.CreateAdd(*ScalarBase, WideLaneOffset, "global_addr");
  } else {
    Expected<Value *> LaneAddress =
        Ctx.registers().readOp64(Di, LaneAddressIndex);
    if (!LaneAddress)
      return LaneAddress.takeError();
    Address = *LaneAddress;
  }

  Value *FrozenAddress = Ctx.freezeMemAddr(Address);
  PointerType *PointerTy =
      PointerType::get(Ctx.B.getContext(), AMDGPUAS::GLOBAL_ADDRESS);
  Value *Pointer = Ctx.B.CreateIntToPtr(FrozenAddress, PointerTy, "global_ptr");

  if (Offset == 0)
    return Pointer;
  // The offset is free to leave the object the base address points into, so
  // the displacement is not `inbounds`.
  return Ctx.B.CreateGEP(Ctx.B.getInt8Ty(), Pointer, Ctx.B.getInt64(Offset),
                         "global_offset_ptr");
}

} // namespace COMGR::hotswap
