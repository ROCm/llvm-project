//===- handle-mubuf.cpp - Transpiler --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/raiser/reg-file.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"

#include <cstdint>
#include <optional>

using namespace llvm;

namespace COMGR::transpiler {

namespace {

/// Payload and register interpretation of an unformatted buffer access.
struct BufferAccess {
  enum Flags { Store = 1, Signed = 2, LowHalf = 4, HighHalf = 8 };
  /// Bytes checked and transferred independently.
  unsigned ComponentBytes;
  /// Number of consecutive data VGPRs.
  unsigned Components;
  /// Load extension and half-register selection.
  unsigned DataFlags;
};

} // namespace

/// Return the per-component payload and extension required by the opcode.
static std::optional<BufferAccess> bufferAccess(CanonicalOp Opcode) {
  using Access = BufferAccess;
  switch (Opcode) {
  case CanonicalOp::BUFFER_LOAD_U8:
    return Access{1, 1, 0};
  case CanonicalOp::BUFFER_LOAD_I8:
    return Access{1, 1, Access::Signed};
  case CanonicalOp::BUFFER_LOAD_U16:
    return Access{2, 1, 0};
  case CanonicalOp::BUFFER_LOAD_I16:
    return Access{2, 1, Access::Signed};
  case CanonicalOp::BUFFER_LOAD_B32:
    return Access{4, 1, 0};
  case CanonicalOp::BUFFER_LOAD_B64:
    return Access{4, 2, 0};
  case CanonicalOp::BUFFER_LOAD_B96:
    return Access{4, 3, 0};
  case CanonicalOp::BUFFER_LOAD_B128:
    return Access{4, 4, 0};
  case CanonicalOp::BUFFER_LOAD_D16_U8:
    return Access{1, 1, Access::LowHalf};
  case CanonicalOp::BUFFER_LOAD_D16_I8:
    return Access{1, 1, Access::LowHalf | Access::Signed};
  case CanonicalOp::BUFFER_LOAD_D16_B16:
    return Access{2, 1, Access::LowHalf};
  case CanonicalOp::BUFFER_LOAD_D16_HI_U8:
    return Access{1, 1, Access::HighHalf};
  case CanonicalOp::BUFFER_LOAD_D16_HI_I8:
    return Access{1, 1, Access::HighHalf | Access::Signed};
  case CanonicalOp::BUFFER_LOAD_D16_HI_B16:
    return Access{2, 1, Access::HighHalf};
  case CanonicalOp::BUFFER_STORE_B8:
    return Access{1, 1, Access::Store};
  case CanonicalOp::BUFFER_STORE_B16:
    return Access{2, 1, Access::Store};
  case CanonicalOp::BUFFER_STORE_B32:
    return Access{4, 1, Access::Store};
  case CanonicalOp::BUFFER_STORE_B64:
    return Access{4, 2, Access::Store};
  case CanonicalOp::BUFFER_STORE_B96:
    return Access{4, 3, Access::Store};
  case CanonicalOp::BUFFER_STORE_B128:
    return Access{4, 4, Access::Store};
  case CanonicalOp::BUFFER_STORE_D16_HI_B8:
    return Access{1, 1, Access::Store | Access::HighHalf};
  case CanonicalOp::BUFFER_STORE_D16_HI_B16:
    return Access{2, 1, Access::Store | Access::HighHalf};
  default:
    return std::nullopt;
  }
}

/// Read a required operand index after checking the decoded operand count.
static Expected<unsigned> bufferOperandIndex(RaiseContext &Context,
                                             const DecodedInst &Instruction,
                                             AMDGPU::OpName Name) {
  int Index =
      COMGR::transpiler::getNamedOperandIdx(Instruction.Inst.getOpcode(), Name);
  if (Index < 0 || static_cast<unsigned>(Index) >= Instruction.numOperands()) {
    return unsupported(Context, Instruction,
                       "buffer access is missing a required named operand");
  }
  return Index;
}

/// Read a register operand of the required bank and width.
static Expected<ParsedReg> bufferRegister(RaiseContext &Context,
                                          const DecodedInst &Instruction,
                                          unsigned Index, ParsedReg::Kind Kind,
                                          unsigned Words) {
  if (Index >= Instruction.numOperands() || !Instruction.isReg(Index)) {
    return unsupported(Context, Instruction,
                       "buffer operand must be a register");
  }
  Expected<ParsedReg> Register =
      Context.registers().parseReg(Instruction, Index);
  if (!Register)
    return Register.takeError();
  if (Register->RegKind != Kind || Register->WidthInDwords != Words) {
    return unsupported(
        Context, Instruction,
        "buffer operand has an unsupported register bank or width");
  }
  return *Register;
}

Error handleMUBUF(RaiseContext &Context, const DecodedInst &Instruction) {
  std::optional<BufferAccess> Access = bufferAccess(Instruction.CanonOp);
  if (!Access) {
    return unsupported(Context, Instruction,
                       "unsupported buffer opcode or addressing form");
  }
  if (Context.Projection.SourceSTI.getCPU() != "gfx1250") {
    return unsupported(Context, Instruction,
                       "buffer source descriptor layout is not modeled");
  }
  StringRef Target = Context.Projection.TargetSTI.getCPU();
  if (Target != "gfx942" && Target != "gfx950" && Target != "gfx1250") {
    return unsupported(Context, Instruction,
                       "buffer target memory behavior is not modeled");
  }

  int VectorIndex = COMGR::transpiler::getNamedOperandIdx(
      Instruction.Inst.getOpcode(), AMDGPU::OpName::vaddr);
  int TiedIndex = COMGR::transpiler::getNamedOperandIdx(
      Instruction.Inst.getOpcode(), AMDGPU::OpName::vdata_in);
  unsigned OperandCount = 6 + (VectorIndex >= 0) + (TiedIndex >= 0);
  if (Instruction.numOperands() != OperandCount) {
    return unsupported(Context, Instruction,
                       "buffer operand count does not match its encoding");
  }

  Expected<unsigned> CacheIndex =
      bufferOperandIndex(Context, Instruction, AMDGPU::OpName::cpol);
  if (!CacheIndex)
    return CacheIndex.takeError();
  if (!Instruction.isImm(*CacheIndex)) {
    return unsupported(Context, Instruction,
                       "buffer cache policy must be an immediate");
  }
  if (Instruction.getImm(*CacheIndex) != 0) {
    return unsupported(Context, Instruction,
                       "non-default buffer cache policy is not modeled");
  }
  Expected<unsigned> SwizzleIndex =
      bufferOperandIndex(Context, Instruction, AMDGPU::OpName::swz);
  if (!SwizzleIndex)
    return SwizzleIndex.takeError();
  if (!Instruction.isImm(*SwizzleIndex) || Instruction.getImm(*SwizzleIndex)) {
    return unsupported(Context, Instruction,
                       "buffer cache swizzle is not modeled");
  }

  Expected<unsigned> ImmediateIndex =
      bufferOperandIndex(Context, Instruction, AMDGPU::OpName::offset);
  if (!ImmediateIndex)
    return ImmediateIndex.takeError();
  if (!Instruction.isImm(*ImmediateIndex) ||
      Instruction.getImm(*ImmediateIndex) < 0 ||
      Instruction.getImm(*ImmediateIndex) >= (1 << 23)) {
    return unsupported(Context, Instruction,
                       "buffer offset must be an unsigned 23-bit immediate");
  }
  Expected<unsigned> ResourceIndex =
      bufferOperandIndex(Context, Instruction, AMDGPU::OpName::srsrc);
  if (!ResourceIndex)
    return ResourceIndex.takeError();
  Expected<ParsedReg> Resource =
      bufferRegister(Context, Instruction, *ResourceIndex, ParsedReg::SGPR, 4);
  if (!Resource)
    return Resource.takeError();

  Expected<unsigned> ScalarIndex =
      bufferOperandIndex(Context, Instruction, AMDGPU::OpName::soffset);
  if (!ScalarIndex)
    return ScalarIndex.takeError();
  if (!Instruction.isReg(*ScalarIndex)) {
    return unsupported(Context, Instruction,
                       "VBUFFER scalar offset must be an SGPR, M0, or null");
  }
  Expected<ParsedReg> Scalar =
      Context.registers().parseReg(Instruction, *ScalarIndex);
  if (!Scalar)
    return Scalar.takeError();
  const bool NullOffset =
      stripRegEncoding(Instruction.getReg(*ScalarIndex)) == AMDGPU::SGPR_NULL;
  if ((Scalar->RegKind != ParsedReg::SGPR && Scalar->RegKind != ParsedReg::M0 &&
       !NullOffset) ||
      Scalar->WidthInDwords != 1) {
    return unsupported(Context, Instruction,
                       "VBUFFER scalar offset must be an SGPR, M0, or null");
  }

  Expected<unsigned> DataIndex =
      bufferOperandIndex(Context, Instruction, AMDGPU::OpName::vdata);
  if (!DataIndex)
    return DataIndex.takeError();
  Expected<ParsedReg> Data = bufferRegister(
      Context, Instruction, *DataIndex, ParsedReg::VGPR, Access->Components);
  if (!Data)
    return Data.takeError();
  if (TiedIndex >= 0 &&
      (static_cast<unsigned>(TiedIndex) >= Instruction.numOperands() ||
       !Instruction.isReg(TiedIndex) ||
       Instruction.getReg(TiedIndex) != Instruction.getReg(*DataIndex))) {
    return unsupported(Context, Instruction,
                       "D16 load requires a tied VGPR input");
  }

  IRBuilder<> &Builder = Context.B;
  AllocaRegFile &Registers = Context.registers().regFile();
  Expected<Value *> ScalarOffset =
      Context.registers().readOp32(Instruction, *ScalarIndex);
  if (!ScalarOffset)
    return ScalarOffset.takeError();
  Value *Offset = Builder.CreateZExt(*ScalarOffset, Builder.getInt64Ty());
  Offset = Builder.CreateAdd(
      Offset, Builder.getInt64(Instruction.getImm(*ImmediateIndex)));
  if (VectorIndex >= 0) {
    Expected<ParsedReg> Vector =
        bufferRegister(Context, Instruction, VectorIndex, ParsedReg::VGPR, 1);
    if (!Vector)
      return Vector.takeError();
    Value *VectorOffset = Builder.CreateZExt(
        Registers.readReg32(Builder, *Vector), Builder.getInt64Ty());
    Offset = Builder.CreateAdd(Offset, VectorOffset);
  }

  Value *Word0 = Context.registers().readSgpr32(*Resource->BaseIdx);
  Value *Word1 = Context.registers().readSgpr32(*Resource->BaseIdx + 1);
  Value *Word2 = Context.registers().readSgpr32(*Resource->BaseIdx + 2);
  Value *Word3 = Context.registers().readSgpr32(*Resource->BaseIdx + 3);
  Context.requireZeroBits(Word3, 0xc0000000, Instruction,
                          "buffer descriptor type must be provably zero");
  Context.requireZeroBits(
      Word3, 0x00000fc0, Instruction,
      "buffer descriptor reserved bits must be provably zero");
  Context.requireZeroBits(Word3, 0x10000000, Instruction,
                          "swizzled buffer descriptors are not modeled");
  Context.requireZeroBits(
      Word3, 0x0ffff000, Instruction,
      "buffer stride and stride scale must be provably zero");
  Context.requireZeroBits(Word3, 0x20000000, Instruction,
                          "structured buffer bounds are not modeled");

  // The base occupies bits 56:0; NUM_RECORDS spans bits 101:57.
  Value *BaseHigh = Builder.CreateAnd(Word1, Builder.getInt32((1u << 25) - 1));
  Value *Base = Builder.CreateOr(
      Builder.CreateZExt(Word0, Builder.getInt64Ty()),
      Builder.CreateShl(Builder.CreateZExt(BaseHigh, Builder.getInt64Ty()),
                        32));
  Value *Extent = Builder.CreateOr(
      Builder.CreateZExt(Builder.CreateLShr(Word1, 25), Builder.getInt64Ty()),
      Builder.CreateShl(Builder.CreateZExt(Word2, Builder.getInt64Ty()), 7));
  Value *ExtentHigh = Builder.CreateAnd(Word3, Builder.getInt32(63));
  Extent = Builder.CreateOr(
      Extent, Builder.CreateShl(
                  Builder.CreateZExt(ExtentHigh, Builder.getInt64Ty()), 39));
  Value *Unbounded =
      Builder.CreateICmpEQ(Extent, Builder.getInt64((uint64_t(1) << 45) - 1));

  const bool IsStore = Access->DataFlags & BufferAccess::Store;
  const bool IsSigned = Access->DataFlags & BufferAccess::Signed;
  const bool HighHalf = Access->DataFlags & BufferAccess::HighHalf;
  const bool IsD16 =
      Access->DataFlags & (BufferAccess::LowHalf | BufferAccess::HighHalf);
  Value *Preserved = nullptr;
  if (!IsStore && IsD16) {
    Value *Previous = Registers.readReg32(Builder, *Data);
    uint32_t Mask = HighHalf ? 0x0000ffff : 0xffff0000;
    if (!Context.sourceSramEcc()) {
      Context.requireZeroBits(Previous, Mask, Instruction,
                              "D16 load requires a source SRAM ECC setting or "
                              "a zero untouched half");
    }
    Preserved = Context.sourceSramEcc() == true
                    ? Builder.getInt32(0)
                    : Builder.CreateAnd(Previous, Builder.getInt32(Mask));
  }

  Offset = Context.freezeMemAddr(Offset);
  Base = Context.freezeMemAddr(Base);
  Context.registers().emitUnderExec([&] {
    for (unsigned I = 0; I != Access->Components; ++I) {
      Value *ComponentOffset = Builder.CreateAdd(
          Offset, Builder.getInt64(I * Access->ComponentBytes));
      Value *End = Builder.CreateAdd(ComponentOffset,
                                     Builder.getInt64(Access->ComponentBytes));
      // gfx1250 over-clamps the equality case, including each dword of a
      // multi-dword access. The all-ones extent disables the check.
      Value *InBounds =
          Builder.CreateOr(Unbounded, Builder.CreateICmpULT(End, Extent));
      BasicBlock *Before = Builder.GetInsertBlock();
      Function *Kernel = Before->getParent();
      BasicBlock *AccessBlock =
          BasicBlock::Create(Builder.getContext(), "buffer_access", Kernel);
      BasicBlock *Continue =
          BasicBlock::Create(Builder.getContext(), "buffer_continue", Kernel);
      Builder.CreateCondBr(InBounds, AccessBlock, Continue);
      Builder.SetInsertPoint(AccessBlock);

      Value *Address = Builder.CreateAdd(Base, ComponentOffset);
      Value *Pointer = Builder.CreateIntToPtr(
          Address,
          PointerType::get(Builder.getContext(), AMDGPUAS::GLOBAL_ADDRESS));
      Type *PayloadType = Builder.getIntNTy(8 * Access->ComponentBytes);
      ParsedReg Component = *Data;
      Component.BaseIdx = *Data->BaseIdx + I;
      Component.WidthInDwords = 1;
      Value *Loaded = nullptr;
      if (IsStore) {
        Value *Stored = Registers.readReg32(Builder, Component);
        if (HighHalf)
          Stored = Builder.CreateLShr(Stored, 16);
        Stored = Builder.CreateTruncOrBitCast(Stored, PayloadType);
        Builder.CreateAlignedStore(Stored, Pointer, Align(1));
      } else {
        Loaded = Builder.CreateAlignedLoad(PayloadType, Pointer, Align(1));
      }
      Builder.CreateBr(Continue);
      Builder.SetInsertPoint(Continue);
      if (IsStore)
        continue;

      PHINode *Result = Builder.CreatePHI(PayloadType, 2);
      Result->addIncoming(Constant::getNullValue(PayloadType), Before);
      Result->addIncoming(Loaded, AccessBlock);
      Type *ExtendedType = IsD16 ? Builder.getInt16Ty() : Builder.getInt32Ty();
      Value *Extended = IsSigned
                            ? Builder.CreateSExtOrBitCast(Result, ExtendedType)
                            : Builder.CreateZExtOrBitCast(Result, ExtendedType);
      if (IsD16) {
        Extended = Builder.CreateZExt(Extended, Builder.getInt32Ty());
        if (HighHalf)
          Extended = Builder.CreateShl(Extended, 16);
        Extended = Builder.CreateOr(Extended, Preserved);
      }
      Registers.writeReg32(Builder, Component, Extended);
    }
  });
  return Error::success();
}

} // namespace COMGR::transpiler
