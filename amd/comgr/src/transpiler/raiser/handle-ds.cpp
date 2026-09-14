//===- handle-ds.cpp - Transpiler ---------------------------------===//
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
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/reg-file.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <cassert>
#include <cstdint>

using namespace llvm;

namespace COMGR::transpiler {

/// Return the index of a required named operand of a single-address DS load.
static unsigned dsOperandIndex(const DecodedInst &Instruction,
                               AMDGPU::OpName Name) {
  const int Index =
      COMGR::transpiler::getNamedOperandIdx(Instruction.Inst.getOpcode(), Name);
  assert(Index >= 0 &&
         static_cast<unsigned>(Index) < Instruction.numOperands() &&
         "DS load is missing a required operand");
  return Index;
}

/// Gather eight elements per lane from LDS and pack them into the destination.
static void emitTransposedDSLoad(RaiseContext &Context, ParsedReg Destination,
                                 Value *ByteAddress, unsigned ElementBits) {
  IRBuilder<> &B = Context.B;
  Value *Lane = Context.emitLaneIdx();
  Value *ElementOffset = B.CreateMul(B.CreateAnd(Lane, B.getInt32(7)),
                                     B.getInt32(ElementBits / 8));
  Value *SourceBase =
      B.CreateAnd(Lane, B.getInt32(ElementBits == 8 ? ~15 : ~7));
  if (ElementBits == 8) {
    Value *Half = B.CreateAnd(B.CreateLShr(Lane, 1), B.getInt32(4));
    SourceBase = B.CreateOr(SourceBase, Half);
  }

  // A nonzero source EXEC makes every lane generate an address and write back.
  Context.registers().emitWithNonzeroExec([&] {
    const unsigned ElementsPerDword = 32 / ElementBits;
    Type *ResultType =
        FixedVectorType::get(B.getInt32Ty(), Destination.WidthInDwords);
    Value *Result = PoisonValue::get(ResultType);
    for (unsigned I = 0; I < Destination.WidthInDwords; ++I) {
      Value *Word = B.getInt32(0);
      for (unsigned J = 0; J < ElementsPerDword; ++J) {
        const unsigned SourceOffset = I * (ElementBits == 8 ? 8 : 2) + J;
        Value *SourceLane = B.CreateAdd(SourceBase, B.getInt32(SourceOffset));
        Value *Index = B.CreateMul(SourceLane, B.getInt32(4));
        Value *Address = B.CreateIntrinsic(Intrinsic::amdgcn_ds_bpermute, {},
                                           {Index, ByteAddress});
        Address = B.CreateAdd(Address, ElementOffset);
        Value *Pointer = B.CreateIntToPtr(
            Address, PointerType::get(B.getContext(), AMDGPUAS::LOCAL_ADDRESS));
        Value *Element =
            B.CreateAlignedLoad(B.getIntNTy(ElementBits), Pointer, Align(1));
        Element = B.CreateZExt(Element, B.getInt32Ty());
        Word = B.CreateOr(Word, B.CreateShl(Element, J * ElementBits));
      }
      Result = B.CreateInsertElement(Result, Word, I);
    }
    Context.registers().regFile().writeRegVec(B, Destination, Result);
  });
}

Error handleDS(RaiseContext &Context, const DecodedInst &Instruction) {
  unsigned WidthInDwords;
  unsigned TransposeElementBits = 0;
  switch (Instruction.CanonOp) {
  case CanonicalOp::DS_LOAD_B32:
    WidthInDwords = 1;
    break;
  case CanonicalOp::DS_LOAD_B64:
    WidthInDwords = 2;
    break;
  case CanonicalOp::DS_LOAD_B128:
    WidthInDwords = 4;
    break;
  case CanonicalOp::DS_LOAD_TR8_B64:
    WidthInDwords = 2;
    TransposeElementBits = 8;
    break;
  case CanonicalOp::DS_LOAD_TR16_B128:
    WidthInDwords = 4;
    TransposeElementBits = 16;
    break;
  default:
    return unsupported(Context, Instruction, "unsupported DS operation");
  }

  if (TransposeElementBits &&
      (!AMDGPU::isGFX1250(Context.Projection.SourceSTI) ||
       Context.Projection.sourceWaveSize() != 32)) {
    return unsupported(Context, Instruction,
                       "DS transpose loads require a gfx1250 wave32 source");
  }

  if (AMDGPU::getIsaVersion(Context.MC.SubtargetInfo->getCPU()).Major < 9) {
    return unsupported(Context, Instruction,
                       "M0-bounded LDS loads are not modeled");
  }
  // Source address alignment and CU mode are not established, so refuse wide
  // loads on hardware affected by the WGP misalignment bug.
  if (WidthInDwords > 1 &&
      Context.MC.SubtargetInfo->hasFeature(AMDGPU::FeatureLDSMisalignedBug)) {
    return unsupported(Context, Instruction,
                       "wide LDS loads with the WGP misalignment bug are not "
                       "modeled");
  }

  const unsigned GDSIndex = dsOperandIndex(Instruction, AMDGPU::OpName::gds);
  assert(Instruction.isImm(GDSIndex) && "GDS operand must be an immediate");
  if (Instruction.getImm(GDSIndex)) {
    return unsupported(Context, Instruction, "GDS loads are not modeled");
  }

  Expected<ParsedReg> Destination = Context.registers().parseReg(
      Instruction, dsOperandIndex(Instruction, AMDGPU::OpName::vdst));
  if (!Destination) {
    return Destination.takeError();
  }
  if (Destination->RegKind != ParsedReg::VGPR) {
    return unsupported(Context, Instruction,
                       "DS loads require a VGPR destination");
  }
  assert(Destination->WidthInDwords == WidthInDwords &&
         "DS destination width does not match the load");

  Expected<Value *> Address = Context.registers().readOp32(
      Instruction, dsOperandIndex(Instruction, AMDGPU::OpName::addr));
  if (!Address) {
    return Address.takeError();
  }
  const unsigned OffsetIndex =
      dsOperandIndex(Instruction, AMDGPU::OpName::offset);
  assert(Instruction.isImm(OffsetIndex) && "DS offset must be an immediate");
  const int64_t OffsetInBytes = Instruction.getImm(OffsetIndex);
  assert(OffsetInBytes >= 0 && OffsetInBytes <= UINT16_MAX &&
         "DS offset must fit its unsigned 16-bit field");
  Value *ByteAddress = Context.B.CreateAdd(
      *Address, Context.B.getInt32(OffsetInBytes), "lds_address");
  ByteAddress = Context.freezeMemAddr(ByteAddress);

  if (TransposeElementBits) {
    emitTransposedDSLoad(Context, *Destination, ByteAddress,
                         TransposeElementBits);
    return Error::success();
  }

  // Inactive lanes can hold invalid addresses, so guard the load itself.
  Context.registers().emitUnderExec([&] {
    Value *Pointer = Context.B.CreateIntToPtr(
        ByteAddress,
        PointerType::get(Context.B.getContext(), AMDGPUAS::LOCAL_ADDRESS),
        "lds_ptr");
    Type *LoadType = Context.B.getInt32Ty();
    if (WidthInDwords == 2) {
      LoadType = Context.B.getInt64Ty();
    } else if (WidthInDwords == 4) {
      LoadType = FixedVectorType::get(Context.B.getInt32Ty(), WidthInDwords);
    }

    // AMDHSA uses unaligned access mode; the opcode width implies no alignment.
    Value *Loaded =
        Context.B.CreateAlignedLoad(LoadType, Pointer, Align(1), "lds_load");
    AllocaRegFile &Registers = Context.registers().regFile();
    if (WidthInDwords == 1) {
      Registers.writeReg32(Context.B, *Destination, Loaded);
    } else if (WidthInDwords == 2) {
      Registers.writeReg64(Context.B, *Destination, Loaded);
    } else {
      Registers.writeRegVec(Context.B, *Destination, Loaded);
    }
  });
  return Error::success();
}

} // namespace COMGR::transpiler
