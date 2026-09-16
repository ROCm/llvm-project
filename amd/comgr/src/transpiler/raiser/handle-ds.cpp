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

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <cassert>
#include <cstdint>

using namespace llvm;

namespace COMGR::transpiler {

/// Return the index of a required named operand of a direct DS load.
static unsigned dsOperandIndex(const DecodedInst &Instruction,
                               AMDGPU::OpName Name) {
  const int Index =
      COMGR::transpiler::getNamedOperandIdx(Instruction.Inst.getOpcode(), Name);
  assert(Index >= 0 &&
         static_cast<unsigned>(Index) < Instruction.numOperands() &&
         "direct DS load is missing a required operand");
  return Index;
}

Error handleDS(RaiseContext &Context, const DecodedInst &Instruction) {
  unsigned WidthInDwords;
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
  default:
    return unsupported(Context, Instruction, "unsupported DS operation");
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
