//===- handle-sopk.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>

using namespace llvm;

namespace COMGR::hotswap {
namespace {

enum class HardwareRegisterReadAction {
  ReturnZero, // Replace the read with zero.
  Reject      // Reject the instruction.
};

enum class HardwareRegisterWriteAction {
  Ignore, // Drop the write.
  Emit,   // Preserve the write with an intrinsic.
  Reject  // Reject the instruction.
};

// Read and write policies for one hardware-register identifier.
struct HardwareRegisterPolicy {
  HardwareRegisterReadAction Read;
  HardwareRegisterWriteAction Write;
};

// Return the conservative policy for hardware-register identifier Id.
// Hardware-register numbers are reused between ISA generations. Where a number
// denotes a load-bearing register on any supported source ISA, refuse the
// access instead of treating it as diagnostic state.
HardwareRegisterPolicy classifyHardwareRegister(unsigned Id) {
  using namespace AMDGPU::Hwreg;
  switch (Id) {
  case ID_MODE:
    // MODE is initialized from the kernel descriptor and can be changed by
    // preceding SETREG instructions. Until that state is tracked, returning a
    // fabricated value would silently change source-visible behavior.
    return {HardwareRegisterReadAction::Reject,
            HardwareRegisterWriteAction::Emit};

  case ID_MEM_BASES:
  case ID_FLAT_SCR_LO:
  case ID_FLAT_SCR_HI:
  case ID_XNACK_MASK:
  case ID_XNACK_STATE_PRIV:
  case ID_XNACK_MASK_gfx1250:
    return {HardwareRegisterReadAction::Reject,
            HardwareRegisterWriteAction::Reject};

  case ID_TBA_LO:
  case ID_TBA_HI:
  case ID_TMA_LO:
  case ID_TMA_HI:
    return {HardwareRegisterReadAction::ReturnZero,
            HardwareRegisterWriteAction::Reject};

  case ID_STATUS:
  case ID_TRAPSTS:
  case ID_HW_ID:
  case ID_GPR_ALLOC:
  case ID_LDS_ALLOC:
  case ID_IB_STS:
  case ID_PERF_SNAPSHOT_DATA_gfx12:
  case ID_PERF_SNAPSHOT_PC_LO_gfx12:
  case ID_PERF_SNAPSHOT_PC_HI_gfx12:
  case ID_HW_ID1:
  case ID_HW_ID2:
  case ID_POPS_PACKER:
  case ID_SCHED_MODE:
  case ID_PERF_SNAPSHOT_DATA_gfx11:
  case ID_IB_STS2:
  case ID_SHADER_CYCLES:
  case ID_SHADER_CYCLES_HI:
  case ID_DVGPR_ALLOC_LO:
  case ID_DVGPR_ALLOC_HI:
    return {HardwareRegisterReadAction::ReturnZero,
            HardwareRegisterWriteAction::Ignore};

  default:
    return {HardwareRegisterReadAction::Reject,
            HardwareRegisterWriteAction::Reject};
  }
}

// Read the hardware-register selector from Di.
Expected<unsigned> readHardwareRegisterSelector(RaiseContext &Ctx,
                                                const DecodedInst &Di) {
  unsigned SelectorOperand = 1;
  if (Di.numOperands() <= SelectorOperand || !Di.isImm(SelectorOperand))
    return unsupportedInstruction(
        Ctx, Di, "hardware-register selector is not immediate operand 1");
  return static_cast<unsigned>(Di.getImm(SelectorOperand)) & UINT16_MAX;
}

// Return whether the bit range overlaps MODE.VGPR_MSB.
bool overlapsVgprMsb(unsigned BitOffset, unsigned BitWidth) {
  uint32_t FieldMask = llvm::maskTrailingOnes<uint32_t>(BitWidth) << BitOffset;
  return FieldMask & AMDGPU::Hwreg::VGPR_MSB_MASK;
}

constexpr unsigned VgprMsbFieldWidth =
    llvm::popcount(static_cast<unsigned>(AMDGPU::Hwreg::DST_VGPR_MSB));
constexpr unsigned VgprMsbLowBit = llvm::countr_zero_constexpr<unsigned>(
    static_cast<unsigned>(AMDGPU::Hwreg::VGPR_MSB_MASK));
constexpr unsigned VgprMsbHighBit =
    std::numeric_limits<unsigned>::digits -
    llvm::countl_zero_constexpr<unsigned>(
        static_cast<unsigned>(AMDGPU::Hwreg::VGPR_MSB_MASK)) -
    1;

uint8_t getVgprMsbField(uint64_t Value, uint32_t Mask, unsigned Slot) {
  unsigned Shift = llvm::countr_zero(Mask);
  return static_cast<uint8_t>(((Value & Mask) >> Shift)
                              << (Slot * VgprMsbFieldWidth));
}

// Update the tracked VGPR_MSB state from an immediate MODE value.
void updateImmediateModeVgprMsb(RaiseContext &Ctx, uint64_t Value) {
  uint8_t Encoding = getVgprMsbField(Value, AMDGPU::Hwreg::SRC0_VGPR_MSB, 0) |
                     getVgprMsbField(Value, AMDGPU::Hwreg::SRC1_VGPR_MSB, 1) |
                     getVgprMsbField(Value, AMDGPU::Hwreg::SRC2_VGPR_MSB, 2) |
                     getVgprMsbField(Value, AMDGPU::Hwreg::DST_VGPR_MSB, 3);
  Ctx.registers().setVgprMsBs(Encoding);
}

// Raise a hardware-register read according to Policy.
Error handleGetreg(RaiseContext &Ctx, const DecodedInst &Di,
                   OperandResolver &Op, unsigned Id,
                   HardwareRegisterPolicy Policy) {
  if (Policy.Read == HardwareRegisterReadAction::Reject)
    return unsupportedInstruction(
        Ctx, Di,
        Twine("cannot reproduce hardware-register read for id ") + Twine(Id));

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Ctx.registers().writeReg32(*Dst, Ctx.B.getInt32(0));
  return Error::success();
}

// Raise a hardware-register write according to Policy.
Error handleSetreg(RaiseContext &Ctx, const DecodedInst &Di,
                   OperandResolver &Op, unsigned Selector, unsigned Id,
                   unsigned BitOffset, unsigned BitWidth,
                   HardwareRegisterPolicy Policy) {
  if (Policy.Write == HardwareRegisterWriteAction::Reject)
    return unsupportedInstruction(
        Ctx, Di,
        Twine("cannot reproduce hardware-register write for id ") + Twine(Id));
  if (Policy.Write == HardwareRegisterWriteAction::Ignore)
    return Error::success();

  bool HasExtendedVgprs =
      Ctx.Projection.SourceSTI.hasFeature(AMDGPU::Feature1024AddressableVGPRs);
  Value *ValueArg;
  if (Di.CanonOp == CanonicalOp::S_SETREG_IMM32_B32) {
    uint64_t Value = static_cast<uint64_t>(Di.getImm(0));
    ValueArg = Ctx.B.getInt32(Value);
    if (HasExtendedVgprs && Id == AMDGPU::Hwreg::ID_MODE)
      updateImmediateModeVgprMsb(Ctx, Value);
  } else {
    if (HasExtendedVgprs && Id == AMDGPU::Hwreg::ID_MODE &&
        overlapsVgprMsb(BitOffset, BitWidth))
      return unsupportedInstruction(
          Ctx, Di,
          Twine("dynamic MODE write overlaps VGPR_MSB bits [") +
              Twine(VgprMsbLowBit) + ":" + Twine(VgprMsbHighBit) + "]");
    Expected<Value *> Source = Op.src(0);
    if (!Source)
      return Source.takeError();
    ValueArg = *Source;
  }

  Module *M = Ctx.B.GetInsertBlock()->getModule();
  Function *Setreg =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_setreg);
  Ctx.B.CreateCall(Setreg, {Ctx.B.getInt32(Selector), ValueArg});
  return Error::success();
}

} // namespace

// Raise a hardware-register SOPK instruction.
Error handleSOPK(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::S_GETREG_B32:
  case CanonicalOp::S_SETREG_B32:
  case CanonicalOp::S_SETREG_IMM32_B32:
    break;
  default:
    return unsupportedInstruction(Ctx, Di);
  }

  Expected<unsigned> Selector = readHardwareRegisterSelector(Ctx, Di);
  if (!Selector)
    return Selector.takeError();
  auto [Id, BitOffset, BitWidth] =
      AMDGPU::Hwreg::HwregEncoding::decode(*Selector);
  HardwareRegisterPolicy Policy = classifyHardwareRegister(Id);

  if (Di.CanonOp == CanonicalOp::S_GETREG_B32)
    return handleGetreg(Ctx, Di, Op, Id, Policy);
  return handleSetreg(Ctx, Di, Op, *Selector, Id, BitOffset, BitWidth, Policy);
}

} // namespace COMGR::hotswap
