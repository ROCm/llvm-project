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

#include "SIDefines.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>

using namespace llvm;

namespace COMGR::hotswap {
namespace {

// Policy for reading a hardware register.
enum class HwregRead { Zero, Abort };

// Policy for writing a hardware register.
enum class HwregWrite { Drop, Preserve, Abort };

// Read and write policies for one hardware-register identifier.
struct HwregPolicy {
  HwregRead Read;
  HwregWrite Write;
};

// Return the conservative policy for hardware-register identifier Id.
// HWREG numbers are reused between ISA generations. Where a number denotes a
// load-bearing register on any supported source ISA, use the conservative
// policy: refusing is preferable to treating an aperture or retry-control
// register as diagnostic state.
HwregPolicy classifyHwreg(unsigned Id) {
  using namespace AMDGPU::Hwreg;
  switch (Id) {
  case ID_MODE:
    return {HwregRead::Zero, HwregWrite::Preserve};

  case ID_MEM_BASES:
  case ID_FLAT_SCR_LO:
  case ID_FLAT_SCR_HI:
  case ID_XNACK_MASK:
  case ID_XNACK_STATE_PRIV:
  case ID_XNACK_MASK_gfx1250:
    return {HwregRead::Abort, HwregWrite::Abort};

  case ID_TBA_LO:
  case ID_TBA_HI:
  case ID_TMA_LO:
  case ID_TMA_HI:
    return {HwregRead::Zero, HwregWrite::Abort};

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
    return {HwregRead::Zero, HwregWrite::Drop};

  default:
    return {HwregRead::Abort, HwregWrite::Abort};
  }
}

// Return an unsupported-instruction failure for Di with optional Detail.
Error unsupported(RaiseContext &Ctx, const DecodedInst &Di,
                  const Twine &Detail = {}) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags), Detail);
}

// Read and validate the 16-bit hardware-register selector from Di.
Expected<unsigned> readHwregSelector(RaiseContext &Ctx, const DecodedInst &Di) {
  constexpr unsigned SelectorOperand = 1;
  if (Di.numOperands() <= SelectorOperand || !Di.isImm(SelectorOperand))
    return unsupported(Ctx, Di,
                       "hardware-register selector is not immediate operand 1");
  int64_t Raw = Di.getImm(SelectorOperand);
  if (Raw < INT16_MIN || Raw > UINT16_MAX)
    return unsupported(Ctx, Di,
                       "hardware-register selector is outside 16 bits");
  return static_cast<unsigned>(Raw) & UINT16_MAX;
}

// Return the hardware-register identifier encoded in Selector.
unsigned hwregId(unsigned Selector) { return Selector & 0x3fu; }

// A bit field selected within a hardware register.
struct HwregField {
  unsigned Offset;
  unsigned Size;
};

// Decode the offset and size encoded in Selector.
HwregField decodeHwregField(unsigned Selector) {
  return {(Selector >> 6) & 0x1fu, ((Selector >> 11) & 0x1fu) + 1u};
}

// Return whether Field overlaps MODE.VGPR_MSB.
bool overlapsVgprMsb(HwregField Field) {
  constexpr unsigned VgprMsbLow = 12;
  constexpr unsigned VgprMsbHigh = 19;
  return Field.Offset <= VgprMsbHigh &&
         Field.Offset + Field.Size - 1 >= VgprMsbLow;
}

// Update the tracked VGPR_MSB state from an immediate MODE value.
void updateImmediateModeVgprMsb(RaiseContext &Ctx, uint64_t Value) {
  constexpr unsigned VgprMsbLow = 12;
  uint8_t ModeEncoding = static_cast<uint8_t>((Value >> VgprMsbLow) & 0xffu);
  // MODE stores the slots as (dst, src0, src1, src2), while RegisterState
  // stores them as (src0, src1, src2, dst).
  Ctx.registers().setVgprMsBs(llvm::rotr<uint8_t>(ModeEncoding, 2));
}

// Raise a hardware-register read according to Policy.
Error handleGetreg(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op,
                   unsigned Id, HwregPolicy Policy) {
  if (Policy.Read == HwregRead::Abort)
    return unsupported(
        Ctx, Di,
        Twine("cannot reproduce hardware-register read for id ") + Twine(Id));

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Ctx.registers().writeReg32(*Dst, Ctx.B.getInt32(0));
  return Error::success();
}

// Raise a hardware-register write according to Policy.
Error handleSetreg(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op,
                   unsigned Selector, unsigned Id, HwregPolicy Policy) {
  if (Policy.Write == HwregWrite::Abort)
    return unsupported(
        Ctx, Di,
        Twine("cannot reproduce hardware-register write for id ") + Twine(Id));
  if (Policy.Write == HwregWrite::Drop)
    return Error::success();

  Value *ValueArg;
  if (Di.CanonOp == CanonicalOp::S_SETREG_IMM32_B32) {
    if (Di.numOperands() < 2 || !Di.isImm(0))
      return unsupported(Ctx, Di,
                         "setreg immediate value is not immediate operand 0");
    uint64_t Value = static_cast<uint64_t>(Di.getImm(0));
    ValueArg = Ctx.B.getInt32(Value);
    if (Ctx.Projection.sourceIsa().has1024AddressableVgprs() &&
        Id == AMDGPU::Hwreg::ID_MODE)
      updateImmediateModeVgprMsb(Ctx, Value);
  } else {
    if (Ctx.Projection.sourceIsa().has1024AddressableVgprs() &&
        Id == AMDGPU::Hwreg::ID_MODE &&
        overlapsVgprMsb(decodeHwregField(Selector)))
      return unsupported(Ctx, Di,
                         "dynamic MODE write overlaps VGPR_MSB bits [12:19]");
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
Error handleSOPK(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::S_GETREG_B32:
  case CanonicalOp::S_SETREG_B32:
  case CanonicalOp::S_SETREG_IMM32_B32:
    break;
  default:
    return unsupported(Ctx, Di);
  }

  Expected<unsigned> Selector = readHwregSelector(Ctx, Di);
  if (!Selector)
    return Selector.takeError();
  unsigned Id = hwregId(*Selector);
  HwregPolicy Policy = classifyHwreg(Id);

  if (Di.CanonOp == CanonicalOp::S_GETREG_B32)
    return handleGetreg(Ctx, Di, Op, Id, Policy);
  return handleSetreg(Ctx, Di, Op, *Selector, Id, Policy);
}

} // namespace COMGR::hotswap
