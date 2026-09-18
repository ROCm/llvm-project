//===- raise-context.cpp - Transpiler -------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/raise-context.h"

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"

#include <cassert>
#include <utility>

using namespace llvm;

namespace COMGR::transpiler {

Expected<RaiseContext>
RaiseContext::create(IRBuilder<> &B, const WaveProjection &Projection,
                     const MCState &MC, const KernelMeta &Meta,
                     ArrayRef<uint8_t> SourceTextBytes,
                     uint64_t SourceTextBaseAddress,
                     ArrayRef<TextSection::ImageSection> SourceImageSections,
                     uint64_t KernelStartOffset, uint64_t KernelEndOffset,
                     std::optional<bool> SourceSramEcc) {
  Expected<RegisterState> Registers =
      RegisterState::create(B, Projection, MC, Meta);
  if (!Registers)
    return Registers.takeError();
  const unsigned SourceFloatRoundMode32 = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
  const unsigned SourceFloatRoundMode16_64 = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64);
  const bool SourceFp16Overflow = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_GFX9_PLUS_FP16_OVFL);
  bool Dx10Clamp = true;
  bool IeeeMode = true;
  if (Projection.SourceSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode)) {
    Dx10Clamp =
        AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                        amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP);
    IeeeMode =
        AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                        amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE);
  }
  RaiseContext Context(B, Projection, MC, std::move(*Registers),
                       SourceTextBytes, SourceTextBaseAddress,
                       SourceImageSections, KernelStartOffset, KernelEndOffset,
                       SourceFloatRoundMode32, SourceFloatRoundMode16_64,
                       SourceFp16Overflow, Dx10Clamp, IeeeMode);
  Context.SourceSramEcc = SourceSramEcc;
  return Context;
}

void RaiseContext::requireZeroBits(Value *Value, uint32_t Mask,
                                   const DecodedInst &Di, StringRef Detail) {
  assert(Value->getType()->isIntegerTy(32) && "expected a register word");
  BitRequirements.push_back({Value, Mask, &Di, Detail});
}

Error RaiseContext::validateRequiredBits() const {
  const DataLayout &Layout = B.GetInsertBlock()->getModule()->getDataLayout();
  for (const RequiredBits &Requirement : BitRequirements) {
    assert(Requirement.Value && "required value was deleted before validation");
    KnownBits Bits = computeKnownBits(Requirement.Value, Layout);
    if ((Bits.Zero.getZExtValue() & Requirement.Mask) != Requirement.Mask) {
      const DecodedInst &Di = *Requirement.Instruction;
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags), Requirement.Detail);
    }
  }
  return Error::success();
}

RaiseContext::RaiseContext(
    IRBuilder<> &B, const WaveProjection &Projection, const MCState &MC,
    RegisterState Registers, ArrayRef<uint8_t> SourceTextBytes,
    uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset,
    unsigned SourceFloatRoundMode32, unsigned SourceFloatRoundMode16_64,
    bool SourceFp16Overflow, bool SourceDx10Clamp, bool SourceIeeeMode)
    : B(B), Projection(Projection), MC(MC), Registers(std::move(Registers)),
      SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset),
      SourceFloatRoundMode32(SourceFloatRoundMode32),
      SourceFloatRoundMode16_64(SourceFloatRoundMode16_64),
      SourceFp16Overflow(SourceFp16Overflow), SourceDx10Clamp(SourceDx10Clamp),
      SourceIeeeMode(SourceIeeeMode) {}

Error RaiseContext::validateFPEnvironment(const DecodedInst &Di,
                                          Type *Ty) const {
  assert((Ty->isHalfTy() || Ty->isFloatTy() || Ty->isDoubleTy()) &&
         "unsupported floating-point type");

  if (Ty->isFloatTy() &&
      !Projection.TargetSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode)) {
    if (!SourceDx10Clamp) {
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedFloatingPointMode,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          "source DX10_CLAMP=0 is not representable on a target with fixed "
          "DX10 clamp mode");
    }

    if (!SourceIeeeMode) {
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedFloatingPointMode,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          "source IEEE_MODE=0 is not representable on a target with fixed "
          "IEEE mode");
    }
  }

  if (Ty->isHalfTy() && SourceFp16Overflow) {
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedFloatingPointMode,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "FP16 overflow saturation is unsupported");
  }

  unsigned RoundMode =
      Ty->isFloatTy() ? SourceFloatRoundMode32 : SourceFloatRoundMode16_64;
  if (RoundMode != amdhsa::FLOAT_ROUND_MODE_NEAR_EVEN) {
    StringRef TypeName = Ty->isHalfTy()    ? "f16"
                         : Ty->isFloatTy() ? "f32"
                                           : "f64";
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedFloatingPointMode,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        Twine(TypeName) + " rounding mode " + Twine(RoundMode) +
            " is unsupported");
  }

  return Error::success();
}

BasicBlock *RaiseContext::lookupBB(uint64_t Addr) {
  DenseMap<uint64_t, BasicBlock *>::iterator It = OffsetToBb.find(Addr);
  if (It != OffsetToBb.end())
    return It->second;
  // Every branch target is a block leader recorded during CFG layout, so a
  // miss is a raiser bug, not a recoverable case.
  report_fatal_error("transpiler: missing basic block for offset 0x" +
                     Twine::utohexstr(Addr));
}

void RaiseContext::defineBB(uint64_t Addr, BasicBlock *BB) {
  if (!OffsetToBb.try_emplace(Addr, BB).second)
    report_fatal_error("transpiler: duplicate basic block for offset 0x" +
                       Twine::utohexstr(Addr));
}

Value *RaiseContext::emitLaneIdx() { return Projection.emitLaneIdx(B); }

Value *RaiseContext::freezeMemAddr(Value *Addr) {
  if (Projection.sourceWaveSize() != 32 || Projection.targetWaveSize() == 32)
    return Addr;
  return B.CreateFreeze(Addr, "mem_addr_frozen");
}

} // namespace COMGR::transpiler
