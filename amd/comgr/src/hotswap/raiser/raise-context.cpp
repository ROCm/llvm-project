//===- raise-context.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise-context.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

Expected<RaiseContext>
RaiseContext::create(IRBuilder<> &B, const WaveProjection &Projection,
                     const MCState &MC, const KernelMeta &Meta,
                     ArrayRef<uint8_t> SourceTextBytes,
                     uint64_t SourceTextBaseAddress,
                     ArrayRef<TextSection::ImageSection> SourceImageSections,
                     uint64_t KernelStartOffset, uint64_t KernelEndOffset) {
  Expected<RegisterState> Registers =
      RegisterState::create(B, Projection, MC, Meta);
  if (!Registers)
    return Registers.takeError();
  const unsigned SourceFloatRoundMode32 = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
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
  return RaiseContext(B, Projection, MC, std::move(*Registers), SourceTextBytes,
                      SourceTextBaseAddress, SourceImageSections,
                      KernelStartOffset, KernelEndOffset,
                      SourceFloatRoundMode32, Dx10Clamp, IeeeMode);
}

RaiseContext::RaiseContext(
    IRBuilder<> &B, const WaveProjection &Projection, const MCState &MC,
    RegisterState Registers, ArrayRef<uint8_t> SourceTextBytes,
    uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset,
    unsigned SourceFloatRoundMode32, bool SourceDx10Clamp, bool SourceIeeeMode)
    : B(B), Projection(Projection), MC(MC), Registers(std::move(Registers)),
      SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset),
      SourceFloatRoundMode32(SourceFloatRoundMode32),
      SourceDx10Clamp(SourceDx10Clamp), SourceIeeeMode(SourceIeeeMode) {
  // The builder is positioned in the entry block, which is what the source
  // kernel's first instruction raised into.
  OffsetToBb[KernelStartOffset] = B.GetInsertBlock();
}

Error RaiseContext::validateF32Environment(const DecodedInst &Di) const {
  if (!Projection.TargetSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode)) {
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

  if (SourceFloatRoundMode32 != amdhsa::FLOAT_ROUND_MODE_NEAR_EVEN) {
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedFloatingPointMode,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        Twine("f32 rounding mode ") + Twine(SourceFloatRoundMode32) +
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
  report_fatal_error(Twine("transpiler: missing basic block for offset 0x") +
                     utohexstr(Addr));
}

Value *RaiseContext::emitLaneIdx() { return Projection.emitLaneIdx(B); }

Value *RaiseContext::freezeMemAddr(Value *Addr) {
  if (Projection.sourceWaveSize() != 32 || Projection.targetWaveSize() == 32)
    return Addr;
  return B.CreateFreeze(Addr, "mem_addr_frozen");
}

} // namespace COMGR::hotswap
