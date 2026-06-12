//===- raise-failure.cpp - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise-failure.h"

#include "decoded-inst.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR::hotswap {

// Stable diagnostic token for each structured raise-failure category.
const char *reasonString(RaiseFailureReason R) {
  switch (R) {
  case RaiseFailureReason::None:
    return "None";
  case RaiseFailureReason::BadInput:
    return "BadInput";
  case RaiseFailureReason::InternalError:
    return "internal-raise-failure";
  case RaiseFailureReason::UnsupportedOpcode:
    return "UnsupportedOpcode";
  case RaiseFailureReason::UnsupportedInstructionForm:
    return "unsupported-instruction-form";
  case RaiseFailureReason::UnsupportedSourceHiddenArg:
    return "unsupported-source-hidden-arg";
  case RaiseFailureReason::SPEUnsafeExecWriter:
    return "SPE-unmodeled-EXEC-writer";
  case RaiseFailureReason::TargetMachineCreationFailed:
    return "TargetMachineCreationFailed";
  case RaiseFailureReason::IRVerificationFailed:
    return "IRVerificationFailed";
  case RaiseFailureReason::KernelBoundaryViolation:
    return "kernel-boundary-violation";
  case RaiseFailureReason::DeviceLibraryLinkFailed:
    return "device-library-link-failed";
  case RaiseFailureReason::CrossWaveLaneIdLeak:
    return "cross-wave-lane-id-leak";
  case RaiseFailureReason::CrossWaveUnrewritableShuffle:
    return "cross-wave-unrewritable-shuffle";
  case RaiseFailureReason::CrossWaveShuffleRewritePending:
    return "cross-wave-shuffle-rewrite-pending";
  case RaiseFailureReason::CrossWaveReplicaRace:
    return "cross-wave-replica-race";
  case RaiseFailureReason::CrossWaveLanePredicatedExec:
    return "cross-wave-lane-predicated-exec";
  case RaiseFailureReason::CrossWavePredicateChain:
    return "cross-wave-predicate-chain";
  case RaiseFailureReason::StrictUnsafeLowering:
    return "strict-unsafe-lowering";
  case RaiseFailureReason::MissingKernelDescriptor:
    return "missing-kernel-descriptor";
  case RaiseFailureReason::UserSgprLayoutMismatch:
    return "user-sgpr-layout-mismatch";
  }
  llvm_unreachable("unhandled RaiseFailureReason");
}

// Emit the canonical diagnostic spelling without forcing callers to allocate.
void printRaiseFailure(llvm::raw_ostream &OS, const RaiseFailure &F) {
  OS << reasonString(F.Reason);
  if (!F.Mnemonic.empty()) {
    OS << ": " << F.Mnemonic;
    if (!F.Format.empty())
      OS << " [" << F.Format << "]";
    OS << " @offset=0x";
    OS.write_hex(F.Offset);
  } else if (!F.Format.empty()) {
    OS << ": [" << F.Format << "]";
  }
  if (!F.Detail.empty())
    OS << " :: " << F.Detail;
}

// Return the canonical diagnostic spelling for APIs that need an owned string.
std::string formatRaiseFailure(const RaiseFailure &F) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printRaiseFailure(OS, F);
  OS.flush();
  return Result;
}

RaiseFailure RaiseFailure::unsupportedInstructionForm(
    const DecodedInst &Di, llvm::StringRef Format, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UnsupportedInstructionForm;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Format.str();
  F.Offset = Di.Offset;
  F.Detail = Detail.str();
  return F;
}

RaiseFailure RaiseFailure::unsupportedSourceHiddenArg(
    const DecodedInst &Di, llvm::StringRef Format, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UnsupportedSourceHiddenArg;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Format.str();
  F.Offset = Di.Offset;
  F.Detail = Detail.str();
  return F;
}

RaiseFailure RaiseFailure::unsupportedOpcode(const DecodedInst &Di,
                                             llvm::StringRef Format) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UnsupportedOpcode;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Format.str();
  F.Offset = Di.Offset;
  return F;
}

RaiseFailure RaiseFailure::speUnsafeExecWriter(const DecodedInst &Di) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::SPEUnsafeExecWriter;
  F.Mnemonic = Di.Mnemonic;
  F.Format = "SPE-unmodeled-EXEC-writer";
  F.Offset = Di.Offset;
  return F;
}

RaiseFailure RaiseFailure::targetMachineCreationFailed() {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::TargetMachineCreationFailed;
  F.Format = reasonString(RaiseFailureReason::TargetMachineCreationFailed);
  F.Detail = "createTargetMachine returned null";
  return F;
}

RaiseFailure RaiseFailure::internalFailure(const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::InternalError;
  F.Format = reasonString(RaiseFailureReason::InternalError);
  F.Detail = Detail.str();
  return F;
}

RaiseFailure RaiseFailure::irVerificationFailed(const llvm::Twine &Err) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::IRVerificationFailed;
  F.Format = reasonString(RaiseFailureReason::IRVerificationFailed);
  F.Detail = Err.str();
  return F;
}

RaiseFailure RaiseFailure::kernelBoundaryViolation(
    llvm::StringRef KernelName, uint64_t TargetOffset,
    const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::KernelBoundaryViolation;
  F.Mnemonic = "<kernel-boundary>";
  F.Format = reasonString(RaiseFailureReason::KernelBoundaryViolation);
  F.Offset = TargetOffset;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

RaiseFailure RaiseFailure::deviceLibraryLinkFailed(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::DeviceLibraryLinkFailed;
  F.Mnemonic = "<device-library-link>";
  F.Format = reasonString(RaiseFailureReason::DeviceLibraryLinkFailed);
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

// ----------------------------------------------------------------------------
// Phase 1.4.5 wave-size-obstruction factories. All share the same
// structure: take the refused instruction for mnemonic / offset, and
// a kind-specific detail string for the `detail` field.
// ----------------------------------------------------------------------------

namespace {

RaiseFailure makeCrossWaveFailure(RaiseFailureReason Reason,
                                  const DecodedInst &Di,
                                  const llvm::Twine &KindDetail) {
  RaiseFailure F;
  F.Reason = Reason;
  F.Mnemonic = Di.Mnemonic;
  F.Format = reasonString(Reason);
  F.Offset = Di.Offset;
  F.Detail = KindDetail.str();
  return F;
}

} // namespace

RaiseFailure RaiseFailure::crossWaveLaneIdLeak(const DecodedInst &Di,
                                               const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveLaneIdLeak, Di,
                              KindDetail);
}

RaiseFailure RaiseFailure::crossWaveUnrewritableShuffle(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveUnrewritableShuffle, Di, KindDetail);
}

RaiseFailure RaiseFailure::crossWaveShuffleRewritePending(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveShuffleRewritePending, Di, KindDetail);
}

RaiseFailure RaiseFailure::crossWaveReplicaRace(const DecodedInst &Di,
                                                const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveReplicaRace, Di,
                              KindDetail);
}

RaiseFailure
RaiseFailure::crossWaveLanePredicatedExec(const DecodedInst &Di,
                                          const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveLanePredicatedExec,
                              Di, KindDetail);
}

// see hotswap/docs/modrep-predicate-chain.md §5 (narrow-O1)
RaiseFailure RaiseFailure::crossWavePredicateChain(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::CrossWavePredicateChain;
  F.Mnemonic = "workitem.id.x-predicate-chain-classifier";
  F.Format = reasonString(RaiseFailureReason::CrossWavePredicateChain);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

RaiseFailure RaiseFailure::strictUnsafeLowering(const DecodedInst &Di,
                                                llvm::StringRef Site,
                                                const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::StrictUnsafeLowering;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Site.str();
  F.Offset = Di.Offset;
  F.Detail = Detail.str();
  return F;
}

RaiseFailure RaiseFailure::missingKernelDescriptor(llvm::StringRef KernelName) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::MissingKernelDescriptor;
  F.Mnemonic = "<kernel-descriptor>";
  F.Format = reasonString(RaiseFailureReason::MissingKernelDescriptor);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': .kd symbol not parsed").str();
  return F;
}

RaiseFailure RaiseFailure::userSgprLayoutMismatch(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UserSgprLayoutMismatch;
  F.Mnemonic = "<user-sgpr-layout>";
  F.Format = reasonString(RaiseFailureReason::UserSgprLayoutMismatch);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

RaiseFailure RaiseFailure::crossWaveRewriteOracleDisagreement(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::CrossWaveLaneIdLeak;
  F.Mnemonic = "writelane/readlane-post-raise-safety-net";
  F.Format = reasonString(RaiseFailureReason::CrossWaveLaneIdLeak);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

} // namespace COMGR::hotswap
