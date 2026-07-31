//===- raise-failure.cpp - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise-failure.h"

#include "decoded-inst.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR::hotswap {

char RaiseFailure::ID = 0;

void RaiseFailure::log(llvm::raw_ostream &OS) const {
  OS << reasonString(Reason);
  if (!Mnemonic.empty()) {
    OS << ": " << Mnemonic;
    if (!Format.empty())
      OS << " [" << Format << "]";
    OS << " @offset=0x";
    OS.write_hex(Offset);
  } else if (!Format.empty()) {
    OS << ": [" << Format << "]";
  }
  if (!Detail.empty())
    OS << " :: " << Detail;
}

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
  case RaiseFailureReason::SPEUnsafeExecWriter:
    return "SPE-unmodeled-EXEC-writer";
  case RaiseFailureReason::LiftedKernargSegmentMismatch:
    return "lifted-kernarg-segment-mismatch";
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
  case RaiseFailureReason::UnsupportedSourceClusterDims:
    return "unsupported-source-cluster-dims";
  }
  llvm_unreachable("unhandled RaiseFailureReason");
}

namespace {

// Build a kernel-scoped failure: fixed pseudo-mnemonic tag, `reasonString` as
// the format field, and a `kernel '<name>': <detail>` message.
llvm::Error makeKernelScopedFailure(RaiseFailureReason Reason,
                                    llvm::StringRef Mnemonic,
                                    llvm::StringRef KernelName,
                                    const llvm::Twine &Detail,
                                    uint64_t Offset = 0) {
  return llvm::make_error<RaiseFailure>(
      Reason, Mnemonic.str(), reasonString(Reason), Offset,
      ("kernel '" + KernelName + "': " + Detail).str());
}

} // namespace

llvm::Error RaiseFailure::unsupportedInstructionForm(
    const DecodedInst &Di, llvm::StringRef Format, const llvm::Twine &Detail) {
  return llvm::make_error<RaiseFailure>(
      RaiseFailureReason::UnsupportedInstructionForm, Di.Mnemonic, Format.str(),
      Di.Offset, Detail.str());
}

llvm::Error RaiseFailure::unsupportedOpcode(const DecodedInst &Di,
                                            llvm::StringRef Format) {
  return llvm::make_error<RaiseFailure>(RaiseFailureReason::UnsupportedOpcode,
                                        Di.Mnemonic, Format.str(), Di.Offset,
                                        std::string());
}

llvm::Error RaiseFailure::speUnsafeExecWriter(const DecodedInst &Di,
                                              const llvm::Twine &Detail) {
  return llvm::make_error<RaiseFailure>(
      RaiseFailureReason::SPEUnsafeExecWriter, Di.Mnemonic,
      "SPE-unmodeled-EXEC-writer", Di.Offset, Detail.str());
}

llvm::Error RaiseFailure::targetMachineCreationFailed() {
  return llvm::make_error<RaiseFailure>(
      RaiseFailureReason::TargetMachineCreationFailed, std::string(),
      reasonString(RaiseFailureReason::TargetMachineCreationFailed), 0,
      "createTargetMachine returned null");
}

llvm::Error RaiseFailure::internalFailure(const llvm::Twine &Detail) {
  return llvm::make_error<RaiseFailure>(
      RaiseFailureReason::InternalError, std::string(),
      reasonString(RaiseFailureReason::InternalError), 0, Detail.str());
}

llvm::Error RaiseFailure::badInput(const llvm::Twine &Detail) {
  return llvm::make_error<RaiseFailure>(
      RaiseFailureReason::BadInput, std::string(),
      reasonString(RaiseFailureReason::BadInput), 0, Detail.str());
}

llvm::Error RaiseFailure::irVerificationFailed(llvm::StringRef Err) {
  return llvm::make_error<RaiseFailure>(
      RaiseFailureReason::IRVerificationFailed, std::string(),
      reasonString(RaiseFailureReason::IRVerificationFailed), 0, Err.str());
}

llvm::Error RaiseFailure::kernelBoundaryViolation(llvm::StringRef KernelName,
                                                  uint64_t TargetOffset,
                                                  const llvm::Twine &Detail) {
  return makeKernelScopedFailure(RaiseFailureReason::KernelBoundaryViolation,
                                 "<kernel-boundary>", KernelName, Detail,
                                 TargetOffset);
}

llvm::Error RaiseFailure::deviceLibraryLinkFailed(llvm::StringRef KernelName,
                                                  const llvm::Twine &Detail) {
  return makeKernelScopedFailure(RaiseFailureReason::DeviceLibraryLinkFailed,
                                 "<device-library-link>", KernelName, Detail);
}

// ----------------------------------------------------------------------------
// Phase 1.4.5 wave-size-obstruction factories. All share the same
// structure: take the refused instruction for mnemonic / offset, and
// a kind-specific detail string for the `detail` field.
// ----------------------------------------------------------------------------

namespace {

llvm::Error makeCrossWaveFailure(RaiseFailureReason Reason,
                                 const DecodedInst &Di,
                                 const llvm::Twine &KindDetail) {
  return llvm::make_error<RaiseFailure>(
      Reason, Di.Mnemonic, reasonString(Reason), Di.Offset, KindDetail.str());
}

} // namespace

llvm::Error RaiseFailure::crossWaveLaneIdLeak(const DecodedInst &Di,
                                              const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveLaneIdLeak, Di,
                              KindDetail);
}

llvm::Error
RaiseFailure::crossWaveUnrewritableShuffle(const DecodedInst &Di,
                                           const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveUnrewritableShuffle,
                              Di, KindDetail);
}

llvm::Error
RaiseFailure::crossWaveShuffleRewritePending(const DecodedInst &Di,
                                             const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveShuffleRewritePending, Di, KindDetail);
}

llvm::Error RaiseFailure::crossWaveReplicaRace(const DecodedInst &Di,
                                               const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveReplicaRace, Di,
                              KindDetail);
}

llvm::Error
RaiseFailure::crossWaveLanePredicatedExec(const DecodedInst &Di,
                                          const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveLanePredicatedExec,
                              Di, KindDetail);
}

// see hotswap/docs/modrep-predicate-chain.md sec. 5 (narrow-O1)
llvm::Error RaiseFailure::crossWavePredicateChain(llvm::StringRef KernelName,
                                                  const llvm::Twine &Detail) {
  return makeKernelScopedFailure(RaiseFailureReason::CrossWavePredicateChain,
                                 "workitem.id.x-predicate-chain-classifier",
                                 KernelName, Detail);
}

llvm::Error RaiseFailure::strictUnsafeLowering(const DecodedInst &Di,
                                               llvm::StringRef Site,
                                               llvm::StringRef Detail) {
  return llvm::make_error<RaiseFailure>(
      RaiseFailureReason::StrictUnsafeLowering, Di.Mnemonic, Site.str(),
      Di.Offset, Detail.str());
}

llvm::Error RaiseFailure::missingKernelDescriptor(llvm::StringRef KernelName) {
  return makeKernelScopedFailure(RaiseFailureReason::MissingKernelDescriptor,
                                 "<kernel-descriptor>", KernelName,
                                 ".kd symbol not parsed");
}

llvm::Error RaiseFailure::userSgprLayoutMismatch(llvm::StringRef KernelName,
                                                 const llvm::Twine &Detail) {
  return makeKernelScopedFailure(RaiseFailureReason::UserSgprLayoutMismatch,
                                 "<user-sgpr-layout>", KernelName, Detail);
}

llvm::Error
RaiseFailure::unsupportedSourceClusterDims(llvm::StringRef KernelName,
                                           const llvm::Twine &Detail) {
  return makeKernelScopedFailure(
      RaiseFailureReason::UnsupportedSourceClusterDims, "<source-cluster-dims>",
      KernelName, Detail);
}

llvm::Error
RaiseFailure::crossWaveRewriteOracleDisagreement(llvm::StringRef KernelName,
                                                 const llvm::Twine &Detail) {
  return makeKernelScopedFailure(RaiseFailureReason::CrossWaveLaneIdLeak,
                                 "writelane/readlane-post-raise-safety-net",
                                 KernelName, Detail);
}

} // namespace COMGR::hotswap
