//===- hotswap-error.cpp - HotswapError IDs + categorised factories ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-error.h"

#include "decoded-inst.h"

namespace COMGR::hotswap {

// ErrorInfo IDs. One per subclass; the address of each `ID` is what
// `Err.isA<X>()` discriminates on internally.
char HotswapError::ID = 0;
char HotswapBadInputError::ID = 0;
char HotswapUnsupportedOpcodeError::ID = 0;
char HotswapUnsupportedShapeError::ID = 0;
char HotswapSPEUnsafeExecWriterError::ID = 0;
char HotswapTargetMachineError::ID = 0;
char HotswapIRVerificationError::ID = 0;
char HotswapCrossWaveLaneIdLeakError::ID = 0;
char HotswapCrossWaveUnrewritableShuffleError::ID = 0;
char HotswapCrossWaveShuffleRewritePendingError::ID = 0;
char HotswapCrossWaveReplicaRaceError::ID = 0;
char HotswapCrossWaveLanePredicatedExecError::ID = 0;
char HotswapCrossWavePredicateChainError::ID = 0;
char HotswapStrictUnsafeLoweringError::ID = 0;
char HotswapMissingKernelDescriptorError::ID = 0;
char HotswapUserSgprLayoutMismatchError::ID = 0;

// Categorised factories. Each constructs the subclass with the
// stable category string the lit tests grep on as the `Format`
// field (matches the former `reasonString(RaiseFailureReason)`
// output character-for-character).

llvm::Error makeHotswapBadInputError(const llvm::Twine &Detail) {
  return llvm::make_error<HotswapBadInputError>("BadInput", "", 0, Detail);
}

llvm::Error makeHotswapUnsupportedOpcodeError(const DecodedInst &Di,
                                              llvm::StringRef FormatName) {
  return llvm::make_error<HotswapUnsupportedOpcodeError>(
      FormatName, Di.Mnemonic, Di.Offset, "");
}

llvm::Error makeHotswapUnsupportedShapeError(const DecodedInst &Di,
                                             llvm::StringRef FormatName,
                                             const llvm::Twine &Detail) {
  return llvm::make_error<HotswapUnsupportedShapeError>(
      FormatName, Di.Mnemonic, Di.Offset, Detail);
}

llvm::Error makeHotswapSPEUnsafeExecWriterError(const DecodedInst &Di) {
  return llvm::make_error<HotswapSPEUnsafeExecWriterError>(
      "SPE-unmodeled-EXEC-writer", Di.Mnemonic, Di.Offset, "");
}

llvm::Error makeHotswapTargetMachineError() {
  return llvm::make_error<HotswapTargetMachineError>(
      "TargetMachineCreationFailed", "", 0,
      "createTargetMachine returned null");
}

llvm::Error makeHotswapIRVerificationError(const llvm::Twine &VerifierMsg) {
  return llvm::make_error<HotswapIRVerificationError>(
      "IRVerificationFailed", "", 0, VerifierMsg);
}

llvm::Error
makeHotswapCrossWaveLaneIdLeakError(const DecodedInst &Di,
                                    const llvm::Twine &KindDetail) {
  return llvm::make_error<HotswapCrossWaveLaneIdLeakError>(
      "cross-wave-lane-id-leak", Di.Mnemonic, Di.Offset, KindDetail);
}

llvm::Error makeHotswapCrossWaveUnrewritableShuffleError(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return llvm::make_error<HotswapCrossWaveUnrewritableShuffleError>(
      "cross-wave-unrewritable-shuffle", Di.Mnemonic, Di.Offset, KindDetail);
}

llvm::Error makeHotswapCrossWaveShuffleRewritePendingError(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return llvm::make_error<HotswapCrossWaveShuffleRewritePendingError>(
      "cross-wave-shuffle-rewrite-pending", Di.Mnemonic, Di.Offset, KindDetail);
}

llvm::Error
makeHotswapCrossWaveReplicaRaceError(const DecodedInst &Di,
                                     const llvm::Twine &KindDetail) {
  return llvm::make_error<HotswapCrossWaveReplicaRaceError>(
      "cross-wave-replica-race", Di.Mnemonic, Di.Offset, KindDetail);
}

llvm::Error makeHotswapCrossWaveLanePredicatedExecError(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return llvm::make_error<HotswapCrossWaveLanePredicatedExecError>(
      "cross-wave-lane-predicated-exec", Di.Mnemonic, Di.Offset, KindDetail);
}

llvm::Error
makeHotswapCrossWavePredicateChainError(llvm::StringRef KernelName,
                                        const llvm::Twine &Detail) {
  return llvm::make_error<HotswapCrossWavePredicateChainError>(
      "cross-wave-predicate-chain", "workitem.id.x-predicate-chain-classifier",
      0, ("kernel '" + KernelName + "': " + Detail).str());
}

llvm::Error makeHotswapCrossWaveRewriteOracleDisagreementError(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  // Maps to the LaneIdLeak subclass intentionally: matches the rocm-
  // side reuse of `CrossWaveLaneIdLeak` for the safety-net so corpus
  // dashboards see this as a Class 1 refusal alongside the
  // syntactic-classifier wave-id-leak kinds.
  return llvm::make_error<HotswapCrossWaveLaneIdLeakError>(
      "cross-wave-lane-id-leak", "writelane/readlane-post-raise-safety-net", 0,
      ("kernel '" + KernelName + "': " + Detail).str());
}

llvm::Error makeHotswapStrictUnsafeLoweringError(const DecodedInst &Di,
                                                 llvm::StringRef Site,
                                                 const llvm::Twine &Detail) {
  return llvm::make_error<HotswapStrictUnsafeLoweringError>(
      Site, Di.Mnemonic, Di.Offset, Detail);
}

llvm::Error
makeHotswapMissingKernelDescriptorError(llvm::StringRef KernelName) {
  return llvm::make_error<HotswapMissingKernelDescriptorError>(
      "missing-kernel-descriptor", "<kernel-descriptor>", 0,
      ("kernel '" + KernelName + "': .kd symbol not parsed").str());
}

llvm::Error
makeHotswapUserSgprLayoutMismatchError(llvm::StringRef KernelName,
                                       const llvm::Twine &Detail) {
  return llvm::make_error<HotswapUserSgprLayoutMismatchError>(
      "user-sgpr-layout-mismatch", "<user-sgpr-layout>", 0,
      ("kernel '" + KernelName + "': " + Detail).str());
}

} // namespace COMGR::hotswap
