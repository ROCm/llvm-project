//===- raise_failure.cpp - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise_failure.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR::hotswap {

char RaiseFailure::ID = 0;

void RaiseFailure::log(llvm::raw_ostream &OS) const {
  OS << reasonString(Reason);
  if (!Mnemonic.empty())
    OS << ": " << Mnemonic;
  if (Format)
    OS << " [" << *Format << "]";
  if (Offset) {
    OS << " @offset=0x";
    OS.write_hex(*Offset);
  }
  if (Origin)
    OS << " in kernel '" << Origin->KernelName << "' (" << Origin->SourceCpu
       << " -> " << Origin->TargetCpu << ")";
  if (!Detail.empty())
    OS << " :: " << Detail;
}

// Stable diagnostic token for each structured raise-failure category.
llvm::StringRef reasonString(RaiseFailureReason R) {
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
  case RaiseFailureReason::UnsupportedFloatingPointMode:
    return "unsupported-floating-point-mode";
  case RaiseFailureReason::SPEUnsafeExecWriter:
    return "SPE-unmodeled-EXEC-writer";
  case RaiseFailureReason::TargetMachineCreationFailed:
    return "TargetMachineCreationFailed";
  case RaiseFailureReason::IRVerificationFailed:
    return "IRVerificationFailed";
  case RaiseFailureReason::KernelBoundaryViolation:
    return "kernel-boundary-violation";
  case RaiseFailureReason::UnterminatedKernelExtent:
    return "unterminated-kernel-extent";
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
  case RaiseFailureReason::UnsupportedEntrySgprSource:
    return "unsupported-entry-sgpr-source";
  case RaiseFailureReason::UnsupportedSourceClusterDims:
    return "unsupported-source-cluster-dims";
  case RaiseFailureReason::UnsupportedWavePriority:
    return "unsupported-wave-priority";
  }
  llvm_unreachable("unhandled RaiseFailureReason");
}

llvm::Error RaiseFailure::atInstruction(RaiseFailureReason Reason,
                                        llvm::StringRef Mnemonic,
                                        uint64_t Offset, llvm::StringRef Format,
                                        const llvm::Twine &Detail) {
  return llvm::make_error<RaiseFailure>(
      Reason, Mnemonic.str(), std::optional<std::string>(Format.str()),
      std::optional<uint64_t>(Offset), Detail.str());
}

llvm::Error RaiseFailure::inKernel(RaiseFailureReason Reason,
                                   llvm::StringRef KernelName,
                                   const llvm::Twine &Detail) {
  return llvm::make_error<RaiseFailure>(
      Reason, std::string(), std::optional<std::string>(std::nullopt),
      std::optional<uint64_t>(std::nullopt),
      ("kernel '" + KernelName + "': " + Detail).str());
}

llvm::Error RaiseFailure::general(RaiseFailureReason Reason,
                                  const llvm::Twine &Detail) {
  return llvm::make_error<RaiseFailure>(
      Reason, std::string(), std::optional<std::string>(std::nullopt),
      std::optional<uint64_t>(std::nullopt), Detail.str());
}

llvm::Error RaiseFailure::withOrigin(llvm::Error Err,
                                     llvm::StringRef KernelName,
                                     llvm::StringRef SourceCpu,
                                     llvm::StringRef TargetCpu) {
  return llvm::handleErrors(
      std::move(Err), [&](std::unique_ptr<RaiseFailure> F) -> llvm::Error {
        return llvm::make_error<RaiseFailure>(
            F->Reason, F->Mnemonic, F->Format, F->Offset, F->Detail,
            FailureOrigin{KernelName.str(), SourceCpu.str(), TargetCpu.str()});
      });
}

llvm::Error unsupportedInstruction(RaiseContext &Ctx, const DecodedInst &Di) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

} // namespace COMGR::hotswap
