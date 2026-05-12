//===- hotswap-error.h - Hotswap-originated llvm::Error payload ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `HotswapError` is the dedicated `llvm::ErrorInfo` subclass for failures
// that the hotswap transpiler detects itself: missing ELF sections,
// kernels absent from the AMDGPU MsgPack metadata, MC target-stack
// construction failures, unsupported instruction encodings, wave-size
// obstructions, etc. Errors that the transpiler *forwards* from
// `llvm::object`, `MC`, or `COMGR::lookupSymbolByName` are passed
// through as their original ErrorInfo type, so callers can still
// `handleErrors` on them and tell hotswap-originated failures apart
// from upstream LLVM ones.
//
// Two construction shapes:
//
//   * Simple-message: `makeHotswapError(detail)` returns a base
//     `HotswapError` whose only payload is the detail string. Used by
//     code-object-utils, mc-state, raiser entry-point validation,
//     etc., where a single message suffices.
//
//   * Categorised: per-category `ErrorInfo` subclasses
//     (`HotswapUnsupportedShapeError`, `HotswapCrossWaveLaneIdLeakError`,
//     ...) carry structured metadata (`Format`, `Mnemonic`, `Offset`,
//     `Detail`) so consumers can bucket failures by category via
//     `Err.isA<X>()` / `handleErrors([] (const X &) { ... })` without
//     parsing the message string, and so pipeline-level diagnostic
//     records can populate `Fail{Format,Mnemonic,Offset,Detail}`
//     fields directly. The `Format` field's value at each factory
//     matches the stable category strings the lit tests grep on
//     (`cross-wave-lane-id-leak`, `cross-wave-shuffle-rewrite-pending`,
//     etc.).
//
// Both shapes share the same base `HotswapError` so existing
// `Err.isA<HotswapError>()` discrimination across hotswap-internal vs.
// upstream-forwarded errors keeps working uniformly.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HOTSWAP_ERROR_H
#define HOTSWAP_TRANSPILER_HOTSWAP_ERROR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <system_error>

namespace COMGR::hotswap {

struct DecodedInst;

class HotswapError : public llvm::ErrorInfo<HotswapError> {
public:
  static char ID;

  // Single-message field used by `makeHotswapError(detail)` callers.
  // For categorised subclasses this is populated as
  // "<Format>: <Detail>" so `log()` output is stable regardless of
  // construction shape.
  std::string Msg;

  // Structured fields populated by categorised subclasses; empty
  // (default) for simple-message construction. Pipeline-level
  // diagnostic records read these directly to populate
  // `PipelineResult::Fail{Format,Mnemonic,Offset,Detail}` without
  // dyn_casting per category.
  std::string Format;
  std::string Mnemonic;
  uint64_t Offset = 0;
  std::string Detail;

  // Simple-message constructor. Used by call sites that don't have a
  // category (e.g. ELF/MC failures).
  explicit HotswapError(const llvm::Twine &DetailIn) : Msg(DetailIn.str()) {}

  // Categorised constructor. Stores the structured fields and
  // builds `Msg` as "<Format>: <Detail>" (omitting the colon and
  // detail if `DetailIn` is empty) so `log()` emits the stable
  // category string the lit tests grep on.
  HotswapError(llvm::StringRef FormatIn, llvm::StringRef MnemonicIn,
               uint64_t OffsetIn, const llvm::Twine &DetailIn)
      : Format(FormatIn.str()), Mnemonic(MnemonicIn.str()), Offset(OffsetIn),
        Detail(DetailIn.str()) {
    if (Detail.empty()) {
      Msg = Format;
    } else {
      Msg = (FormatIn + ": " + DetailIn).str();
    }
  }

  void log(llvm::raw_ostream &OS) const override { OS << "hotswap: " << Msg; }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

// Per-category subclasses. Each inherits HotswapError's constructors
// via `using ErrorInfo::ErrorInfo;` (LLVM's ErrorInfo template
// forwards parent-class constructors), so factory functions below
// can call `llvm::make_error<HotswapXError>(Format, Mnemonic, Offset,
// Detail)`. `Err.isA<HotswapError>()` returns true for any subclass
// because the type chain walks up via classID().

class HotswapBadInputError
    : public llvm::ErrorInfo<HotswapBadInputError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapUnsupportedOpcodeError
    : public llvm::ErrorInfo<HotswapUnsupportedOpcodeError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapUnsupportedShapeError
    : public llvm::ErrorInfo<HotswapUnsupportedShapeError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapSPEUnsafeExecWriterError
    : public llvm::ErrorInfo<HotswapSPEUnsafeExecWriterError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapTargetMachineError
    : public llvm::ErrorInfo<HotswapTargetMachineError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapIRVerificationError
    : public llvm::ErrorInfo<HotswapIRVerificationError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapCrossWaveLaneIdLeakError
    : public llvm::ErrorInfo<HotswapCrossWaveLaneIdLeakError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapCrossWaveUnrewritableShuffleError
    : public llvm::ErrorInfo<HotswapCrossWaveUnrewritableShuffleError,
                             HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapCrossWaveShuffleRewritePendingError
    : public llvm::ErrorInfo<HotswapCrossWaveShuffleRewritePendingError,
                             HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapCrossWaveReplicaRaceError
    : public llvm::ErrorInfo<HotswapCrossWaveReplicaRaceError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapCrossWaveLanePredicatedExecError
    : public llvm::ErrorInfo<HotswapCrossWaveLanePredicatedExecError,
                             HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapCrossWavePredicateChainError
    : public llvm::ErrorInfo<HotswapCrossWavePredicateChainError,
                             HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapStrictUnsafeLoweringError
    : public llvm::ErrorInfo<HotswapStrictUnsafeLoweringError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapMissingKernelDescriptorError
    : public llvm::ErrorInfo<HotswapMissingKernelDescriptorError,
                             HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

class HotswapUserSgprLayoutMismatchError
    : public llvm::ErrorInfo<HotswapUserSgprLayoutMismatchError, HotswapError> {
public:
  static char ID;
  using ErrorInfo::ErrorInfo;
};

// ===== Factory functions =====
//
// Simple-message factory: existing call sites that just want a
// HotswapError carrying a single detail string.

inline llvm::Error makeHotswapError(const llvm::Twine &Detail) {
  return llvm::make_error<HotswapError>(Detail);
}

// Categorised factories. Each mirrors the corresponding former
// `RaiseFailure::xxx(...)` factory's signature so handler-site
// rewrites are a 1:1 substitution. The `Format` string at each
// factory matches the stable category string the lit tests grep on
// (kept stable across the migration: see the rocm-side
// `reasonString()` for the canonical list).

llvm::Error makeHotswapBadInputError(const llvm::Twine &Detail);
llvm::Error makeHotswapUnsupportedOpcodeError(const DecodedInst &Di,
                                              llvm::StringRef FormatName);
llvm::Error makeHotswapUnsupportedShapeError(const DecodedInst &Di,
                                             llvm::StringRef FormatName,
                                             const llvm::Twine &Detail = {});
llvm::Error makeHotswapSPEUnsafeExecWriterError(const DecodedInst &Di);
llvm::Error makeHotswapTargetMachineError();
llvm::Error makeHotswapIRVerificationError(const llvm::Twine &VerifierMsg);
llvm::Error makeHotswapCrossWaveLaneIdLeakError(const DecodedInst &Di,
                                                const llvm::Twine &KindDetail);
llvm::Error makeHotswapCrossWaveUnrewritableShuffleError(
    const DecodedInst &Di, const llvm::Twine &KindDetail);
llvm::Error makeHotswapCrossWaveShuffleRewritePendingError(
    const DecodedInst &Di, const llvm::Twine &KindDetail);
llvm::Error
makeHotswapCrossWaveReplicaRaceError(const DecodedInst &Di,
                                     const llvm::Twine &KindDetail);
llvm::Error makeHotswapCrossWaveLanePredicatedExecError(
    const DecodedInst &Di, const llvm::Twine &KindDetail);
llvm::Error
makeHotswapCrossWavePredicateChainError(llvm::StringRef KernelName,
                                        const llvm::Twine &Detail);
// Maps to the LaneIdLeak subclass (matches the rocm-side
// `crossWaveRewriteOracleDisagreement` which reused the LaneIdLeak
// reason so corpus-level regression dashboards see it as
// "Class 1 refusal" alongside other wave-id-leak kinds).
llvm::Error makeHotswapCrossWaveRewriteOracleDisagreementError(
    llvm::StringRef KernelName, const llvm::Twine &Detail);
llvm::Error makeHotswapStrictUnsafeLoweringError(const DecodedInst &Di,
                                                 llvm::StringRef Site,
                                                 const llvm::Twine &Detail);
llvm::Error makeHotswapMissingKernelDescriptorError(llvm::StringRef KernelName);
llvm::Error
makeHotswapUserSgprLayoutMismatchError(llvm::StringRef KernelName,
                                       const llvm::Twine &Detail);

} // namespace COMGR::hotswap

#endif
