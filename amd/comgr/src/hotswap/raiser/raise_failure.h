//===- raise_failure.h - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_H
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR::hotswap {

struct DecodedInst;
class RaiseContext;

// Structured reason for a raise failure.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  // Caller-supplied input was rejected before any IR was built (e.g. an empty
  // or non-AMDGPU source ISA string). `detail()` carries the offending input.
  BadInput,
  // Internal contract violation: a failure return path was reached with no
  // structured failure to explain it. A Hotswap bug, not a source-kernel
  // property.
  InternalError,
  // The instruction's opcode is not lifted.
  UnsupportedOpcode,
  // The instruction's opcode is lifted, but this operand shape or encoding
  // variant is not. `detail()` carries shape-specific context when available.
  UnsupportedInstructionForm,
  // A source floating-point mode is unsupported or cannot be represented on
  // the target.
  UnsupportedFloatingPointMode,
  // An instruction writes EXEC through a path the lift does not model.
  SPEUnsafeExecWriter,
  // `createTargetMachine` returned null.
  TargetMachineCreationFailed,
  // `verifyModule` rejected the emitted IR.
  IRVerificationFailed,
  // Control flow targets an offset outside the selected kernel symbol, or an
  // in-extent target could not be decoded. Crossing the boundary would inspect
  // neighboring symbols.
  KernelBoundaryViolation,
  // The kernel extent runs out without an instruction that ends the program,
  // so the code is truncated or the extent is misbounded. Distinct from a
  // boundary violation, which is a branch leaving the extent rather than
  // execution falling off its end.
  UnterminatedKernelExtent,
  // A helper or device-library bitcode link step failed. Distinct from a
  // verifier failure: the module is intentionally incomplete until the linked
  // body is inlined.
  DeviceLibraryLinkFailed,
  // Wave-size-obstruction refusals, split one enumerator per refusal so
  // diagnostics can bucket them without parsing the message text.
  CrossWaveLaneIdLeak,
  CrossWaveUnrewritableShuffle,
  CrossWaveShuffleRewritePending,
  CrossWaveReplicaRace,
  CrossWaveLanePredicatedExec,
  // A lane-position value gates a side effect without being masked to the
  // source wave width.
  CrossWavePredicateChain,
  // `HSA_HOTSWAP_STRICT=1` refusal: a lowering that would otherwise warn and
  // continue is rejected as potentially miscompiling.
  StrictUnsafeLowering,
  // The kernel descriptor could not be read from `.rodata` via the `<name>.kd`
  // symbol, so the user-SGPR layout cannot be derived.
  MissingKernelDescriptor,
  // Source descriptor fields do not describe a valid, self-consistent user
  // SGPR layout.
  UserSgprLayoutMismatch,
  // The source ABI preloads an entry SGPR the raiser cannot reproduce on the
  // target, so its value would be read as undef.
  UnsupportedEntrySgprSource,
  // The source object declares non-disabled workgroup cluster dimensions, so
  // TTMP6 carries per-cluster state the Hotswap ABI model does not reconstruct.
  UnsupportedSourceClusterDims,
  // Source and target use different models to combine the program-controlled
  // user priority with the system-assigned priority. The dispatch-time system
  // priority is unavailable, so the raiser cannot prove that source wave
  // ordering is preserved.
  UnsupportedWavePriority,
};

// Human-readable name for a `RaiseFailureReason`. Stable enough for
// diagnostics and tests to bucket on.
llvm::StringRef reasonString(RaiseFailureReason R);

// Which kernel of a batch raise a failure came out of, and the ISA pair that
// raise ran under. Both processor names are the ones the MC layers were built
// for, so they name a GPU even when the caller passed a full target identifier.
struct FailureOrigin {
  std::string KernelName;
  std::string SourceCpu;
  std::string TargetCpu;
};

// Payload of the `llvm::Error` the raiser produces on a refusal. Build one
// through the shape factories below, which return the `llvm::Error` directly;
// the constructor is exposed only because `make_error` needs it. The data
// members are private, so a failure is always fully formed and cannot be
// mutated field by field.
struct RaiseFailure : public llvm::ErrorInfo<RaiseFailure> {
  static char ID;

  RaiseFailure(RaiseFailureReason Reason, std::string Mnemonic,
               std::optional<std::string> Format,
               std::optional<uint64_t> Offset, std::string Detail,
               std::optional<FailureOrigin> Origin = std::nullopt)
      : Reason(Reason), Mnemonic(std::move(Mnemonic)),
        Format(std::move(Format)), Offset(Offset), Detail(std::move(Detail)),
        Origin(std::move(Origin)) {}

  RaiseFailureReason reason() const { return Reason; }

  // Offending instruction mnemonic (e.g. `global_store_dwordx4`); empty for a
  // failure not tied to a decoded instruction.
  llvm::StringRef mnemonic() const { return Mnemonic; }

  // Encoding-format category of the offending instruction (e.g. `VALU`,
  // `FLAT`); absent for a failure not tied to a decoded instruction. This is
  // the instruction's format, distinct from the failure reason, which is
  // `reason()`.
  std::optional<llvm::StringRef> format() const {
    if (Format)
      return llvm::StringRef(*Format);
    return std::nullopt;
  }

  // Byte offset of the offending instruction into the disassembled text
  // section; absent for a failure not tied to a decoded instruction.
  std::optional<uint64_t> offset() const { return Offset; }

  // Optional human-readable context.
  llvm::StringRef detail() const { return Detail; }

  void log(llvm::raw_ostream &OS) const override;

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  // Failure tied to a decoded instruction, located by `Mnemonic` and `Offset`.
  // `Format` is the encoding-format category of the offending instruction.
  static llvm::Error atInstruction(RaiseFailureReason Reason,
                                   llvm::StringRef Mnemonic, uint64_t Offset,
                                   llvm::StringRef Format,
                                   const llvm::Twine &Detail = {});

  // Failure scoped to a whole kernel rather than one instruction. The rendered
  // message is `kernel '<KernelName>': <Detail>`.
  static llvm::Error inKernel(RaiseFailureReason Reason,
                              llvm::StringRef KernelName,
                              const llvm::Twine &Detail);

  // Pipeline-level failure carrying only a detail string, with no instruction
  // or kernel context.
  static llvm::Error general(RaiseFailureReason Reason,
                             const llvm::Twine &Detail);

  // Name the kernel and ISA pair `Err` came out of. Refusals are raised deep in
  // the dispatch, which knows the offending instruction but not which of a
  // batch's kernels holds it, so the raise stamps that on the way out. An error
  // that is not a `RaiseFailure` passes through unchanged.
  static llvm::Error withOrigin(llvm::Error Err, llvm::StringRef KernelName,
                                llvm::StringRef SourceCpu,
                                llvm::StringRef TargetCpu);

private:
  RaiseFailureReason Reason;
  std::string Mnemonic;
  std::optional<std::string> Format;
  std::optional<uint64_t> Offset;
  std::string Detail;
  std::optional<FailureOrigin> Origin;
};

// Return a failure for an unsupported decoded instruction.
llvm::Error unsupportedInstruction(RaiseContext &Ctx, const DecodedInst &Di,
                                   const llvm::Twine &Detail = {});

} // namespace COMGR::hotswap

#endif
