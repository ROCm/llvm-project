//===- raise-failure.h - Structured raise-failure values ----------------===//
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
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR::hotswap {

struct DecodedInst;

// Structured reason for a raise failure. Lives in its own header so the
// handler layer (`raise-context.h`) can depend on failure values
// without pulling in `RaiseResult` and the rest of the top-level
// `raiser.h` interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  // Caller-supplied input that the entry validator rejects before the
  // MC stack is even constructed. Today this fires on an empty or
  // non-AMDGPU `sourceISA` string, which would otherwise reach
  // `createMCDisassembler` and trip the `LLVM ERROR: disassembly not
  // yet supported for subtarget` `report_fatal_error` (process abort).
  // `detail` carries the offending input string.
  BadInput,
  // Internal contract violation: a caller reached a failure return path without
  // the structured failure that should explain it. This is a Hotswap bug, not a
  // property of the source kernel.
  InternalError,
  // Main loop: no handler matched on TSFlags, or every matching handler
  // returned unhandled without setting a more specific failure. The
  // `mnemonic` / `format` / `offset` triple locates the instruction.
  UnsupportedOpcode,
  // A handler matched on CanonicalOp but the specific operand shape /
  // encoding variant it saw is not yet modelled. Today's format-
  // specific failure sites (handle_valu, handle_flat, handle_mubuf,
  // handle_mfma, handle_vopd) all use this category. `detail` carries
  // shape-specific context when available.
  UnsupportedInstructionForm,
  // Phase 1.5 gate: an EXEC-writing instruction whose CanonicalOp does not
  // have `routesExecThroughStoreExec` set in `canonical-op-attrs.cpp`.
  SPEUnsafeExecWriter,
  // Phase 2: `TargetRegistry::createTargetMachine` returned null.
  TargetMachineCreationFailed,
  // Phase 7: `verifyModule` rejected the emitted IR.
  IRVerificationFailed,
  // Post-codegen: the kernarg segment the backend emitted for the lifted kernel
  // does not match the source's, so the runtime's buffer does not describe what
  // the kernel reads. See `checkLiftedKernargSegment` in code-object-utils.h.
  // Failing to read the emitted metadata back at all is `InternalError`.
  LiftedKernargSegmentMismatch,
  // Decoded control flow targets outside the selected kernel symbol extent, or
  // the raiser could not decode an in-extent target required by static CFG
  // recovery. The selected kernel boundary is part of the source object
  // contract; crossing it would inspect/lift neighboring symbols.
  KernelBoundaryViolation,
  // A helper/device-library bitcode link step failed before verification. This
  // is distinct from verifier failure: the module is intentionally incomplete
  // until the embedded helper or device-library body is linked and inlined.
  DeviceLibraryLinkFailed,
  // Phase 1.4.5 wave-size-obstruction classifier (hotswap/docs/
  // wave-size-translation.md sec. 7's three-outcome decision procedure).
  // One reason per refusal *decision* so diagnostics can bucket
  // failures without parsing the failure text. See
  // `wave-size-obstruction.h` for the classifier
  // taxonomy and the mapping between these reasons and the more
  // specific `ObstructionKind` values.
  //
  // The Class 1..4 grouping from wave-size-translation.md sec. 6 is
  // preserved as cross-references in the comments below; it is not
  // part of the enum-value identity.
  CrossWaveLaneIdLeak, // Class 1: MbcntHiLaneIdLeak / OutOfRangeLaneOperand.
  CrossWaveUnrewritableShuffle,   // Class 2: FullWaveRotate (no sec. 7 rewrite
                                  // available).
  CrossWaveShuffleRewritePending, // Class 2: shuffle with a sec. 5.3 P-item
                                  // whose handler has not landed.
  CrossWaveReplicaRace,           // Class 3: NonCommutativeAtomic.
  CrossWaveLanePredicatedExec, // Class 4: CmpxFromLaneId / SaveExecFromLaneId.
  CrossWavePredicateChain,     // Class 5: workitem.id.x() feeds a lane-
                               // position-scoped icmp (compile-time K
                               // <= W_s-1) that gates a side effect, and
                               // the chain was not AND-masked by W_s-1.
                               // Surfaced by the post-mem2reg classifier
                               // in `c5_predicate_chain_classifier.{hpp,cpp}`,
                               // not by the MC-level
                               // `buildObstructionReport` walk. See
                               // hotswap/docs/modrep-predicate-chain.md
                               // sec. 5 (narrow-O1).
  // `HSA_HOTSWAP_STRICT=1`-only refusal (see `pipeline.h::isStrictMode`).
  // A handler recognised the CanonicalOp and *would* have lifted it under the
  // existing warn-and-continue policy, but strict mode requires the
  // honest "unsupported, may silently miscompile" verdict instead.
  // Today this covers MODE-register writes (`handle-sopk.cpp`); see the
  // integration-gap investigation for the diagnosis. Hidden-argument reads are
  // deliberately not a site: they are ordinary loads of bytes the target
  // runtime populated, correct in both modes.
  StrictUnsafeLowering,
  // Phase 4 init: extractKernelMeta failed to read the kernel descriptor
  // from .rodata via the `<name>.kd` symbol. Without the KD we cannot
  // derive UserSgprLayout (which kernel_code_properties bits are set,
  // how many dwords are preloaded, where workgroup-id SGPRs live), so
  // every Phase-4 SGPR seed would be a guess. We refuse the lift.
  MissingKernelDescriptor,
  // Phase 4 init: the KD was present, but its raw USER_SGPR_COUNT field
  // disagreed with the layout implied by kernel_code_properties plus
  // kernarg_preload for the source ISA.
  UserSgprLayoutMismatch,
  // Phase 4 init: the source code object declares non-disabled workgroup
  // cluster dimensions. TTMP6 then carries real per-cluster workgroup state
  // that the current HotSwap ABI model does not reconstruct.
  UnsupportedSourceClusterDims,
};

// Human-readable name for a `RaiseFailureReason`. Stable enough for
// diagnostics and tests to bucket on.
const char *reasonString(RaiseFailureReason R);

// RaiseFailure is both the structured failure value used throughout the
// handler layer and an `llvm::ErrorInfo`, so it can be carried directly as the
// payload of a `llvm::Error` / `Expected` failure (multiple failures are
// combined with `joinErrors`). A default-constructed value has
// `Reason == None` and does not represent a real failure.
struct RaiseFailure : public llvm::ErrorInfo<RaiseFailure> {
  static char ID;

  RaiseFailureReason Reason = RaiseFailureReason::None;
  // Offending instruction mnemonic (e.g. `global_store_dwordx4`).
  std::string Mnemonic;
  // Encoding-format category (e.g. `VALU`, `FLAT`, `MUBUF`) -- stable
  // bucketing key for the batch / corpus test summaries. For non-
  // decode-level failures (e.g. `TargetMachineCreationFailed`) this
  // is the `reasonString` of `Reason`.
  std::string Format;
  // Byte offset inside the disassembled text section, in host order.
  // Zero for failures not tied to a specific instruction.
  uint64_t Offset = 0;
  // Optional human-readable context; may include shape hints,
  // attempted rewrites, etc.
  std::string Detail;

  RaiseFailure() = default;
  RaiseFailure(RaiseFailureReason Reason, std::string Mnemonic,
               std::string Format, uint64_t Offset, std::string Detail)
      : Reason(Reason), Mnemonic(std::move(Mnemonic)),
        Format(std::move(Format)), Offset(Offset), Detail(std::move(Detail)) {}

  void log(llvm::raw_ostream &OS) const override;

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  // Handler recognised the CanonicalOp but refused the specific instruction
  // form, operand profile, or target capability. `di` supplies the mnemonic and
  // source offset.
  static llvm::Error unsupportedInstructionForm(const DecodedInst &Di,
                                                llvm::StringRef Format,
                                                const llvm::Twine &Detail = {});

  // Main loop: no handler claimed the CanonicalOp (either no TSFlags match
  // or every matching handler returned `handled=false` without
  // setting a more specific failure). `di` supplies the mnemonic /
  // offset; `format` is the human-readable encoding label.
  static llvm::Error unsupportedOpcode(const DecodedInst &Di,
                                       llvm::StringRef Format);

  // Phase 1.5 gate: an EXEC-writing instruction whose CanonicalOp does not
  // have `routesExecThroughStoreExec` declared in any handler's
  // `get*Attrs()` registration.
  static llvm::Error speUnsafeExecWriter(const DecodedInst &Di,
                                         const llvm::Twine &Detail);

  // Phase 2: `TargetRegistry::createTargetMachine` returned null.
  static llvm::Error targetMachineCreationFailed();

  // Internal invariant violation surfaced as a structured failure so callers do
  // not misclassify it as an unsupported source instruction.
  static llvm::Error internalFailure(const llvm::Twine &Detail);

  // Caller-supplied input rejected before the MC stack is built (e.g. an
  // empty or non-AMDGPU source ISA string).
  static llvm::Error badInput(const llvm::Twine &Detail);

  // Phase 7: `verifyModule` rejected the emitted IR.
  // `err` carries the verifier's diagnostic text for the `detail` field.
  static llvm::Error irVerificationFailed(llvm::StringRef Err);

  // Kernel-symbol boundary check failed during CFG recovery.
  static llvm::Error kernelBoundaryViolation(llvm::StringRef KernelName,
                                             uint64_t TargetOffset,
                                             const llvm::Twine &Detail);

  // Embedded helper/device-library linking failed before verification.
  // `kernelName` and `detail` preserve attribution for proof logs without
  // mis-bucketing the failure as an LLVM verifier rejection.
  static llvm::Error deviceLibraryLinkFailed(llvm::StringRef KernelName,
                                             const llvm::Twine &Detail);

  // Phase 1.4.5 wave-size-obstruction classifier (hotswap/docs/
  // wave-size-translation.md sec. 7). `di` supplies the offending
  // mnemonic + offset. `kindDetail` should carry the human-readable
  // `ObstructionKind` name (from `obstructionKindName`), the P-item
  // identifier from the sec. 5.3 rewrite table (where applicable), and
  // any operand-level context the classifier extracted (e.g.
  // "operand value N >= W_s=M"). The resulting failure is renderable
  // by `reasonString` for batch-test bucketing without parsing
  // `detail`.
  static llvm::Error crossWaveLaneIdLeak(const DecodedInst &Di,
                                         const llvm::Twine &KindDetail);

  static llvm::Error
  crossWaveUnrewritableShuffle(const DecodedInst &Di,
                               const llvm::Twine &KindDetail);

  static llvm::Error
  crossWaveShuffleRewritePending(const DecodedInst &Di,
                                 const llvm::Twine &KindDetail);

  static llvm::Error crossWaveReplicaRace(const DecodedInst &Di,
                                          const llvm::Twine &KindDetail);

  static llvm::Error crossWaveLanePredicatedExec(const DecodedInst &Di,
                                                 const llvm::Twine &KindDetail);

  // Phase 6.6 (post-mem2reg) IR-level classifier for the Class-5
  // predicate-chain refusal. No `DecodedInst` because
  // `workitem.id.x()` is an IR-level intrinsic call, not an MC
  // opcode. `kernelName` is captured for bucketing; `detail` names
  // the first failing call's icmp + constant so callers can surface
  // attribution without parsing `detail`. See
  // `c5_predicate_chain_classifier.{hpp,cpp}` and
  // hotswap/docs/modrep-predicate-chain.md sec. 5 (narrow-O1).
  static llvm::Error crossWavePredicateChain(llvm::StringRef KernelName,
                                             const llvm::Twine &Detail);

  // Post-raise safety net for the cross-lane writelane/readlane
  // rewrite path. Fires when the syntactic classifier (Phase 1.4.5)
  // matched the `WaveIdLiftScalarized` three-way co-occurrence (the
  // kernel contains the canonical ttmp8 wave_id BFE rescue + a
  // `v_writelane_b32` / `v_readlane_b32` site + a WMMA intrinsic)
  // AND the post-mem2reg rewrite pass rewrote zero sites. The
  // oracle disagreeing with the classifier means either the
  // classifier is over-approximating (false positive, benign-
  // looking) or the oracle is under-approximating (false negative,
  // which would let a silent miscompile through). We cannot
  // distinguish these two without a precise dataflow check, so we
  // refuse on the safe side. Uses the `CrossWaveLaneIdLeak` bucket
  // so corpus-level regression dashboards see it as "Class 1
  // refusal" alongside the other wave-id-leak kinds. No
  // `DecodedInst` because this is an IR-level decision, not tied to
  // one specific MC site.
  static llvm::Error
  crossWaveRewriteOracleDisagreement(llvm::StringRef KernelName,
                                     const llvm::Twine &Detail);

  // `HSA_HOTSWAP_STRICT=1` refusal. `site` is a short stable label
  // (e.g. `"HWREG_MODE_write"`) that callers can
  // bucket on without parsing `detail`; `detail` carries the human-readable
  // explanation of *why*
  // the lowering would silently miscompile.
  static llvm::Error strictUnsafeLowering(const DecodedInst &Di,
                                          llvm::StringRef Site,
                                          llvm::StringRef Detail);

  // Phase 4 init: kernel descriptor was not parsed from .rodata so
  // UserSgprLayout cannot be derived. `kernelName` is captured for the
  // diagnostic; there is no `DecodedInst` because the failure happens
  // before the disassembly is consumed.
  static llvm::Error missingKernelDescriptor(llvm::StringRef KernelName);

  // Phase 4 init: descriptor-derived UserSgprLayout consistency check failed.
  static llvm::Error userSgprLayoutMismatch(llvm::StringRef KernelName,
                                            const llvm::Twine &Detail);

  // Phase 4 init: source cluster dimensions are explicit and non-disabled.
  static llvm::Error unsupportedSourceClusterDims(llvm::StringRef KernelName,
                                                  const llvm::Twine &Detail);
};

} // namespace COMGR::hotswap

#endif
