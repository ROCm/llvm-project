//===-- handle-valu-output-mods.h - VOP3 output modifier guards ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared clamp/omod refusal for VALU handlers. Centralises operand reads and
// diagnostic wording so e32/e64-shared and VOP3-only opcodes do not drift.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLE_VALU_OUTPUT_MODS_H
#define HOTSWAP_TRANSPILER_HANDLE_VALU_OUTPUT_MODS_H

#include "canonical-op.h"
#include "handlers.h"

#include "llvm/ADT/StringRef.h"

namespace COMGR::hotswap {

/// Whether clamp/omod operands must exist in the MC operand table.
enum class VOP3OutputModPresence {
  /// e32 may omit both; e64/VOP3 forms must expose them when present.
  IfPresent,
  /// VOP3-only opcodes: both operands are required.
  Required,
};

/// Diagnostic shape for non-default or malformed output modifiers.
enum class VOP3OutputModDiag {
  /// Single combined message (small-op converts sharing e32/e64 CanonicalOps).
  Combined,
  /// Separate clamp vs omod messages (FP VALU ternary / clamp family).
  FpValuSplit,
  /// gfx12 VOP3 pseudo-scalar profile (hardware-intrinsic base lifts).
  PseudoScalar,
};

/// Refuse non-default VOP3 output modifiers (clamp / omod).
///
/// \p DiagnosticName is printed in errors (mnemonic string or canonicalOpName).
llvm::Error requireDefaultVOP3OutputMods(const DecodedInst &Di,
                                         llvm::StringRef DiagnosticName,
                                         VOP3OutputModPresence Presence,
                                         VOP3OutputModDiag Diag);

inline llvm::Error requireDefaultOutputModsIfPresent(const DecodedInst &Di) {
  return requireDefaultVOP3OutputMods(Di, canonicalOpName(Di.CanonOp),
                                      VOP3OutputModPresence::IfPresent,
                                      VOP3OutputModDiag::Combined);
}

inline llvm::Error requireDefaultPseudoScalarOutputMods(const DecodedInst &Di) {
  return requireDefaultVOP3OutputMods(Di, canonicalOpName(Di.CanonOp),
                                      VOP3OutputModPresence::Required,
                                      VOP3OutputModDiag::PseudoScalar);
}

/// VOP3-only FP VALU (e.g. min/max clamp). Used by handle-valu.cpp.
inline llvm::Error requireDefaultVOP3FpValuOutputMods(const DecodedInst &Di,
                                                      llvm::StringRef OpName) {
  return requireDefaultVOP3OutputMods(Di, OpName,
                                      VOP3OutputModPresence::Required,
                                      VOP3OutputModDiag::FpValuSplit);
}

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_HANDLE_VALU_OUTPUT_MODS_H
