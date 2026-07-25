//===- comgr-hotswap-liveness-analysis.h - HotSwap register liveness -----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Backward, CFG-aware SGPR/VGPR/ACC_VGPR liveness for the HotSwap register-
/// liveness port. This is the fourth stage: it ties together the RegisterSet
/// data layer, the InstDefUse extractor, and the reglive::Cfg into an
/// iterative dataflow solver, then materializes a live-before set for each
/// decoded instruction.
///
/// The analysis is kernel-scoped: callers may restrict it to the blocks
/// reachable from one entry, and edges leaving that scope are ignored. Only
/// ordinary SGPRs, VGPRs, and ACC_VGPRs are modeled; special state (EXEC, VCC,
/// SCC, M0, FLAT_SCRATCH, ...) is not part of the dataflow. Like the rest of
/// the port, nothing here is wired into a production rewrite path yet.
///
/// EXEC-masked vector writes and predicated writes preserve their old value on
/// at least one lane or path, so they are not treated as unconditional kills
/// (see the def/use flags produced by InstDefUse). VGPR/ACC liveness is
/// therefore conservative by construction, while scalar liveness is precise.
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_LIVENESS_ANALYSIS_H
#define COMGR_HOTSWAP_LIVENESS_ANALYSIS_H

#include "comgr-hotswap-cfg.h"
#include "comgr-hotswap-liveness.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llvm {
class MCInstrInfo;
class MCRegisterInfo;
} // namespace llvm

namespace COMGR {
namespace hotswap {

struct InternalDecodedInst;

namespace reglive {

/// Block-level dataflow state.
///
/// \c Gen is the upward-exposed use set (registers read before any local
/// definition); \c Kill is the set of registers definitely overwritten in the
/// block. The backward equations are:
///   LiveOut(B) = union of LiveIn(S) for each successor S
///   LiveIn(B)  = Gen(B) | (LiveOut(B) - Kill(B))
struct BlockLiveness {
  RegisterSet LiveIn;
  RegisterSet LiveOut;
  RegisterSet Gen;
  RegisterSet Kill;
};

/// Tunables for the analysis and its scratch queries.
struct LivenessAnalysisOptions {
  /// Lowest VGPR index the scratch finders may return. An allocation floor
  /// (not a dataflow fact): the live-before sets are unaffected, but
  /// findFreeRun() will not hand out a VGPR below this index, so callers can
  /// force scratch above a descriptor-declared range.
  uint16_t MinFreeVgpr = 0;
};

/// Backward SGPR/VGPR/ACC_VGPR liveness over one decoded CFG scope.
class LivenessAnalysis {
public:
  /// Solve liveness for \p Graph over \p Decoded.
  ///
  /// \p MCII / \p MRI drive per-instruction def/use extraction. When \p Scope
  /// is empty every block is analyzed; otherwise only the listed block indices
  /// participate and edges leaving the scope are ignored (pass the blocks
  /// reachable from a single kernel entry to analyze one kernel in isolation).
  LivenessAnalysis(llvm::ArrayRef<InternalDecodedInst> Decoded,
                   const Cfg &Graph, const llvm::MCInstrInfo &MCII,
                   const llvm::MCRegisterInfo &MRI,
                   llvm::ArrayRef<unsigned> Scope = {},
                   LivenessAnalysisOptions Options = {});

  /// Dataflow state for the block at \p BlockIndex (into Cfg::Blocks). Blocks
  /// outside the analyzed scope report empty sets.
  [[nodiscard]] const BlockLiveness &blockLiveness(unsigned BlockIndex) const;

  /// Registers live immediately before the instruction at \p InstIndex (into
  /// the decoded vector). Instructions outside the scope report the empty set.
  [[nodiscard]] const RegisterSet &liveBefore(size_t InstIndex) const;

  /// Whether \p Ref is live immediately before the instruction at \p InstIndex.
  [[nodiscard]] bool isLiveBefore(size_t InstIndex, RegisterRef Ref) const;

  /// Base index of \p Count consecutive VGPRs that are all dead immediately
  /// before the instruction at \p InstIndex, or std::nullopt if none fit.
  ///
  /// The search starts at max(\p SearchStart, options.MinFreeVgpr). Used to
  /// allocate temporary VGPRs when a lowering expands one instruction into a
  /// host sequence. Returns std::nullopt when \p InstIndex was not analyzed.
  [[nodiscard]] std::optional<uint16_t>
  findFreeRun(size_t InstIndex, uint16_t Count, uint16_t SearchStart = 0) const;

  /// Base index of an even-aligned dead SGPR pair before the instruction at
  /// \p InstIndex, or std::nullopt. Even alignment is required for pair
  /// operations such as saving EXEC with an s_mov_b64-style move.
  [[nodiscard]] std::optional<uint16_t>
  findFreeSgprPair(size_t InstIndex, uint16_t SearchStart = 0) const;

  /// Index of one dead SGPR before the instruction at \p InstIndex, or
  /// std::nullopt.
  [[nodiscard]] std::optional<uint16_t>
  findFreeSgpr(size_t InstIndex, uint16_t SearchStart = 0) const;

private:
  std::vector<BlockLiveness> BlockState;
  std::unordered_map<size_t, RegisterSet> LiveBeforeByIndex;
  RegisterSet Empty;
  uint16_t MinFreeVgpr = 0;
};

} // namespace reglive
} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_LIVENESS_ANALYSIS_H
