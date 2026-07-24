//===- comgr-hotswap-liveness.h - HotSwap register set / liveness --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// ISA-independent register references and register sets for HotSwap
/// dataflow analyses (register liveness, def/use, scratch allocation).
///
/// This is the leaf data layer of the HotSwap register-liveness port. It has
/// no dependency on the LLVM MC layer or on the rest of the HotSwap pipeline
/// and is intentionally not wired into any production rewrite path yet: later
/// stages build def/use extraction, a CFG-scoped dataflow solver, and scratch
/// finders on top of these types before anything consumes them.
///
/// Only ordinary SGPRs, VGPRs, and CDNA-style accumulator VGPRs are tracked.
/// Special architectural state (EXEC, VCC, SCC, M0, FLAT_SCRATCH, ...) is
/// deliberately ignored: it is represented in \c RegClass so callers can name
/// it, but a \c RegisterSet stores no bits for it.
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_LIVENESS_H
#define COMGR_HOTSWAP_LIVENESS_H

#include <bitset>
#include <cstddef>
#include <cstdint>

namespace COMGR {
namespace hotswap {
namespace reglive {

/// Register-file storage capacities for the tracked classes.
///
/// These are storage bounds for an ISA-independent analysis set, not
/// per-kernel allocation limits. They are fixed constants sized for the
/// gfx125x targets HotSwap rewrites today; a later stage can derive them from
/// \c MCSubtargetInfo / \c MCRegisterInfo if the analysis is widened past
/// gfx1250. They are kept generous so a decoded register index never overflows
/// the underlying bitset.
inline constexpr unsigned RegisterSetMaxSgprs = 128;
inline constexpr unsigned RegisterSetMaxVgprs = 256;
inline constexpr unsigned RegisterSetMaxAccVgprs = 256;

/// Ordinary SGPRs considered safe for scratch allocation. Conservative bound
/// used by later scratch-selection stages; liveness itself tracks up to
/// \c RegisterSetMaxSgprs.
inline constexpr unsigned RegisterSetAllocatableSgprs = 106;

/// ISA-independent register-file class.
///
/// Each class has its own index namespace: SGPR 4 and VGPR 4 are different
/// registers and never collide in the same set. The special classes below are
/// nameable for parity with the eventual MC mapping, but a \c RegisterSet
/// stores no bits for them.
enum class RegClass : uint8_t {
  SGPR,         ///< Scalar general-purpose register, indexed as sN.
  VGPR,         ///< Vector general-purpose register, indexed as vN.
  ACC_VGPR,     ///< CDNA accumulator VGPR, indexed as accN.
  EXEC,         ///< EXEC mask. Not tracked by RegisterSet.
  VCC,          ///< VCC condition mask. Not tracked by RegisterSet.
  SCC,          ///< Scalar condition code bit. Not tracked by RegisterSet.
  M0,           ///< M0 special scalar register. Not tracked by RegisterSet.
  FLAT_SCRATCH, ///< Flat-scratch base pair. Not tracked by RegisterSet.
  TTMP,         ///< Trap-temporary registers. Not tracked by RegisterSet.
  PC,           ///< Program counter dependency. Not tracked by RegisterSet.
};

/// A contiguous register reference within one register file.
///
/// \c Index is relative to \c Cls, not a raw operand encoding value. \c Width
/// is measured in 32-bit register lanes, so a 64-bit SGPR pair is
/// <tt>{RegClass::SGPR, base, 2}</tt>.
struct RegisterRef {
  RegClass Cls;
  uint16_t Index;
  uint8_t Width = 1;

  constexpr bool operator==(const RegisterRef &Rhs) const {
    return Cls == Rhs.Cls && Index == Rhs.Index && Width == Rhs.Width;
  }
  constexpr bool operator!=(const RegisterRef &Rhs) const {
    return !(*this == Rhs);
  }
};

/// Per-class register set used for def/use and liveness dataflow.
///
/// A RegisterSet can represent an instruction's use set, def set, a block
/// live-in/live-out set, or a live-before/live-after set. Set operations are
/// member-wise across the tracked register classes, so SGPR, VGPR, and
/// AccVGPR lanes stay disjoint.
class RegisterSet {
public:
  /// Add every 32-bit register lane covered by \p Ref. No-op for untracked
  /// classes.
  void expand(RegisterRef Ref);

  /// Remove every 32-bit register lane covered by \p Ref. No-op for untracked
  /// classes.
  void erase(RegisterRef Ref);

  /// Remove all tracked registers in one register class.
  void clearClass(RegClass Cls);

  /// Return true if every lane covered by \p Ref is present. Always false for
  /// untracked classes.
  [[nodiscard]] bool contains(RegisterRef Ref) const;

  /// Return true when no tracked register class contains any live bits.
  [[nodiscard]] bool none() const;

  /// Total number of single-lane registers tracked across all classes.
  [[nodiscard]] size_t size() const;

  /// Return true if any register lane is present in both sets.
  [[nodiscard]] bool intersects(const RegisterSet &Rhs) const;

  RegisterSet &operator|=(const RegisterSet &Rhs);
  RegisterSet &operator&=(const RegisterSet &Rhs);
  RegisterSet &operator-=(const RegisterSet &Rhs);

  friend RegisterSet operator|(RegisterSet Lhs, const RegisterSet &Rhs) {
    Lhs |= Rhs;
    return Lhs;
  }
  friend RegisterSet operator&(RegisterSet Lhs, const RegisterSet &Rhs) {
    Lhs &= Rhs;
    return Lhs;
  }
  friend RegisterSet operator-(RegisterSet Lhs, const RegisterSet &Rhs) {
    Lhs -= Rhs;
    return Lhs;
  }

  [[nodiscard]] bool operator==(const RegisterSet &Rhs) const {
    return Sgprs == Rhs.Sgprs && Vgprs == Rhs.Vgprs && AccVgprs == Rhs.AccVgprs;
  }
  [[nodiscard]] bool operator!=(const RegisterSet &Rhs) const {
    return !(*this == Rhs);
  }

  /// Invoke \p F with each tracked single-lane register in the set, visiting
  /// SGPRs, then VGPRs, then AccVGPRs in ascending index order. Each yielded
  /// RegisterRef has \c Width==1 -- multi-lane refs inserted via \c expand are
  /// visited as their constituent lanes.
  template <typename F> void forEach(F &&Fn) const {
    for (size_t I = 0; I < Sgprs.size(); ++I) {
      if (Sgprs.test(I))
        Fn(RegisterRef{RegClass::SGPR, static_cast<uint16_t>(I), 1});
    }
    for (size_t I = 0; I < Vgprs.size(); ++I) {
      if (Vgprs.test(I))
        Fn(RegisterRef{RegClass::VGPR, static_cast<uint16_t>(I), 1});
    }
    for (size_t I = 0; I < AccVgprs.size(); ++I) {
      if (AccVgprs.test(I))
        Fn(RegisterRef{RegClass::ACC_VGPR, static_cast<uint16_t>(I), 1});
    }
  }

private:
  std::bitset<RegisterSetMaxSgprs> Sgprs;
  std::bitset<RegisterSetMaxVgprs> Vgprs;
  std::bitset<RegisterSetMaxAccVgprs> AccVgprs;
};

} // namespace reglive
} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_LIVENESS_H
