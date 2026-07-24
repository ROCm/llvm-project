//===- comgr-hotswap-def-use.h - HotSwap instruction def/use extraction --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Instruction-level register def/use extraction for the HotSwap register
/// liveness port. This is the bridge between a decoded LLVM MC instruction and
/// the ISA-independent RegisterSet dataflow layer in comgr-hotswap-liveness.h.
///
/// Direction is determined by operand position: the leading MCInstrDesc def
/// operands define registers, the remaining explicit operands use registers,
/// and MCInstrDesc implicit defs/uses are added on top. Register identity and
/// class are resolved from the LLVM MCRegisterInfo via toRegisterRef(): only
/// numbered SGPRs, VGPRs, and AccVGPRs are tracked; special registers (EXEC,
/// VCC, SCC, M0, FLAT_SCR, ...) map to no RegisterRef and are ignored.
///
/// Like the rest of the register-liveness port, this is not wired into any
/// production rewrite path yet.
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_DEF_USE_H
#define COMGR_HOTSWAP_DEF_USE_H

#include "comgr-hotswap-liveness.h"

#include "llvm/MC/MCRegister.h"

#include <optional>

namespace llvm {
class MCInst;
class MCInstrInfo;
class MCRegisterInfo;
} // namespace llvm

namespace COMGR {
namespace hotswap {
namespace reglive {

/// Map a physical MC register to a tracked RegisterRef.
///
/// Classifies \p Reg by its MCRegisterInfo record-name prefix (VGPR*, SGPR*,
/// AGPR*) so special registers such as EXEC/VCC/SCC/M0/FLAT_SCR are naturally
/// excluded (they map to std::nullopt). The lane width comes from the smallest
/// enclosing register class and the base index from the register's hardware
/// encoding, so a 64-bit pair such as s[6:7] becomes {SGPR, 6, 2}. Registers
/// narrower than 32 bits (e.g. 16-bit VGPR halves) are not tracked and return
/// std::nullopt.
[[nodiscard]] std::optional<RegisterRef>
toRegisterRef(llvm::MCRegister Reg, const llvm::MCRegisterInfo &MRI);

/// Registers read and written by one decoded MC instruction.
class InstDefUse {
public:
  /// Extract explicit operand and implicit def/use register refs from \p Inst.
  /// \p MCII supplies the instruction description (def count, implicit
  /// defs/uses); \p MRI resolves register identities.
  InstDefUse(const llvm::MCInst &Inst, const llvm::MCInstrInfo &MCII,
             const llvm::MCRegisterInfo &MRI);

  RegisterSet Defs; ///< Registers overwritten by the instruction.
  RegisterSet Uses; ///< Registers read by the instruction.

  /// True if any explicit vector (VGPR/ACC_VGPR) def is present. Such writes
  /// are EXEC-masked, so inactive lanes preserve their old value; a later
  /// liveness stage must not treat them as unconditional kills.
  bool HasExecMaskedVectorDef = false;

  /// True if the destination update is conditional and must not kill the old
  /// value. AMDGPU MC exposes no direct signal for this today, so it is
  /// currently always false; it stays in the interface for parity with the
  /// source analysis and for a later, target-specific refinement.
  bool HasPredicatedDef = false;
};

} // namespace reglive
} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_DEF_USE_H
