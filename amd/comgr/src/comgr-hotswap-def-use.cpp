//===- comgr-hotswap-def-use.cpp - HotSwap instruction def/use -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the InstDefUse extraction and the MCRegister ->
/// RegisterRef mapping. See comgr-hotswap-def-use.h. Not wired into any
/// production rewrite path.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-def-use.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace COMGR {
namespace hotswap {
namespace reglive {

namespace {

// Low 10 bits index a VGPR/SGPR/AGPR within its file; the same masking the
// WMMA-split pass uses for VGPR base extraction.
constexpr unsigned RegIndexMask = 0x3ff;

// Number of 32-bit lanes covered by \p Reg within its own register file.
//
// The width is taken from the smallest register class that contains \p Reg
// *and* belongs to the same register file (name prefix), so a single sN/vN is
// 1 lane and an aligned pair s[i:i+1] is 2. Restricting to the register's own
// file is essential: the AMDGPU MC layer also defines synthetic cross-file
// operand classes (e.g. the nominally 16-bit "VS_16", which lists full 32-bit
// SGPRs as members for operand-legality reasons). Considering those would
// report a bogus sub-dword width.
unsigned laneWidth(llvm::MCRegister Reg, const llvm::MCRegisterInfo &MRI,
                   llvm::StringRef Prefix, llvm::StringRef AltPrefix) {
  unsigned Bits = 0;
  for (unsigned I = 0, E = MRI.getNumRegClasses(); I < E; ++I) {
    const llvm::MCRegisterClass &RC = MRI.getRegClass(I);
    llvm::StringRef Name = MRI.getRegClassName(&RC);
    if (!Name.starts_with(Prefix) && !Name.starts_with(AltPrefix))
      continue;
    if (RC.getSizeInBits() < 32)
      continue; // ignore sub-dword sibling classes (lo16/hi16, SReg_1, ...).
    if (!RC.contains(Reg))
      continue;
    if (Bits == 0 || RC.getSizeInBits() < Bits)
      Bits = RC.getSizeInBits();
  }
  return Bits / 32;
}

bool isExecMaskedDefClass(RegClass Cls) {
  return Cls == RegClass::VGPR || Cls == RegClass::ACC_VGPR;
}

void addUse(InstDefUse &DU, llvm::MCRegister Reg,
            const llvm::MCRegisterInfo &MRI) {
  if (auto Ref = toRegisterRef(Reg, MRI))
    DU.Uses.expand(*Ref);
}

void addDef(InstDefUse &DU, llvm::MCRegister Reg,
            const llvm::MCRegisterInfo &MRI, bool TrackExecMask) {
  auto Ref = toRegisterRef(Reg, MRI);
  if (!Ref)
    return;
  DU.Defs.expand(*Ref);
  if (TrackExecMask && isExecMaskedDefClass(Ref->Cls))
    DU.HasExecMaskedVectorDef = true;
}

} // namespace

std::optional<RegisterRef> toRegisterRef(llvm::MCRegister Reg,
                                         const llvm::MCRegisterInfo &MRI) {
  if (!Reg)
    return std::nullopt;

  // Classify by the MCRegisterInfo record-name prefix. Numbered GPR registers
  // (and their tuple aliases, e.g. "SGPR0_SGPR1") begin with the file prefix;
  // special registers (VCC, EXEC, SCC, M0, FLAT_SCR, TTMP, NULL, ...) do not,
  // so they map to no tracked ref.
  llvm::StringRef Name = MRI.getName(Reg);
  RegClass Cls;
  llvm::StringRef Prefix;
  llvm::StringRef AltPrefix;
  if (Name.starts_with("VGPR")) {
    Cls = RegClass::VGPR;
    Prefix = "VGPR";
    AltPrefix = "VReg";
  } else if (Name.starts_with("AGPR")) {
    Cls = RegClass::ACC_VGPR;
    Prefix = "AGPR";
    AltPrefix = "AReg";
  } else if (Name.starts_with("SGPR")) {
    Cls = RegClass::SGPR;
    Prefix = "SGPR";
    AltPrefix = "SReg";
  } else {
    return std::nullopt;
  }

  const unsigned Width = laneWidth(Reg, MRI, Prefix, AltPrefix);
  if (Width == 0)
    return std::nullopt; // sub-dword (e.g. 16-bit register halves) not tracked.

  const unsigned Base = MRI.getEncodingValue(Reg) & RegIndexMask;
  return RegisterRef{Cls, static_cast<uint16_t>(Base),
                     static_cast<uint8_t>(Width)};
}

InstDefUse::InstDefUse(const llvm::MCInst &Inst, const llvm::MCInstrInfo &MCII,
                       const llvm::MCRegisterInfo &MRI) {
  const llvm::MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  const unsigned NumDefs = Desc.getNumDefs();

  // Explicit operands: the leading NumDefs register operands are defs, the
  // remaining register operands are uses. A register that appears in both a
  // def slot and a use slot (read-modify-write / two-address form) is added to
  // both sets, so it stays live before the instruction.
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const llvm::MCOperand &Op = Inst.getOperand(I);
    if (!Op.isReg())
      continue;
    if (I < NumDefs)
      addDef(*this, Op.getReg(), MRI, /*TrackExecMask=*/true);
    else
      addUse(*this, Op.getReg(), MRI);
  }

  // Implicit defs/uses are not part of the operand list. They do not
  // participate in the EXEC-masked-vector-def signal, matching the source
  // analysis (which derives that flag from explicit destinations only).
  for (llvm::MCPhysReg Reg : Desc.implicit_defs())
    addDef(*this, llvm::MCRegister(Reg), MRI, /*TrackExecMask=*/false);
  for (llvm::MCPhysReg Reg : Desc.implicit_uses())
    addUse(*this, llvm::MCRegister(Reg), MRI);
}

} // namespace reglive
} // namespace hotswap
} // namespace COMGR
