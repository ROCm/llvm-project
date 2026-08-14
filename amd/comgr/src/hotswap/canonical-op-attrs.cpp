//===- canonical-op-attrs.cpp - Hotswap transpiler ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonical-op-attrs.h"

#include "opcode-map.h"
#include "raise-failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, EXEC_LO, EXEC_HI
#include "Utils/AMDGPUBaseInfo.h"            // AMDGPU::mc2PseudoReg

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <iterator>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Materialise the attribute table once at first-use. Aggregates every
// per-handler registration declared in `canonical-op-attrs.h`. The
// `Meyers singleton`-style static local sidesteps any cross-TU static
// initialisation ordering fuss -- the table is built lazily, not at
// dynamic-init time.
class AttrTable {
public:
  AttrTable() {
    auto Ingest = [&](ArrayRef<CanonicalOpAttrSpec> Specs) {
      for (const CanonicalOpAttrSpec &S : Specs) {
        auto Idx = static_cast<std::size_t>(S.Op);
        assert(Idx < std::size(Entries) &&
               "CanonicalOp out of range; bump entries_ to CanonicalOp_COUNT");
        Entries[Idx] = S.Attrs;
      }
    };
    Ingest(getHandlerSOP1Attrs());
    Ingest(getHandlerSOP2Attrs());
    Ingest(getHandlerValuVcmpAttrs());
  }

  const CanonicalOpAttrs &operator[](CanonicalOp Op) const {
    auto Idx = static_cast<std::size_t>(Op);
    assert(Idx < std::size(Entries) && "CanonicalOp out of range");
    return Entries[Idx];
  }

private:
  CanonicalOpAttrs
      Entries[static_cast<std::size_t>(CanonicalOp::CanonicalOp_COUNT)] = {};
};

const AttrTable &theTable() {
  static const AttrTable T;
  return T;
}

static bool descImplicitlyDefinesEXEC(const MCInstrDesc &Desc) {
  for (MCPhysReg R : Desc.implicit_defs()) {
    MCRegister Reg = AMDGPU::mc2PseudoReg(R);
    if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO || Reg == AMDGPU::EXEC_HI)
      return true;
  }
  return false;
}

} // namespace

const CanonicalOpAttrs &getCanonicalOpAttrs(CanonicalOp Op) {
  return theTable()[Op];
}

llvm::Error verifyExecAttrCoverage(const MCInstrInfo &MCII) {
  for (unsigned Mc = 0, End = MCII.getNumOpcodes(); Mc < End; ++Mc) {
    const MCInstrDesc &Desc = MCII.get(Mc);
    if (!descImplicitlyDefinesEXEC(Desc))
      continue;

    CanonicalOp Sop = canonicalOpFor(Mc);
    if (Sop == CanonicalOp::Unknown)
      continue; // covered by the generic unsupported-opcode path

    if (getCanonicalOpAttrs(Sop).RoutesExecThroughStoreExec)
      continue;

    return RaiseFailure::internalFailure(
        "MC opcode " + MCII.getName(Mc) + " (#" + Twine(Mc) +
        ") declares EXEC as an implicit def but its CanonicalOp "
        "(" +
        canonicalOpName(Sop) +
        ") is not marked routesExecThroughStoreExec. Audit "
        "the handler's EXEC write path against SPE before "
        "declaring the CanonicalOp in that handler's "
        "get*Attrs() registration.");
  }
  return llvm::Error::success();
}

} // namespace COMGR::hotswap
