//===- comgr-hotswap-liveness.cpp - HotSwap register set / liveness ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// RegisterSet method bodies for the HotSwap register-liveness data layer.
/// See comgr-hotswap-liveness.h. Not wired into any production rewrite path.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-liveness.h"

namespace COMGR {
namespace hotswap {
namespace reglive {

namespace {

/// Apply \p Op to each tracked lane covered by \p Ref, dispatching on the
/// register class. Lanes that fall outside a bitset's capacity are ignored so
/// a malformed decoded index can never trap. Untracked classes are a no-op.
template <typename Op>
void forEachLane(RegisterRef Ref, std::bitset<RegisterSetMaxSgprs> &Sgprs,
                 std::bitset<RegisterSetMaxVgprs> &Vgprs,
                 std::bitset<RegisterSetMaxAccVgprs> &AccVgprs, Op &&Apply) {
  const unsigned Base = Ref.Index;
  const unsigned Width = Ref.Width == 0 ? 1u : Ref.Width;
  switch (Ref.Cls) {
  case RegClass::SGPR:
    for (unsigned I = 0; I < Width; ++I)
      if (Base + I < Sgprs.size())
        Apply(Sgprs, Base + I);
    return;
  case RegClass::VGPR:
    for (unsigned I = 0; I < Width; ++I)
      if (Base + I < Vgprs.size())
        Apply(Vgprs, Base + I);
    return;
  case RegClass::ACC_VGPR:
    for (unsigned I = 0; I < Width; ++I)
      if (Base + I < AccVgprs.size())
        Apply(AccVgprs, Base + I);
    return;
  default:
    // Untracked register class: no storage.
    return;
  }
}

} // namespace

void RegisterSet::expand(RegisterRef Ref) {
  forEachLane(Ref, Sgprs, Vgprs, AccVgprs,
              [](auto &Bits, unsigned Pos) { Bits.set(Pos); });
}

void RegisterSet::erase(RegisterRef Ref) {
  forEachLane(Ref, Sgprs, Vgprs, AccVgprs,
              [](auto &Bits, unsigned Pos) { Bits.reset(Pos); });
}

void RegisterSet::clearClass(RegClass Cls) {
  switch (Cls) {
  case RegClass::SGPR:
    Sgprs.reset();
    return;
  case RegClass::VGPR:
    Vgprs.reset();
    return;
  case RegClass::ACC_VGPR:
    AccVgprs.reset();
    return;
  default:
    return;
  }
}

bool RegisterSet::contains(RegisterRef Ref) const {
  const unsigned Base = Ref.Index;
  const unsigned Width = Ref.Width == 0 ? 1u : Ref.Width;
  auto allSet = [&](const auto &Bits) {
    for (unsigned I = 0; I < Width; ++I) {
      if (Base + I >= Bits.size() || !Bits.test(Base + I))
        return false;
    }
    return true;
  };
  switch (Ref.Cls) {
  case RegClass::SGPR:
    return allSet(Sgprs);
  case RegClass::VGPR:
    return allSet(Vgprs);
  case RegClass::ACC_VGPR:
    return allSet(AccVgprs);
  default:
    return false;
  }
}

bool RegisterSet::none() const {
  return Sgprs.none() && Vgprs.none() && AccVgprs.none();
}

size_t RegisterSet::size() const {
  return Sgprs.count() + Vgprs.count() + AccVgprs.count();
}

bool RegisterSet::intersects(const RegisterSet &Rhs) const {
  return (Sgprs & Rhs.Sgprs).any() || (Vgprs & Rhs.Vgprs).any() ||
         (AccVgprs & Rhs.AccVgprs).any();
}

RegisterSet &RegisterSet::operator|=(const RegisterSet &Rhs) {
  Sgprs |= Rhs.Sgprs;
  Vgprs |= Rhs.Vgprs;
  AccVgprs |= Rhs.AccVgprs;
  return *this;
}

RegisterSet &RegisterSet::operator&=(const RegisterSet &Rhs) {
  Sgprs &= Rhs.Sgprs;
  Vgprs &= Rhs.Vgprs;
  AccVgprs &= Rhs.AccVgprs;
  return *this;
}

RegisterSet &RegisterSet::operator-=(const RegisterSet &Rhs) {
  Sgprs &= ~Rhs.Sgprs;
  Vgprs &= ~Rhs.Vgprs;
  AccVgprs &= ~Rhs.AccVgprs;
  return *this;
}

} // namespace reglive
} // namespace hotswap
} // namespace COMGR
