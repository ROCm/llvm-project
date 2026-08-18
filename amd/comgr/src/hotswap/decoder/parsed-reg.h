//===- parsed-reg.h - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_PARSED_REG_H
#define HOTSWAP_TRANSPILER_PARSED_REG_H

#include <cstdint>
#include <optional>

namespace COMGR::hotswap {

class ISAProfile;

struct ParsedReg {
  // Compact-predicate "source-only" registers (SIRegisterInfo.td:198-200):
  //   SRC_VCCZ : i1 == (VCC == 0)   read as i32 in VOP/SOP src slots
  //   SRC_EXECZ: i1 == (EXEC == 0)
  //   SRC_SCC  : i1 == SCC
  // These cannot be written. We model them as their own kinds so
  // readOp32 / readOp64 can materialise the boolean result on demand
  // (see register-state.cpp). Tensile gfx1250 emits them as F16 source
  // operands (e.g. `v_sub_f16 v64, src_vccz, v48`), so the dispatch
  // path must recognise them or the kernel crashes inside parseReg.
  // VCC_HI_SCRATCH / EXEC_HI_SCRATCH: on a WAVE32 source, hardware VCC and
  // EXEC are only 32 bits (== VCC_LO / EXEC_LO), so the registers named VCC_HI
  // and EXEC_HI are free general-purpose scalars the compiler uses as scratch
  // They must not alias the (target wave64) VCC/EXEC wave masks, or a `v_cmp`
  // writing VCC (or an EXEC update) would clobber the kernel's scratch value.
  // Each is modelled as its own i32 scalar slot.
  enum Kind {
    SGPR,
    VGPR,
    AGPR,
    VCC,
    EXEC,
    SCC,
    MODE,
    M0,
    FLAT_SCR,
    TTMP,
    LDS_DIRECT,
    SRC_VCCZ,
    SRC_EXECZ,
    SRC_SCC,
    VCC_HI_SCRATCH,
    EXEC_HI_SCRATCH,
    NOREG,
    OTHER
  };
  Kind RegKind = OTHER;
  std::optional<unsigned> BaseIdx;
  // The width of the register in 32-bit double words, 1 means 32 bit, 2 means
  // 64 bit.
  uint8_t WidthInDwords = 1;
};

} // namespace COMGR::hotswap

#endif
