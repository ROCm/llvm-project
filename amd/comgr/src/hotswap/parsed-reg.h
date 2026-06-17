//===- parsed-reg.h - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_PARSED_REG_H
#define HOTSWAP_TRANSPILER_PARSED_REG_H

namespace COMGR::hotswap {

struct ISAProfile;  // forward declaration

struct ParsedReg {
  // Compact-predicate "source-only" registers (SIRegisterInfo.td:198-200):
  //   SRC_VCCZ : i1 == (VCC == 0)   read as i32 in VOP/SOP src slots
  //   SRC_EXECZ: i1 == (EXEC == 0)
  //   SRC_SCC  : i1 == SCC
  // These cannot be written. We model them as their own kinds so
  // readOp32 / readOp64 can materialise the boolean result on demand
  // (see raise-context.cpp). Tensile gfx1250 emits them as F16 source
  // operands (e.g. `v_sub_f16 v64, src_vccz, v48`), so the dispatch
  // path must recognise them or the kernel crashes inside parseReg.
  // VCC_HI_SCRATCH: on a wave32 source, hardware VCC is only 32 bits
  // (== VCC_LO), so the register named VCC_HI is a free general-purpose
  // scalar the compiler uses as scratch. It must not alias the (wave64
  // target) VCC wave mask, or a `v_cmp` writing VCC would clobber it.
  // Modelled as its own i32 scalar slot.
  enum Kind { SGPR, VGPR, AGPR, VCC, EXEC, SCC, MODE, M0, FLAT_SCR, TTMP,
              LDS_DIRECT, SRC_VCCZ, SRC_EXECZ, SRC_SCC, VCC_HI_SCRATCH,
              NOREG, OTHER };
  Kind RegKind = OTHER;
  int BaseIdx = -1;
  int Width = 1;
};

} // namespace COMGR::hotswap

#endif
