//===- amdgpu-formats.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/decoder/amdgpu-formats.h"

namespace COMGR::hotswap {

// Two SIInstrFlags bits that refine the reported name without naming a format,
// so they stay private here rather than joining AmdgpuFormat: IsMAI marks an
// MFMA, which is VOP3-encoded, and TENSOR_CNT marks an instruction counted
// against the tensor wait counter. Mirrored from SIDefines.h under the same
// keep-in-sync caveat as the format bits.
static constexpr uint64_t IsMAI = UINT64_C(1) << 54;
static constexpr uint64_t TENSOR_CNT = UINT64_C(1) << 38;

llvm::StringRef formatName(uint64_t TSFlags) {
  using namespace AmdgpuFormat;
  // Only the gfx1250 VOPD3 form carries a TSFlags bit; classic (gfx11) VOPD has
  // none, but the transpiler's target ISAs do not emit it. MAI is a VOP3
  // subclass and VOP3P coexists with VOP3, so the more specific tests come
  // first.
  if (TSFlags & VOPD3) {
    return "VOPD";
  }
  if (TSFlags & IsMAI) {
    return "MFMA";
  }
  if (TSFlags & DPP) {
    return "DPP";
  }
  if (TSFlags & SDWA) {
    return "SDWA";
  }
  if (TSFlags & SOPP) {
    return "SOPP";
  }
  if (TSFlags & SOPC) {
    return "SOPC";
  }
  if (TSFlags & SOP1) {
    return "SOP1";
  }
  if (TSFlags & SOP2) {
    return "SOP2";
  }
  if (TSFlags & SOPK) {
    return "SOPK";
  }
  if (TSFlags & VOPC) {
    return "VOPC";
  }
  if (TSFlags & VOP3P) {
    return "VOP3P";
  }
  if (TSFlags & VOP3) {
    return "VOP3";
  }
  if (TSFlags & VOP2) {
    return "VOP2";
  }
  if (TSFlags & VOP1) {
    return "VOP1";
  }
  if (TSFlags & SMRD) {
    return "SMEM";
  }
  if (TSFlags & FLAT) {
    return "FLAT";
  }
  if (TSFlags & MUBUF) {
    return "MUBUF";
  }
  if (TSFlags & DS) {
    return "DS";
  }
  if (TSFlags & VIMAGE) {
    return "VIMAGE";
  }
  // The gfx1250 TENSOR pseudos set TENSOR_CNT without the VIMAGE bit.
  if (TSFlags & TENSOR_CNT) {
    return "VIMAGE";
  }
  return "Unknown";
}

} // namespace COMGR::hotswap
