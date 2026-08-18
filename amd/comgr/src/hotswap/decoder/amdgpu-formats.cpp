//===- amdgpu-formats.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/decoder/amdgpu-formats.h"

namespace COMGR::hotswap {

// Mirrors SIInstrFlags bit positions from llvm/lib/Target/AMDGPU/SIDefines.h.
// SIDefines.h is a backend-private header (not installed), and upstream now
// walls off the raw bit constants (SIInstrFlags::DontUseRawTSFlags) behind
// predicate functions that require an MCInst/MCInstrDesc — which formatName's
// raw-uint64_t signature doesn't have. So duplicate the bits here (same idiom
// as patch-wmma-hazard.cpp). Keep in sync if the TSFlags layout changes.
namespace AmdgpuTSFlags {
static constexpr uint64_t SOP1 = UINT64_C(1) << 2;
static constexpr uint64_t SOP2 = UINT64_C(1) << 3;
static constexpr uint64_t SOPC = UINT64_C(1) << 4;
static constexpr uint64_t SOPK = UINT64_C(1) << 5;
static constexpr uint64_t SOPP = UINT64_C(1) << 6;
static constexpr uint64_t VOP1 = UINT64_C(1) << 7;
static constexpr uint64_t VOP2 = UINT64_C(1) << 8;
static constexpr uint64_t VOPC = UINT64_C(1) << 9;
static constexpr uint64_t VOP3 = UINT64_C(1) << 10;
static constexpr uint64_t VOP3P = UINT64_C(1) << 12;
static constexpr uint64_t SDWA = UINT64_C(1) << 14;
static constexpr uint64_t DPP = UINT64_C(1) << 15;
static constexpr uint64_t MUBUF = UINT64_C(1) << 17;
static constexpr uint64_t SMRD = UINT64_C(1) << 19;
static constexpr uint64_t VIMAGE = UINT64_C(1) << 21;
static constexpr uint64_t FLAT = UINT64_C(1) << 24;
static constexpr uint64_t DS = UINT64_C(1) << 25;
static constexpr uint64_t VOPD3 = UINT64_C(1) << 30;
static constexpr uint64_t TENSOR_CNT = UINT64_C(1) << 38;
static constexpr uint64_t IsMAI = UINT64_C(1) << 54;
} // namespace AmdgpuTSFlags

llvm::StringRef formatName(uint64_t TSFlags) {
  using namespace AmdgpuTSFlags;
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
