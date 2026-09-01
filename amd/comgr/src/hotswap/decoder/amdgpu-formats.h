//===- amdgpu-formats.h - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_AMDGPU_FORMATS_H
#define HOTSWAP_TRANSPILER_AMDGPU_FORMATS_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace COMGR::hotswap {

// The bits an MC `TSFlags` value sets to name the format an instruction is
// encoded in, mirroring the instruction-format section of SIInstrFlags in
// llvm/lib/Target/AMDGPU/SIDefines.h. SIDefines.h is a backend-private header
// (not installed), and upstream walls off the raw bit constants
// (SIInstrFlags::DontUseRawTSFlags) behind predicate functions taking an
// MCInst/MCInstrDesc, which a stored TSFlags value does not have. So duplicate
// the bits here (same idiom as patch-wmma-hazard.cpp). Keep in sync if the
// TSFlags layout changes.
//
// These name encodings. The remaining SIInstrFlags bits name a property an
// instruction has rather than the form it is encoded in -- an MFMA sets IsMAI
// on top of the VOP3 bit that encodes it -- and belong to whoever needs the
// property. Encodings nest as well, VOP3P setting VOP3 and a VOP3P DPP variant
// setting both DPP and VOP3P, so a consumer testing several bits orders the
// specific before the general.
namespace AmdgpuFormat {
enum : uint64_t {
  SOP1 = UINT64_C(1) << 2,
  SOP2 = UINT64_C(1) << 3,
  SOPC = UINT64_C(1) << 4,
  SOPK = UINT64_C(1) << 5,
  SOPP = UINT64_C(1) << 6,
  VOP1 = UINT64_C(1) << 7,
  VOP2 = UINT64_C(1) << 8,
  VOPC = UINT64_C(1) << 9,
  VOP3 = UINT64_C(1) << 10,
  VOP3P = UINT64_C(1) << 12,
  SDWA = UINT64_C(1) << 14,
  DPP = UINT64_C(1) << 15,
  MUBUF = UINT64_C(1) << 17,
  SMRD = UINT64_C(1) << 19,
  VIMAGE = UINT64_C(1) << 21,
  FLAT = UINT64_C(1) << 24,
  DS = UINT64_C(1) << 25,
  VOPD3 = UINT64_C(1) << 30,
};
} // namespace AmdgpuFormat

// The instruction-family label (e.g. "SOP1", "MFMA", "FLAT") for an instruction
// with the given MC `TSFlags`, or "Unknown". Names the family a reader would
// recognize rather than the encoding alone, so it is for diagnostics; a
// consumer that has to act on the format tests the AmdgpuFormat bits.
llvm::StringRef formatName(uint64_t TSFlags);

} // namespace COMGR::hotswap

#endif
