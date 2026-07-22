//===- amdgpu-formats.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amdgpu-formats.h"

#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"

namespace COMGR::hotswap {

llvm::StringRef formatName(uint64_t TSFlags, unsigned Opcode) {
  using namespace llvm;
  // VOPD has no dedicated TSFlags bit; MAI is a VOP3 subclass and VOP3P
  // coexists with VOP3, so the more specific tests must come first.
  if (AMDGPU::isVOPD(Opcode))
    return "VOPD";
  if (TSFlags & SIInstrFlags::IsMAI)
    return "MFMA";
  if (TSFlags & SIInstrFlags::DPP)
    return "DPP";
  if (TSFlags & SIInstrFlags::SDWA)
    return "SDWA";
  if (TSFlags & SIInstrFlags::SOPP)
    return "SOPP";
  if (TSFlags & SIInstrFlags::SOPC)
    return "SOPC";
  if (TSFlags & SIInstrFlags::SOP1)
    return "SOP1";
  if (TSFlags & SIInstrFlags::SOP2)
    return "SOP2";
  if (TSFlags & SIInstrFlags::SOPK)
    return "SOPK";
  if (TSFlags & SIInstrFlags::VOPC)
    return "VOPC";
  if (TSFlags & SIInstrFlags::VOP3P)
    return "VOP3P";
  if (TSFlags & SIInstrFlags::VOP3)
    return "VOP3";
  if (TSFlags & SIInstrFlags::VOP2)
    return "VOP2";
  if (TSFlags & SIInstrFlags::VOP1)
    return "VOP1";
  if (TSFlags & SIInstrFlags::SMRD)
    return "SMEM";
  if (TSFlags & SIInstrFlags::FLAT)
    return "FLAT";
  if (TSFlags & SIInstrFlags::MUBUF)
    return "MUBUF";
  if (TSFlags & SIInstrFlags::DS)
    return "DS";
  if (TSFlags & SIInstrFlags::VIMAGE)
    return "VIMAGE";
  // The gfx1250 TENSOR pseudos set TENSOR_CNT without the VIMAGE bit.
  if (TSFlags & SIInstrFlags::TENSOR_CNT)
    return "VIMAGE";
  return "Unknown";
}

} // namespace COMGR::hotswap
