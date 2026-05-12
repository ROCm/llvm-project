//===- amdgpu-formats.h - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_AMDGPU_FORMATS_H
#define HOTSWAP_TRANSPILER_AMDGPU_FORMATS_H

#include "SIDefines.h"            // llvm::SIInstrFlags
#include "Utils/AMDGPUBaseInfo.h" // llvm::AMDGPU::isVOPD

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace COMGR::hotswap {

namespace SIInstrFlags = llvm::SIInstrFlags;

/// Return a human-readable AMDGPU encoding-format label for `Opc` with
/// disassembler `Flags`. The label is consumed only by diagnostics
/// (no runtime dispatch keys on the string). Precedence below mirrors
/// LLVM's own decoder: `IsMAI` is a VOP3 subclass; `DPP` / `SDWA`
/// coexist with VOP1/VOP2/VOPC and must be checked first; `VOP3P`
/// coexists with `VOP3`; VOPD has no dedicated TSFlags bit (LLVM's
/// VOPD3 bit varies across generations) so use `AMDGPU::isVOPD(Opc)`.
///
/// Returns an `llvm::Error` if no known format bit matches -- callers
/// surface the message verbatim in unsupported-instruction
/// diagnostics.
inline llvm::Expected<llvm::StringRef> formatName(uint64_t Flags,
                                                  unsigned Opc) {
  if (llvm::AMDGPU::isVOPD(Opc))     return llvm::StringRef("VOPD");
  if (Flags & SIInstrFlags::IsMAI)   return llvm::StringRef("MFMA");
  if (Flags & SIInstrFlags::DPP)     return llvm::StringRef("DPP");
  if (Flags & SIInstrFlags::SDWA)    return llvm::StringRef("SDWA");
  if (Flags & SIInstrFlags::SOPP)    return llvm::StringRef("SOPP");
  if (Flags & SIInstrFlags::SOPC)    return llvm::StringRef("SOPC");
  if (Flags & SIInstrFlags::SOP1)    return llvm::StringRef("SOP1");
  if (Flags & SIInstrFlags::SOP2)    return llvm::StringRef("SOP2");
  if (Flags & SIInstrFlags::SOPK)    return llvm::StringRef("SOPK");
  if (Flags & SIInstrFlags::VOPC)    return llvm::StringRef("VOPC");
  if (Flags & SIInstrFlags::VOP3P)   return llvm::StringRef("VOP3P");
  if (Flags & SIInstrFlags::VOP3)    return llvm::StringRef("VOP3");
  if (Flags & SIInstrFlags::VOP2)    return llvm::StringRef("VOP2");
  if (Flags & SIInstrFlags::VOP1)    return llvm::StringRef("VOP1");
  if (Flags & SIInstrFlags::SMRD)    return llvm::StringRef("SMEM");
  if (Flags & SIInstrFlags::FLAT)    return llvm::StringRef("FLAT");
  if (Flags & SIInstrFlags::MUBUF)   return llvm::StringRef("MUBUF");
  if (Flags & SIInstrFlags::DS)      return llvm::StringRef("DS");
  // VIMAGE: gfx12+ vector image / tensor encoding family. Pure-image
  // members carry `SIInstrFlags::VIMAGE` directly; the gfx1250 TENSOR
  // pseudos extend `InstSI` directly and only set `VALU = 1` and
  // `TENSOR_CNT = 1`, so the `VIMAGE` field stays 0. Detect them via
  // the `TENSOR_CNT` bit instead. The only other carrier of that bit
  // is `s_wait_tensorcnt` (SOPP), already matched by the SOPP arm
  // above. Both arms produce `"VIMAGE"` so `kerneldex` / `raise_cli`
  // bucket cross-target failures uniformly.
  if (Flags & SIInstrFlags::VIMAGE)     return llvm::StringRef("VIMAGE");
  if (Flags & SIInstrFlags::TENSOR_CNT) return llvm::StringRef("VIMAGE");
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "no known AMDGPU format flag for opcode");
}

} // namespace COMGR::hotswap

#endif
