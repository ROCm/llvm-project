//===- comgr-hotswap-tool-detect.h - gfx target + A0 gate helpers ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hotswap detection helpers (no HSA dep), split out so they can be unit-tested.

#ifndef COMGR_HOTSWAP_TOOL_DETECT_H
#define COMGR_HOTSWAP_TOOL_DETECT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace COMGR::hotswap {

// Processor of an ISA name like amdgcn-amd-amdhsa--gfx1250[:feats] (the field
// after arch-vendor-os-environ; same model as comgr's parseTargetIdentifier).
inline std::string extractGfxTarget(llvm::StringRef IsaName) {
  llvm::StringRef Rest = IsaName;
  for (int I = 0; I < 4; ++I) {
    Rest = Rest.split('-').second;
  }
  return Rest.split(':').first.str();
}

// Arm only on gfx1250 at ASIC revision A0 (0). Callers must confirm the revision
// query succeeded before calling; a failed query is handled at the call site.
inline bool gateAllowsHotswap(const std::string &Gfx, uint32_t Revision) {
  return Gfx == "gfx1250" && Revision == 0;
}

// True for a 64-bit gfx1250 AMDGPU ELF (aligned-copy header read, e_machine checked).
inline bool isGfx1250CodeObject(const void *Data, size_t Size) {
  if (!Data || Size < sizeof(llvm::ELF::Elf64_Ehdr)) {
    return false;
  }
  llvm::ELF::Elf64_Ehdr Header;
  std::memcpy(&Header, Data, sizeof(Header));
  return Header.checkMagic() &&
         Header.getFileClass() == llvm::ELF::ELFCLASS64 &&
         Header.e_machine == llvm::ELF::EM_AMDGPU &&
         (Header.e_flags & llvm::ELF::EF_AMDGPU_MACH) ==
             llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1250;
}

} // namespace COMGR::hotswap

#endif // COMGR_HOTSWAP_TOOL_DETECT_H
