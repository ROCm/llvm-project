//===- comgr-hotswap-tool-detect.h - gfx target + A0 gate helpers ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Header-only hotswap detection helpers (no HSA dep; ELF.h is header-only),
// split out so they can be unit-tested.

#ifndef COMGR_HOTSWAP_TOOL_DETECT_H
#define COMGR_HOTSWAP_TOOL_DETECT_H

#include "llvm/BinaryFormat/ELF.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace COMGR::hotswap {

// Extract the gfx target (e.g. "gfx1250") from an ISA name, dropping any suffix.
inline std::string extractGfxTarget(const std::string &IsaName) {
  const size_t Start = IsaName.find("gfx");
  if (Start == std::string::npos) {
    return {};
  }
  size_t End = Start;
  while (End < IsaName.size() &&
         std::isalnum(static_cast<unsigned char>(IsaName[End]))) {
    ++End;
  }
  return IsaName.substr(Start, End - Start);
}

// Arm only on gfx1250 A0 (revision 0); RevisionValid rejects a failed query.
inline bool gateAllowsHotswap(const std::string &Gfx, uint32_t Revision,
                              bool RevisionValid) {
  return RevisionValid && Gfx == "gfx1250" && Revision == 0;
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
