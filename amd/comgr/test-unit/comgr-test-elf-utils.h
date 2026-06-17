//===- comgr-test-elf-utils.h - shared ELF builders for unit tests --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TEST_UNIT_ELF_UTILS_H
#define COMGR_TEST_UNIT_ELF_UTILS_H

#include "llvm/BinaryFormat/ELF.h"

#include <cstdint>

namespace comgr_test {

// A zero-initialized 64-bit little-endian ELF header with valid magic and the
// given machine + e_flags. Callers set any further fields they need.
inline llvm::ELF::Elf64_Ehdr makeElf64Ehdr(uint16_t Machine,
                                           uint32_t Flags = 0) {
  using namespace llvm::ELF;
  Elf64_Ehdr Ehdr{};
  Ehdr.e_ident[EI_MAG0] = 0x7f;
  Ehdr.e_ident[EI_MAG1] = 'E';
  Ehdr.e_ident[EI_MAG2] = 'L';
  Ehdr.e_ident[EI_MAG3] = 'F';
  Ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  Ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  Ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  Ehdr.e_machine = Machine;
  Ehdr.e_flags = Flags;
  return Ehdr;
}

} // namespace comgr_test

#endif // COMGR_TEST_UNIT_ELF_UTILS_H
