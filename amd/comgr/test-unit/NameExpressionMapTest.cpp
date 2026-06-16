//===- NameExpressionMapTest.cpp - Unit tests for name expression maps ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression tests for amd_comgr_populate_name_expression_map() on the code
// object (AMD_COMGR_DATA_KIND_EXECUTABLE) path. The implementation locates the
// .dynsym, .rela.dyn, and .rodata sections by scanning the section header
// table; if a required section is absent it must report a clean status rather
// than dereferencing an unset section header (which previously fed garbage
// sh_size / sh_entsize values into the ELF accessors).
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// These tests craft raw ELF objects by hand instead of pulling in
// llvm/BinaryFormat/ELF.h. amd_comgr hides all of its statically-linked LLVM
// symbols via the version script (DemangleSymbolNameTest sets
// GTEST_NO_LLVM_SUPPORT for the same reason), so the test binary cannot rely on
// LLVM Support/Object symbols at link time. Self-contained POD definitions keep
// the test hermetic.
namespace {

// System V ELF64 little-endian on-disk structures.
struct Elf64Ehdr {
  uint8_t e_ident[16];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
  uint64_t e_entry;
  uint64_t e_phoff;
  uint64_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
};
static_assert(sizeof(Elf64Ehdr) == 64, "Elf64Ehdr must be 64 bytes");

struct Elf64Shdr {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  uint64_t sh_addr;
  uint64_t sh_offset;
  uint64_t sh_size;
  uint32_t sh_link;
  uint32_t sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
};
static_assert(sizeof(Elf64Shdr) == 64, "Elf64Shdr must be 64 bytes");

struct Elf64Sym {
  uint32_t st_name;
  uint8_t st_info;
  uint8_t st_other;
  uint16_t st_shndx;
  uint64_t st_value;
  uint64_t st_size;
};
static_assert(sizeof(Elf64Sym) == 24, "Elf64Sym must be 24 bytes");

enum {
  ElfClass64 = 2,
  ElfData2Lsb = 1,
  ElfVerCurrent = 1,
  ElfTypeDyn = 3,
  ElfMachineAmdgpu = 224,
  ElfShtStrtab = 3,
  ElfShtDynsym = 11,
  ElfStbGlobal = 1,
  ElfSttFunc = 2,
};

template <typename T> void append(std::vector<char> &Buf, const T &Val) {
  const char *P = reinterpret_cast<const char *>(&Val);
  Buf.insert(Buf.end(), P, P + sizeof(T));
}

// Build a minimal little-endian ELF64 executable. When WithDynsym is true a
// .dynsym / .dynstr pair is emitted; if SymName is non-empty a single global
// symbol with that name is added. No .rela.dyn or .rodata sections are ever
// emitted, which is exactly the shape the section guards must handle.
std::vector<char> buildExecutable(bool WithDynsym, const std::string &SymName) {
  // Section header string table; index 0 is the empty name.
  std::string ShStrtab(1, '\0');
  auto addShName = [&](const char *Name) -> uint32_t {
    uint32_t Off = static_cast<uint32_t>(ShStrtab.size());
    ShStrtab += Name;
    ShStrtab.push_back('\0');
    return Off;
  };
  uint32_t DynsymNameOff = 0, DynstrNameOff = 0;
  if (WithDynsym) {
    DynsymNameOff = addShName(".dynsym");
    DynstrNameOff = addShName(".dynstr");
  }
  uint32_t ShStrtabNameOff = addShName(".shstrtab");

  // Dynamic symbol and string tables.
  std::string DynStr(1, '\0');
  std::vector<Elf64Sym> Syms;
  Syms.emplace_back(); // reserved null symbol (index 0)
  if (WithDynsym && !SymName.empty()) {
    Elf64Sym S{};
    S.st_name = static_cast<uint32_t>(DynStr.size());
    S.st_info = (ElfStbGlobal << 4) | ElfSttFunc;
    S.st_shndx = 1;
    S.st_value = 0x1000;
    Syms.push_back(S);
    DynStr += SymName;
    DynStr.push_back('\0');
  }

  // Lay out section contents directly after the ELF header; the section header
  // table follows the contents.
  std::vector<char> Body;
  const uint64_t BodyBase = sizeof(Elf64Ehdr);

  uint64_t DynsymOff = 0, DynsymSize = 0, DynstrOff = 0, DynstrSize = 0;
  if (WithDynsym) {
    DynsymOff = BodyBase + Body.size();
    for (const Elf64Sym &S : Syms)
      append(Body, S);
    DynsymSize = Syms.size() * sizeof(Elf64Sym);

    DynstrOff = BodyBase + Body.size();
    Body.insert(Body.end(), DynStr.begin(), DynStr.end());
    DynstrSize = DynStr.size();
  }

  uint64_t ShStrtabOff = BodyBase + Body.size();
  Body.insert(Body.end(), ShStrtab.begin(), ShStrtab.end());
  uint64_t ShStrtabSize = ShStrtab.size();

  // Align the section header table to 8 bytes.
  while ((BodyBase + Body.size()) % 8)
    Body.push_back('\0');
  uint64_t ShOff = BodyBase + Body.size();

  // Section header table. Index 0 is always SHT_NULL.
  std::vector<Elf64Shdr> Sh(1); // zero-initialized null section
  uint16_t ShStrNdx = 0;
  if (WithDynsym) {
    Elf64Shdr Dynsym{};
    Dynsym.sh_name = DynsymNameOff;
    Dynsym.sh_type = ElfShtDynsym;
    Dynsym.sh_offset = DynsymOff;
    Dynsym.sh_size = DynsymSize;
    Dynsym.sh_link = 2; // .dynstr is section index 2
    Dynsym.sh_info = 1; // index of first non-local symbol
    Dynsym.sh_addralign = 8;
    Dynsym.sh_entsize = sizeof(Elf64Sym);
    Sh.push_back(Dynsym);

    Elf64Shdr Dynstr{};
    Dynstr.sh_name = DynstrNameOff;
    Dynstr.sh_type = ElfShtStrtab;
    Dynstr.sh_offset = DynstrOff;
    Dynstr.sh_size = DynstrSize;
    Dynstr.sh_addralign = 1;
    Sh.push_back(Dynstr);
  }

  Elf64Shdr ShStr{};
  ShStr.sh_name = ShStrtabNameOff;
  ShStr.sh_type = ElfShtStrtab;
  ShStr.sh_offset = ShStrtabOff;
  ShStr.sh_size = ShStrtabSize;
  ShStr.sh_addralign = 1;
  Sh.push_back(ShStr);
  ShStrNdx = static_cast<uint16_t>(Sh.size() - 1);

  Elf64Ehdr Eh{};
  Eh.e_ident[0] = 0x7f;
  Eh.e_ident[1] = 'E';
  Eh.e_ident[2] = 'L';
  Eh.e_ident[3] = 'F';
  Eh.e_ident[4] = ElfClass64;
  Eh.e_ident[5] = ElfData2Lsb;
  Eh.e_ident[6] = ElfVerCurrent;
  Eh.e_type = ElfTypeDyn;
  Eh.e_machine = ElfMachineAmdgpu;
  Eh.e_version = ElfVerCurrent;
  Eh.e_shoff = ShOff;
  Eh.e_ehsize = sizeof(Elf64Ehdr);
  Eh.e_shentsize = sizeof(Elf64Shdr);
  Eh.e_shnum = static_cast<uint16_t>(Sh.size());
  Eh.e_shstrndx = ShStrNdx;

  std::vector<char> Buf;
  append(Buf, Eh);
  Buf.insert(Buf.end(), Body.begin(), Body.end());
  for (const Elf64Shdr &S : Sh)
    append(Buf, S);
  return Buf;
}

amd_comgr_status_t populate(const std::vector<char> &Elf, size_t *Count) {
  amd_comgr_data_t Data;
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data));
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_set_data(Data, Elf.size(), Elf.data()));
  amd_comgr_status_t Status =
      amd_comgr_populate_name_expression_map(Data, Count);
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS, amd_comgr_release_data(Data));
  return Status;
}

} // namespace

// A code object with no .dynsym has no name-expression stubs; the map is empty
// and the call succeeds rather than reading an unset section header.
TEST(NameExpressionMapTest, NoDynsymReturnsEmpty) {
  std::vector<char> Elf = buildExecutable(/*WithDynsym=*/false, "");
  size_t Count = 123;
  ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS, populate(Elf, &Count));
  ASSERT_EQ(Count, 0u);
}

// A .dynsym without any __amdgcn_name_expr_ stub needs neither .rela.dyn nor
// .rodata; the call succeeds with an empty map.
TEST(NameExpressionMapTest, DynsymWithoutStubReturnsEmpty) {
  std::vector<char> Elf = buildExecutable(/*WithDynsym=*/true, "some_symbol");
  size_t Count = 123;
  ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS, populate(Elf, &Count));
  ASSERT_EQ(Count, 0u);
}

// A code object that carries an __amdgcn_name_expr_ stub but is missing the
// .rela.dyn section is malformed. The call must fail cleanly instead of reading
// an unset section header (the previous behavior surfaced as a bogus
// "RelaRange creation error: section [index <garbage>] has an invalid sh_size"
// from the ELF layer).
TEST(NameExpressionMapTest, StubWithoutRelaDynFailsCleanly) {
  std::vector<char> Elf =
      buildExecutable(/*WithDynsym=*/true, "__amdgcn_name_expr_0");
  size_t Count = 123;
  ASSERT_EQ(AMD_COMGR_STATUS_ERROR, populate(Elf, &Count));
}
