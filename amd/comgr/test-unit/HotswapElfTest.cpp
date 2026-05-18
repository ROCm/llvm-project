//===- HotswapElfTest.cpp - Unit tests for HotSwap ELF layer --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace COMGR::hotswap;

// File-scope helpers for ELF construction in the tests below.

// Lay out one Elf64_Shdr at \p Sh with the named fields. Fields not
// relevant to the parse path (sh_addr, sh_info, sh_addralign) stay zero.
static void writeShdr(uint8_t *Sh, uint32_t Name, uint32_t Type, uint64_t Flags,
                      uint64_t Off, uint64_t Sz, uint32_t Link,
                      uint64_t EntSize) {
  std::memcpy(Sh + 0, &Name, 4);
  std::memcpy(Sh + 4, &Type, 4);
  std::memcpy(Sh + 8, &Flags, 8);
  std::memcpy(Sh + 24, &Off, 8);
  std::memcpy(Sh + 32, &Sz, 8);
  std::memcpy(Sh + 40, &Link, 4);
  std::memcpy(Sh + 56, &EntSize, 8);
}

// -- ElfView::create ----------------------------------------------------------

TEST(ElfView, RejectsTruncatedInput) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Garbage, sizeof(Garbage));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

TEST(ElfView, RejectsNonElfInput) {
  uint8_t NotElf[64] = {};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(NotElf, sizeof(NotElf));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

// -- ElfView::getKernelLdsSize ------------------------------------------------
//
// getKernelLdsSize reads group_segment_fixed_size from a kernel descriptor
// symbol "<KernelName>.kd". Two unit tests cover the helper:
//   * negative path: no .kd symbol -> std::nullopt
//   * positive path: hand-crafted ELF with a .kd symbol pointing at an
//                    embedded kernel descriptor -> the embedded LDS size
// Real gfx1250 code-object coverage is added by the lit tests in #2302.

TEST(ElfView, GetKernelLdsSizeReturnsNulloptWhenKdMissing) {
  // Build a minimal valid ELF64: header + .text + .shstrtab. ELFFile::create
  // succeeds, but no .kd symbol exists, so getKernelLdsSize must take the
  // missing-KD branch.
  static constexpr size_t BufSize = 512;
  alignas(8) uint8_t Buf[BufSize] = {};

  Buf[0] = 0x7f;
  Buf[1] = 'E';
  Buf[2] = 'L';
  Buf[3] = 'F';
  Buf[4] = llvm::ELF::ELFCLASS64;
  Buf[5] = llvm::ELF::ELFDATA2LSB;
  Buf[6] = llvm::ELF::EV_CURRENT;
  Buf[7] = llvm::ELF::ELFOSABI_AMDGPU_HSA;

  uint16_t EType = llvm::ELF::ET_REL;
  std::memcpy(Buf + 16, &EType, 2);
  uint16_t EMachine = llvm::ELF::EM_AMDGPU;
  std::memcpy(Buf + 18, &EMachine, 2);
  uint32_t EVersion = llvm::ELF::EV_CURRENT;
  std::memcpy(Buf + 20, &EVersion, 4);

  uint16_t EhSize = 64;
  std::memcpy(Buf + 52, &EhSize, 2);
  uint16_t ShEntSize = 64;
  std::memcpy(Buf + 58, &ShEntSize, 2);
  uint16_t ShNum = 3;
  std::memcpy(Buf + 60, &ShNum, 2);
  uint16_t ShStrNdx = 2;
  std::memcpy(Buf + 62, &ShStrNdx, 2);

  static constexpr uint64_t StrTabOff = 256;
  const char StrTab[] = "\0.text\0.shstrtab\0";
  std::memcpy(Buf + StrTabOff, StrTab, sizeof(StrTab));

  static constexpr uint64_t TextOff = 320;
  static constexpr uint64_t TextSize = 16;
  static constexpr uint64_t ShOff = 64;
  uint64_t ShOffVal = ShOff;
  std::memcpy(Buf + 40, &ShOffVal, 8);

  // Shdr[1] = .text
  uint8_t *Sh1 = Buf + ShOff + 64;
  uint32_t ShName1 = 1;
  std::memcpy(Sh1 + 0, &ShName1, 4);
  uint32_t ShType1 = llvm::ELF::SHT_PROGBITS;
  std::memcpy(Sh1 + 4, &ShType1, 4);
  uint64_t ShFlags1 = llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR;
  std::memcpy(Sh1 + 8, &ShFlags1, 8);
  uint64_t ShOff1 = TextOff;
  std::memcpy(Sh1 + 24, &ShOff1, 8);
  uint64_t ShSize1 = TextSize;
  std::memcpy(Sh1 + 32, &ShSize1, 8);

  // Shdr[2] = .shstrtab
  uint8_t *Sh2 = Buf + ShOff + 128;
  uint32_t ShName2 = 7;
  std::memcpy(Sh2 + 0, &ShName2, 4);
  uint32_t ShType2 = llvm::ELF::SHT_STRTAB;
  std::memcpy(Sh2 + 4, &ShType2, 4);
  uint64_t ShOff2 = StrTabOff;
  std::memcpy(Sh2 + 24, &ShOff2, 8);
  uint64_t ShSize2 = sizeof(StrTab);
  std::memcpy(Sh2 + 32, &ShSize2, 8);

  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Buf, BufSize);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelLdsSize("nonexistent_kernel"), std::nullopt);
}

TEST(ElfView, GetKernelLdsSizeReadsLdsSizeFromKernelDescriptor) {
  // Build a minimal AMDGPU ELF64 with the section topology that
  // findKernelDescriptor walks: 6 sections (NULL, .text, .rodata, .strtab,
  // .symtab, .shstrtab). The kernel descriptor is embedded at the start of
  // .rodata with a known group_segment_fixed_size value, and a symbol named
  // "test_kernel.kd" in .symtab points at it. getKernelLdsSize must return
  // the embedded LDS size unchanged.
  static constexpr size_t BufSize = 1024;
  alignas(8) uint8_t Buf[BufSize] = {};

  // Section file offsets and sizes. Layout choices keep each section
  // 8-byte aligned so the ELF parser is happy.
  static constexpr uint64_t ShOff = 64;
  static constexpr uint64_t TextOff = 0x1C0;
  static constexpr uint64_t TextSize = 16;
  static constexpr uint64_t RodataOff = 0x1D0;
  static constexpr uint64_t KdSize = 64;
  static constexpr uint64_t StrTabOff = 0x210;
  static constexpr uint64_t SymTabOff = 0x220;
  static constexpr uint64_t SymEntSize = 24;
  static constexpr uint64_t SymCount = 2;
  static constexpr uint64_t ShStrTabOff = 0x250;
  static constexpr uint16_t ShNum = 6;
  static constexpr uint16_t ShStrNdx = 5;
  static constexpr uint32_t TestLdsSize = 16384;

  // ELF header.
  Buf[0] = 0x7f;
  Buf[1] = 'E';
  Buf[2] = 'L';
  Buf[3] = 'F';
  Buf[4] = llvm::ELF::ELFCLASS64;
  Buf[5] = llvm::ELF::ELFDATA2LSB;
  Buf[6] = llvm::ELF::EV_CURRENT;
  Buf[7] = llvm::ELF::ELFOSABI_AMDGPU_HSA;

  uint16_t EType = llvm::ELF::ET_REL;
  std::memcpy(Buf + 16, &EType, 2);
  uint16_t EMachine = llvm::ELF::EM_AMDGPU;
  std::memcpy(Buf + 18, &EMachine, 2);
  uint32_t EVersion = llvm::ELF::EV_CURRENT;
  std::memcpy(Buf + 20, &EVersion, 4);
  uint64_t ShOffVal = ShOff;
  std::memcpy(Buf + 40, &ShOffVal, 8);
  uint16_t EhSize = 64;
  std::memcpy(Buf + 52, &EhSize, 2);
  uint16_t ShEntSize = 64;
  std::memcpy(Buf + 58, &ShEntSize, 2);
  uint16_t ShNumVal = ShNum;
  std::memcpy(Buf + 60, &ShNumVal, 2);
  uint16_t ShStrNdxVal = ShStrNdx;
  std::memcpy(Buf + 62, &ShStrNdxVal, 2);

  // Section name string table. Entries: "" .text .rodata .strtab .symtab
  // .shstrtab. Offsets pinned in the shdr writes below.
  const char ShStrTab[] = "\0.text\0.rodata\0.strtab\0.symtab\0.shstrtab\0";
  std::memcpy(Buf + ShStrTabOff, ShStrTab, sizeof(ShStrTab));

  // Symbol name string table. Single named symbol "test_kernel.kd" at
  // offset 1; offset 0 is the conventional empty name.
  const char StrTab[] = "\0test_kernel.kd\0";
  std::memcpy(Buf + StrTabOff, StrTab, sizeof(StrTab));

  // Section header table.
  uint8_t *Sh = Buf + ShOff;
  // Shdr[0] = NULL section (already zeroed).
  // Shdr[1] = .text, name offset 1.
  writeShdr(Sh + 1 * 64, 1, llvm::ELF::SHT_PROGBITS,
            llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR, TextOff, TextSize,
            0, 0);
  // Shdr[2] = .rodata, name offset 7.
  writeShdr(Sh + 2 * 64, 7, llvm::ELF::SHT_PROGBITS, llvm::ELF::SHF_ALLOC,
            RodataOff, KdSize, 0, 0);
  // Shdr[3] = .strtab, name offset 15.
  writeShdr(Sh + 3 * 64, 15, llvm::ELF::SHT_STRTAB, 0, StrTabOff,
            sizeof(StrTab), 0, 0);
  // Shdr[4] = .symtab, name offset 23; sh_link = 3 (.strtab), sh_entsize = 24
  // (Elf64_Sym).
  writeShdr(Sh + 4 * 64, 23, llvm::ELF::SHT_SYMTAB, 0, SymTabOff,
            SymEntSize * SymCount, 3, SymEntSize);
  // Shdr[5] = .shstrtab, name offset 31.
  writeShdr(Sh + 5 * 64, 31, llvm::ELF::SHT_STRTAB, 0, ShStrTabOff,
            sizeof(ShStrTab), 0, 0);

  // Kernel descriptor body: group_segment_fixed_size at offset 0. The rest
  // of the 64-byte descriptor stays zero, which is fine for a read-only
  // helper that only consumes one field.
  std::memcpy(Buf + RodataOff, &TestLdsSize, sizeof(TestLdsSize));

  // Symbol table. Slot 0 is the conventional null symbol. Slot 1 names
  // "test_kernel.kd" (st_name=1), binding STB_GLOBAL + type STT_OBJECT in
  // st_info, shndx=2 (.rodata), st_value=0 (start of .rodata), st_size=64.
  uint8_t *Sym1 = Buf + SymTabOff + SymEntSize;
  uint32_t StName = 1;
  std::memcpy(Sym1 + 0, &StName, 4);
  Sym1[4] = (llvm::ELF::STB_GLOBAL << 4) | llvm::ELF::STT_OBJECT;
  uint16_t StShndx = 2;
  std::memcpy(Sym1 + 6, &StShndx, 2);
  uint64_t StSize = KdSize;
  std::memcpy(Sym1 + 16, &StSize, 8);

  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Buf, BufSize);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::optional<uint32_t> Lds = ViewOrErr->getKernelLdsSize("test_kernel");
  ASSERT_TRUE(Lds.has_value());
  EXPECT_EQ(*Lds, TestLdsSize);
}
