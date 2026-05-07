//===- CodeObjectUtilsTest.cpp - code-object-utils unit tests -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins the llvm::Error/llvm::Expected error-reporting contract for the
// public extractor functions in `hotswap/code-object-utils.h`:
//
//   * Hotswap-originated failures (no .text, no AMDGPU metadata note,
//     ".text base" / kernel symbol mismatches) return a `HotswapError`
//     ErrorInfo subclass with a recognizable message payload.
//
//   * Errors forwarded from `llvm::object::ObjectFile::createELFObjectFile`
//     and friends pass through unchanged: the returned `Error` is *not*
//     a `HotswapError`, so callers can `handleErrors` to tell the two
//     origins apart.
//
//===----------------------------------------------------------------------===//

#include "hotswap/code-object-utils.h"
#include "hotswap/hotswap-error.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Build a minimal valid AMDGPU 64-bit little-endian ELF in `Storage`,
// containing only SHN_UNDEF + a `.shstrtab` section. No `.text`, no
// `.rodata`, no notes. This is the smallest input that
// `llvm::object::ObjectFile::createELFObjectFile` accepts for the
// AMDGPU OS-ABI, so it lets us exercise the hotswap-originated failure
// branches (no `.text`, no metadata note) without dragging in
// `yaml2obj` from `llvm/ObjectYAML`.
//
// We rely on host endianness matching ELFDATA2LSB: every Comgr-supported
// build target (x86_64, aarch64) is little-endian, so the structs can
// be memcpy'd directly into the buffer. Tests that exercise this would
// have to be revisited if Comgr ever ships on a big-endian host, which
// it does not today.
std::vector<uint8_t> buildMinimalAmdgpuElf() {
  using namespace llvm::ELF;

  constexpr unsigned NumSections = 2;
  constexpr uint64_t HeaderSize = sizeof(Elf64_Ehdr);
  constexpr uint64_t SectionHeaderTableSize = NumSections * sizeof(Elf64_Shdr);
  const char ShStrTab[] = "\0.shstrtab";
  constexpr uint64_t ShStrTabSize = sizeof(ShStrTab);
  const uint64_t ShStrTabOffset = HeaderSize + SectionHeaderTableSize;

  std::vector<uint8_t> Buf(ShStrTabOffset + ShStrTabSize, 0);

  Elf64_Ehdr Ehdr{};
  Ehdr.e_ident[EI_MAG0] = ElfMagic[0];
  Ehdr.e_ident[EI_MAG1] = ElfMagic[1];
  Ehdr.e_ident[EI_MAG2] = ElfMagic[2];
  Ehdr.e_ident[EI_MAG3] = ElfMagic[3];
  Ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  Ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  Ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_DYN;
  Ehdr.e_machine = EM_AMDGPU;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_shoff = HeaderSize;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = NumSections;
  Ehdr.e_shstrndx = 1;
  std::memcpy(Buf.data(), &Ehdr, sizeof(Ehdr));

  Elf64_Shdr Shdrs[NumSections] = {};
  // [0] SHN_UNDEF: all-zero per the ELF spec.
  // [1] .shstrtab
  Shdrs[1].sh_name = 1; // offset of ".shstrtab" within shstrtab data
  Shdrs[1].sh_type = SHT_STRTAB;
  Shdrs[1].sh_offset = ShStrTabOffset;
  Shdrs[1].sh_size = ShStrTabSize;
  Shdrs[1].sh_addralign = 1;
  std::memcpy(Buf.data() + HeaderSize, Shdrs, SectionHeaderTableSize);

  std::memcpy(Buf.data() + ShStrTabOffset, ShStrTab, ShStrTabSize);
  return Buf;
}

llvm::MemoryBufferRef toBuf(const std::vector<uint8_t> &Bytes,
                            llvm::StringRef Name) {
  return llvm::MemoryBufferRef(
      llvm::StringRef(reinterpret_cast<const char *>(Bytes.data()),
                      Bytes.size()),
      Name);
}

// Returns the rendered `toString` of an Error, consuming it. The `bool`
// matches `Err.isA<HotswapError>()` evaluated *before* consumption.
struct InspectedError {
  bool IsHotswapError;
  std::string Message;
};

InspectedError inspect(llvm::Error Err) {
  InspectedError Out{Err.isA<COMGR::hotswap::HotswapError>(), ""};
  Out.Message = llvm::toString(std::move(Err));
  return Out;
}

} // namespace

TEST(CodeObjectUtils, KernelSymbolOffsetMalformedElfReturnsForwardedError) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  llvm::MemoryBufferRef Buf(
      llvm::StringRef(reinterpret_cast<const char *>(Garbage), sizeof(Garbage)),
      "garbage");

  llvm::Expected<uint64_t> Offset =
      COMGR::hotswap::findKernelSymbolOffset(Buf, "missing_kernel");

  ASSERT_FALSE(static_cast<bool>(Offset));
  InspectedError Err = inspect(Offset.takeError());
  EXPECT_FALSE(Err.IsHotswapError)
      << "ELF parse failures must propagate as the upstream "
         "llvm::object::ObjectFile ErrorInfo, not as HotswapError";
}

TEST(CodeObjectUtils, ExtractTextSectionMalformedElfReturnsForwardedError) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  llvm::MemoryBufferRef Buf(
      llvm::StringRef(reinterpret_cast<const char *>(Garbage), sizeof(Garbage)),
      "garbage");

  llvm::Expected<COMGR::hotswap::TextSection> Result =
      COMGR::hotswap::extractTextSection(Buf);

  ASSERT_FALSE(static_cast<bool>(Result));
  InspectedError Err = inspect(Result.takeError());
  EXPECT_FALSE(Err.IsHotswapError)
      << "ELF parse failures must propagate as the upstream ErrorInfo type";
}

TEST(CodeObjectUtils, ExtractTextSectionMissingTextReturnsHotswapError) {
  std::vector<uint8_t> Bytes = buildMinimalAmdgpuElf();
  llvm::Expected<COMGR::hotswap::TextSection> Result =
      COMGR::hotswap::extractTextSection(toBuf(Bytes, "no-text.elf"));

  ASSERT_FALSE(static_cast<bool>(Result));
  InspectedError Err = inspect(Result.takeError());
  EXPECT_TRUE(Err.IsHotswapError)
      << "missing .text is a hotswap-detected condition and must surface "
         "as HotswapError so callers can discriminate";
  EXPECT_NE(Err.Message.find(".text section not found"), std::string::npos)
      << Err.Message;
}

TEST(CodeObjectUtils, ListKernelNamesNoMetadataReturnsHotswapError) {
  std::vector<uint8_t> Bytes = buildMinimalAmdgpuElf();
  llvm::Expected<llvm::SmallVector<std::string>> Result =
      COMGR::hotswap::listKernelNames(toBuf(Bytes, "no-meta.elf"));

  ASSERT_FALSE(static_cast<bool>(Result));
  InspectedError Err = inspect(Result.takeError());
  EXPECT_TRUE(Err.IsHotswapError);
  EXPECT_NE(Err.Message.find("no AMDGPU metadata note"), std::string::npos)
      << Err.Message;
}

TEST(CodeObjectUtils, ExtractKernelMetaNoMetadataReturnsHotswapError) {
  std::vector<uint8_t> Bytes = buildMinimalAmdgpuElf();
  llvm::Expected<COMGR::hotswap::KernelMeta> Result =
      COMGR::hotswap::extractKernelMeta(toBuf(Bytes, "no-meta.elf"),
                                        "any_kernel");

  ASSERT_FALSE(static_cast<bool>(Result));
  InspectedError Err = inspect(Result.takeError());
  EXPECT_TRUE(Err.IsHotswapError);
  EXPECT_NE(Err.Message.find("no AMDGPU metadata note"), std::string::npos)
      << Err.Message;
}

TEST(CodeObjectUtils, FindKernelSymbolOffsetMissingTextReturnsHotswapError) {
  std::vector<uint8_t> Bytes = buildMinimalAmdgpuElf();
  llvm::Expected<uint64_t> Result = COMGR::hotswap::findKernelSymbolOffset(
      toBuf(Bytes, "no-text.elf"), "any_kernel");

  ASSERT_FALSE(static_cast<bool>(Result));
  InspectedError Err = inspect(Result.takeError());
  EXPECT_TRUE(Err.IsHotswapError);
  EXPECT_NE(Err.Message.find("no .text section in ELF"), std::string::npos)
      << Err.Message;
}
