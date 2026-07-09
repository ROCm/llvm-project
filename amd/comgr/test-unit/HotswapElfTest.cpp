//===- HotswapElfTest.cpp - Unit tests for HotSwap ELF layer --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr-test-elf-utils.h"
#include "gtest/gtest.h"

#include <cstring>
#include <limits>

using namespace COMGR::hotswap;

static std::vector<uint8_t> makeText(size_t Size = 16) {
  return std::vector<uint8_t>(Size, 0);
}

static unsigned readReservedSgprs(const std::vector<uint8_t> &Bytes,
                                  uint64_t KernelDescriptorOffset) {
  namespace hsa = llvm::amdhsa;

  uint32_t Rsrc1 = 0;
  std::memcpy(&Rsrc1,
              Bytes.data() + KernelDescriptorOffset +
                  offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  return (AMDHSA_BITS_GET(
              Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
          1) *
         8;
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

// -- ElfView::findKernelAtAddress ---------------------------------------------

TEST(ElfView, FindKernelAtAddressResolvesNearestPrecedingForZeroSizeSymbol) {
  // AMDGPU kernel entry symbols frequently have st_size == 0 (the size lives on
  // the .kd object symbol), so an exact [st_value, st_value + st_size)
  // containment test never matches. The lookup must resolve via the
  // nearest-preceding STT_FUNC symbol instead.
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "zero_size_kernel";
  Opts.TextAddr = 0x1000;
  Opts.ZeroSizeKernelSym = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // findKernelAtAddress takes a virtual address; at the entry and at an
  // interior offset the zero-size symbol still resolves.
  EXPECT_EQ(ViewOrErr->findKernelAtAddress(0x1000), "zero_size_kernel");
  EXPECT_EQ(ViewOrErr->findKernelAtAddress(0x1000 + 4), "zero_size_kernel");
  // An address before the symbol has no preceding function symbol to resolve.
  EXPECT_EQ(ViewOrErr->findKernelAtAddress(0x0FF0), "");
}

// -- findNearestSled ----------------------------------------------------------

TEST(FindNearestSled, SkipsSledsOutsideInstructionFunctionRange) {
  std::vector<NopSled> Sleds;
  // {Start, End, WritePos, FunctionStart, FunctionEnd}
  Sleds.push_back({/*Start=*/0, /*End=*/32, /*WritePos=*/0,
                   /*FunctionStart=*/0, /*FunctionEnd=*/32});
  Sleds.push_back({/*Start=*/96, /*End=*/128, /*WritePos=*/96,
                   /*FunctionStart=*/96, /*FunctionEnd=*/160});

  NopSled *Sled = findNearestSled(Sleds, 108, 8);
  ASSERT_NE(Sled, nullptr);
  EXPECT_EQ(Sled->Start, 96u);

  EXPECT_EQ(findNearestSled(Sleds, 64, 8), nullptr);
}

// -- ElfView::getKernelStaticLdsSize ------------------------------------------

TEST(ElfView, GetKernelStaticLdsSizeReturnsNulloptWhenKdMissing) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelStaticLdsSize("nonexistent_kernel"),
            std::nullopt);
}

TEST(ElfView, GetKernelStaticLdsSizeReadsLdsSizeFromKernelDescriptor) {
  static constexpr uint32_t TestLdsSize = 16384;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.ElfType = llvm::ELF::ET_REL;
  Opts.KernelName = "test_kernel";
  Opts.TextAddr = 0;
  Opts.RodataAddr = 0;
  Opts.GroupSegmentFixedSize = TestLdsSize;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::optional<uint32_t> Lds =
      ViewOrErr->getKernelStaticLdsSize("test_kernel");
  ASSERT_TRUE(Lds.has_value());
  EXPECT_EQ(*Lds, TestLdsSize);
}

TEST(ElfView, KernelDescriptorsEnumeratesAndUpdatesEntryOffset) {
  namespace hsa = llvm::amdhsa;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::vector<KernelDescriptorInfo> KDs = ViewOrErr->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].KernelName, "entry_kernel");
  EXPECT_EQ(KDs[0].VAddr, Obj.RodataAddr);
  EXPECT_EQ(KDs[0].EntryOffset, Obj.EntryOffset);
  EXPECT_EQ(ViewOrErr->getKernelDescriptorVAddr("entry_kernel"),
            Obj.RodataAddr);

  const int64_t NewOff = -128;
  ASSERT_TRUE(
      ViewOrErr->updateKernelDescriptorEntryOffset("entry_kernel", NewOff));
  int64_t ReadBack = 0;
  std::memcpy(
      &ReadBack,
      Obj.Bytes.data() + Obj.KernelDescriptorOffset +
          offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      sizeof(ReadBack));
  EXPECT_EQ(ReadBack, NewOff);

  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
}

TEST(ElfView, KernelDescriptorsSkipsKdWhenFileOffsetOverflows) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "overflow_kernel";
  Opts.RodataAddr = 0x1000;
  Opts.KernelDescriptorSymbolValue =
      std::numeric_limits<uint64_t>::max() - 0x20;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_TRUE(ViewOrErr->kernelDescriptors().empty());
  EXPECT_EQ(ViewOrErr->findKernelDescriptor("overflow_kernel"), nullptr);
}

TEST(ElfView, GrowWithTrampolinesShiftsAllocSectionSymbols) {
  static constexpr uint64_t GrowthBytes = 8;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  Trampoline T;
  T.Bytes.assign(GrowthBytes, 0);
  std::vector<Trampoline> Trampolines;
  Trampolines.push_back(T);
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(Trampolines, SNop);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  std::vector<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].VAddr, Obj.RodataAddr + GrowthBytes);
}

// Covers: addKernelEntryTrampolineSymbols attaches a distinct, correctly
// placed `<kernel>.stub` symbol for every appended entry-trampoline stub, so a
// dispatch whose entry now points at a stub still resolves to a name.
//
// How: build a synthetic AMDGPU code object that has a .symtab, then grow .text
// by two entry-stub-sized (KernelEntryStubStride) blocks with
// growWithTrampolines -- mirroring the pass appending one stub per kernel.
// Call addKernelEntryTrampolineSymbols with two fixups that use distinct kernel
// names and the two stub offsets (0 and KernelEntryStubStride). Re-parse the
// returned buffer with llvm::object::ELFFile and, for each fixup, assert a
// "<name>.stub" symbol exists in .symtab that is (a) STT_FUNC, (b) defined in
// the .text section (st_shndx), (c) located at TextAddr + OldTextSize +
// StubTextOffset, and (d) sized to KernelEntryStubStride. Two fixups (rather
// than one) prove each stub gets its own name at its own address, not a single
// shared or mis-placed entry.
TEST(ElfView, AddKernelEntryTrampolineSymbolsNamesEachStub) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  const unsigned TextIdx = ViewOrErr->textSectionIndex();
  const uint64_t TextAddr = ViewOrErr->textAddr();
  const uint64_t OldTextSize = ViewOrErr->textSize();

  // Grow .text by two entry-stub-sized blocks, mirroring the entry-trampoline
  // pass appending one stub per kernel.
  Trampoline Stub;
  Stub.Bytes.assign(2 * KernelEntryStubStride, 0);
  std::vector<Trampoline> Growth{Stub};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      ViewOrErr->growWithTrampolines(Growth, SNop);
  ASSERT_NE(Grown, nullptr);

  // One fixup per appended stub; the names need not match real kernels, since
  // addKernelEntryTrampolineSymbols only attaches a symbol at each stub address.
  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"kernel_a", /*StubTextOffset=*/0, /*RequiredSgprs=*/10},
      {"kernel_b", /*StubTextOffset=*/KernelEntryStubStride, /*RequiredSgprs=*/12},
  };
  std::unique_ptr<llvm::WritableMemoryBuffer> WithSyms =
      addKernelEntryTrampolineSymbols(*Grown, TextIdx, TextAddr, OldTextSize,
                                      Fixups);
  ASSERT_NE(WithSyms, nullptr);

  using ELFT = llvm::object::ELF64LE;
  const uint8_t *Data =
      reinterpret_cast<const uint8_t *>(WithSyms->getBufferStart());
  llvm::Expected<llvm::object::ELFFile<ELFT>> FileOrErr =
      llvm::object::ELFFile<ELFT>::create(llvm::StringRef(
          reinterpret_cast<const char *>(Data), WithSyms->getBufferSize()));
  ASSERT_TRUE((bool)FileOrErr) << llvm::toString(FileOrErr.takeError());
  llvm::object::ELFFile<ELFT> &File = *FileOrErr;

  llvm::Expected<ELFT::ShdrRange> SecsOrErr = File.sections();
  ASSERT_TRUE((bool)SecsOrErr) << llvm::toString(SecsOrErr.takeError());
  const ELFT::Shdr *SymtabShdr = nullptr;
  for (const ELFT::Shdr &S : *SecsOrErr)
    if (S.sh_type == llvm::ELF::SHT_SYMTAB) {
      SymtabShdr = &S;
      break;
    }
  ASSERT_NE(SymtabShdr, nullptr);
  llvm::Expected<ELFT::SymRange> SymsOrErr = File.symbols(SymtabShdr);
  ASSERT_TRUE((bool)SymsOrErr) << llvm::toString(SymsOrErr.takeError());
  llvm::Expected<llvm::StringRef> StrTabOrErr =
      File.getStringTableForSymtab(*SymtabShdr);
  ASSERT_TRUE((bool)StrTabOrErr) << llvm::toString(StrTabOrErr.takeError());

  auto FindSym = [&](llvm::StringRef Name) -> const ELFT::Sym * {
    for (const ELFT::Sym &Sym : *SymsOrErr) {
      llvm::Expected<llvm::StringRef> N = Sym.getName(*StrTabOrErr);
      if (N && *N == Name)
        return &Sym;
    }
    return nullptr;
  };

  // Every appended stub must have a <kernel>.stub STT_FUNC symbol covering the
  // stub, in the .text section, at the stub's virtual address.
  for (const KernelEntryTrampolineFixup &F : Fixups) {
    const ELFT::Sym *Sym = FindSym(F.KernelName + ".stub");
    ASSERT_NE(Sym, nullptr) << "missing stub symbol for " << F.KernelName;
    EXPECT_EQ(static_cast<unsigned>(Sym->getType()),
              static_cast<unsigned>(llvm::ELF::STT_FUNC));
    EXPECT_EQ(Sym->st_shndx, TextIdx);
    EXPECT_EQ(Sym->st_value, TextAddr + OldTextSize + F.StubTextOffset);
    EXPECT_EQ(Sym->st_size, KernelEntryStubStride);
  }
}

TEST(ElfView, UpdateKernelDescriptorSgprCountUpdatesMetadataAndDescriptor) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 10u);
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMissingMetadataCount) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataOmitSgprCount = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMissingMetadataKernel) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataKernelName = "other_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), std::nullopt);
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsNonIntegerMetadataCount) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCountAsString = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), std::nullopt);
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMetadataSizeChange) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 9;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 128));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 9u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsDescriptorLimitFirst) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 200;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(
      ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 100000));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 200u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}
