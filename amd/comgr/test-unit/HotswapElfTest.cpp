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

static llvm::Expected<ElfView>
createElfView(comgr_test::KernelDescriptorElf &Obj) {
  return ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
}

static bool relocateHeaderTable(std::vector<uint8_t> &Bytes,
                                uint64_t OldOffset, uint16_t EntrySize,
                                uint16_t OldCount, uint16_t NewCount,
                                size_t MinimumEntrySize,
                                uint64_t &NewOffset) {
  if (NewCount < OldCount || EntrySize < MinimumEntrySize ||
      OldOffset > Bytes.size())
    return false;
  const uint64_t OldSize = static_cast<uint64_t>(OldCount) * EntrySize;
  if (OldSize > Bytes.size() - OldOffset)
    return false;

  std::vector<uint8_t> OldTable(
      Bytes.begin() + static_cast<size_t>(OldOffset),
      Bytes.begin() + static_cast<size_t>(OldOffset + OldSize));
  NewOffset = comgr_test::alignTo8(Bytes.size());
  const uint64_t NewSize = static_cast<uint64_t>(NewCount) * EntrySize;
  if (NewOffset > std::numeric_limits<size_t>::max() - NewSize)
    return false;
  Bytes.resize(static_cast<size_t>(NewOffset + NewSize), 0);
  std::memcpy(Bytes.data() + NewOffset, OldTable.data(), OldTable.size());
  return true;
}

static bool setProgramHeaderCount(std::vector<uint8_t> &Bytes,
                                  uint16_t NewCount) {
  if (Bytes.size() < sizeof(llvm::ELF::Elf64_Ehdr))
    return false;
  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Bytes.data(), sizeof(Header));
  uint64_t NewOffset = 0;
  if (!relocateHeaderTable(Bytes, Header.e_phoff, Header.e_phentsize,
                           Header.e_phnum, NewCount,
                           sizeof(llvm::ELF::Elf64_Phdr), NewOffset))
    return false;
  Header.e_phoff = NewOffset;
  Header.e_phnum = NewCount;
  std::memcpy(Bytes.data(), &Header, sizeof(Header));
  return true;
}

static bool setSectionHeaderCount(std::vector<uint8_t> &Bytes,
                                  uint16_t NewCount) {
  if (Bytes.size() < sizeof(llvm::ELF::Elf64_Ehdr))
    return false;
  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Bytes.data(), sizeof(Header));
  uint64_t NewOffset = 0;
  if (!relocateHeaderTable(Bytes, Header.e_shoff, Header.e_shentsize,
                           Header.e_shnum, NewCount,
                           sizeof(llvm::ELF::Elf64_Shdr), NewOffset))
    return false;
  Header.e_shoff = NewOffset;
  Header.e_shnum = NewCount;
  std::memcpy(Bytes.data(), &Header, sizeof(Header));
  return true;
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

static unsigned readReservedVgprs(const std::vector<uint8_t> &Bytes,
                                  uint64_t KernelDescriptorOffset,
                                  unsigned Granule) {
  namespace hsa = llvm::amdhsa;

  uint32_t Rsrc1 = 0;
  std::memcpy(&Rsrc1,
              Bytes.data() + KernelDescriptorOffset +
                  offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  return (AMDHSA_BITS_GET(
              Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) +
          1) *
         Granule;
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

TEST(ElfView, RejectsExtendedSymbolSectionIndices) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Obj.Bytes.data(), sizeof(Header));
  llvm::ELF::Elf64_Shdr Symtab{};
  std::memcpy(&Symtab,
              Obj.Bytes.data() + Header.e_shoff + 4 * Header.e_shentsize,
              sizeof(Symtab));
  llvm::ELF::Elf64_Sym Kernel{};
  const uint64_t KernelOffset = Symtab.sh_offset + sizeof(Kernel);
  std::memcpy(&Kernel, Obj.Bytes.data() + KernelOffset, sizeof(Kernel));
  Kernel.st_shndx = llvm::ELF::SHN_XINDEX;
  std::memcpy(Obj.Bytes.data() + KernelOffset, &Kernel, sizeof(Kernel));

  llvm::Expected<ElfView> View = createElfView(Obj);
  EXPECT_FALSE((bool)View);
  llvm::consumeError(View.takeError());
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
  std::optional<ElfView::FunctionTextRange> Range =
      ViewOrErr->findFunctionTextRangeAtOffset(4);
  ASSERT_TRUE(Range.has_value());
  EXPECT_EQ(Range->Begin, 0u);
  EXPECT_EQ(Range->End, makeText().size());
  EXPECT_NE(Range->Symbol, nullptr);
  EXPECT_NE(Range->Symtab, nullptr);
  // An address before the symbol has no preceding function symbol to resolve.
  EXPECT_EQ(ViewOrErr->findKernelAtAddress(0x0FF0), "");
}

// -- findNearestSled ----------------------------------------------------------

TEST(FindNearestSled, SkipsSledsOutsideSourceOwnerRange) {
  std::vector<NopSled> Sleds;
  // {Start, End, WritePos, OwnerStart, OwnerEnd}
  Sleds.push_back({/*Start=*/0, /*End=*/32, /*WritePos=*/0,
                   /*OwnerStart=*/0, /*OwnerEnd=*/32});
  Sleds.push_back({/*Start=*/96, /*End=*/128, /*WritePos=*/96,
                   /*OwnerStart=*/96, /*OwnerEnd=*/160});

  NopSled *Sled = findNearestSled(Sleds, 108, 8);
  ASSERT_NE(Sled, nullptr);
  EXPECT_EQ(Sled->Start, 96u);

  EXPECT_EQ(findNearestSled(Sleds, 64, 8), nullptr);
}

TEST(FindNearestSled, UsesGatewayOnlyPaddingOnlyWhenRequested) {
  std::vector<NopSled> Sleds;
  Sleds.push_back({/*Start=*/0, /*End=*/32, /*WritePos=*/0,
                   /*OwnerStart=*/0, /*OwnerEnd=*/128,
                   /*GatewayOnly=*/true});
  Sleds.push_back({/*Start=*/64, /*End=*/96, /*WritePos=*/64,
                   /*OwnerStart=*/0, /*OwnerEnd=*/128});

  EXPECT_EQ(findNearestSled(Sleds, 4, 8), &Sleds[1]);
  EXPECT_EQ(findNearestSled(Sleds, 4, 8, NopSledUse::Gateway),
            &Sleds[0]);
}

TEST(FindNearestSled, SeparatesSourceOwnerFromStorageCapacity) {
  std::vector<NopSled> Sleds = {
      {/*Start=*/64, /*End=*/96, /*WritePos=*/64,
       /*OwnerStart=*/0, /*OwnerEnd=*/32}};

  EXPECT_EQ(findNearestSled(Sleds, /*Offset=*/16, /*Needed=*/32), &Sleds[0]);
  EXPECT_EQ(findNearestSled(Sleds, /*Offset=*/64, /*Needed=*/8), nullptr);
}

TEST(FindNearestSled, SharesPostFunctionPaddingAcrossCertifiedRoles) {
  std::vector<NopSled> Sleds = {
      {/*Start=*/64, /*End=*/96, /*WritePos=*/64,
       /*OwnerStart=*/0, /*OwnerEnd=*/32, /*GatewayOnly=*/false,
       /*GlobalGateway=*/true, /*GlobalBody=*/true}};

  NopSled *Body =
      findNearestSled(Sleds, /*Offset=*/16, /*Needed=*/20);
  ASSERT_EQ(Body, &Sleds[0]);
  Body->WritePos += 20;

  EXPECT_EQ(findNearestSled(Sleds, /*Offset=*/128, /*Needed=*/8), nullptr);
  EXPECT_EQ(findNearestSled(Sleds, /*Offset=*/128, /*Needed=*/8,
                            NopSledUse::RelocationBody),
            &Sleds[0]);
  EXPECT_EQ(findNearestSled(Sleds, /*Offset=*/128, /*Needed=*/8,
                            NopSledUse::Gateway),
            &Sleds[0]);
  EXPECT_EQ(Sleds[0].WritePos, 84u);
}

TEST(FindNearestSled, PrefersOwnerBodyBeforeGlobalRelocationBody) {
  std::vector<NopSled> Sleds = {
      {/*Start=*/40, /*End=*/72, /*WritePos=*/40,
       /*OwnerStart=*/64, /*OwnerEnd=*/96, /*GatewayOnly=*/false,
       /*GlobalGateway=*/true, /*GlobalBody=*/true},
      {/*Start=*/200, /*End=*/232, /*WritePos=*/200,
       /*OwnerStart=*/0, /*OwnerEnd=*/32}};

  EXPECT_EQ(findNearestSled(Sleds, /*Offset=*/16, /*Needed=*/20,
                            NopSledUse::RelocationBody),
            &Sleds[1]);
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
  llvm::ArrayRef<KernelDescriptorInfo> KDs = ViewOrErr->kernelDescriptors();
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
  ASSERT_EQ(ViewOrErr->kernelDescriptors().size(), 1u);
  EXPECT_EQ(ViewOrErr->kernelDescriptors()[0].EntryOffset, NewOff);

  // Prime the descriptor fallback cache before changing the encoded count.
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 8u);
  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/true));
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 16u);
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

// growWithTrampolines appends the pool at a fresh high virtual address instead
// of growing .text and shifting everything after it, so existing allocatable
// symbols keep their addresses. (This replaces the earlier test that pinned the
// buggy shifting behavior; the shift is exactly what corrupted the baked ISA
// references -- see GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol.)
TEST(ElfView, GrowWithTrampolinesKeepsAllocSectionSymbols) {
  static constexpr uint64_t GrowthBytes = 8;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
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
      ViewOrErr->growWithTrampolines(
          Trampolines, SNop, ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  llvm::ArrayRef<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].VAddr, Obj.RodataAddr);
}

TEST(ElfView, TrampolinePoolVAddrRejectsAlignmentOverflow) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Obj.Bytes.data(), sizeof(Header));
  llvm::ELF::Elf64_Shdr Rodata{};
  std::memcpy(&Rodata,
              Obj.Bytes.data() + Header.e_shoff + 2 * Header.e_shentsize,
              sizeof(Rodata));
  ASSERT_NE(Rodata.sh_size, 0u);
  Rodata.sh_addr = std::numeric_limits<uint64_t>::max() - Rodata.sh_size;
  std::memcpy(Obj.Bytes.data() + Header.e_shoff + 2 * Header.e_shentsize,
              &Rodata, sizeof(Rodata));

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  EXPECT_FALSE(View->trampolinePoolVAddr().has_value());
}

TEST(ElfView, TrampolinePoolVAddrAccountsForLoadSegmentMemoryTail) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Obj.Bytes.data(), sizeof(Header));
  ASSERT_EQ(Header.e_phnum, 1u);
  llvm::ELF::Elf64_Phdr Load{};
  std::memcpy(&Load, Obj.Bytes.data() + Header.e_phoff, sizeof(Load));
  ASSERT_EQ(Load.p_type, llvm::ELF::PT_LOAD);
  Load.p_memsz = 0x5001;
  std::memcpy(Obj.Bytes.data() + Header.e_phoff, &Load, sizeof(Load));

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  ASSERT_TRUE(View->trampolinePoolVAddr().has_value());
  EXPECT_EQ(*View->trampolinePoolVAddr(), 0x7000u);
}

TEST(ElfView, TrampolinePoolVAddrRejectsLoadSegmentRangeOverflow) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Obj.Bytes.data(), sizeof(Header));
  ASSERT_EQ(Header.e_phnum, 1u);
  llvm::ELF::Elf64_Phdr Load{};
  std::memcpy(&Load, Obj.Bytes.data() + Header.e_phoff, sizeof(Load));
  ASSERT_EQ(Load.p_type, llvm::ELF::PT_LOAD);
  Load.p_vaddr = std::numeric_limits<uint64_t>::max() - 7;
  Load.p_memsz = 16;
  std::memcpy(Obj.Bytes.data() + Header.e_phoff, &Load, sizeof(Load));

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  EXPECT_FALSE(View->trampolinePoolVAddr().has_value());
}

TEST(ElfView, GrowWithTrampolinesAcceptsLargestDirectProgramHeaderCount) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  constexpr uint16_t InputCount = llvm::ELF::PN_XNUM - 2;
  ASSERT_TRUE(setProgramHeaderCount(Obj.Bytes, InputCount));

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  Trampoline T;
  T.Bytes.assign(8, 0);
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      View->growWithTrampolines({T}, SNop,
                                ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Out, nullptr);

  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Out->getBufferStart(), sizeof(Header));
  EXPECT_EQ(Header.e_phnum, llvm::ELF::PN_XNUM - 1);
  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  EXPECT_TRUE(OutView->trampolinePoolVAddr().has_value());
}

TEST(ElfView, GrowWithTrampolinesRejectsProgramHeaderExtendedNumbering) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  constexpr uint16_t InputCount = llvm::ELF::PN_XNUM - 1;
  ASSERT_TRUE(setProgramHeaderCount(Obj.Bytes, InputCount));

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  Trampoline T;
  T.Bytes.assign(8, 0);
  const uint8_t SNop[4] = {};
  EXPECT_EQ(View->growWithTrampolines(
                {T}, SNop, ExecutablePoolTargetState::Neutral),
            nullptr);
}

TEST(ElfView, GrowWithTrampolinesAcceptsLargestDirectSectionHeaderCount) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  constexpr uint16_t InputCount = llvm::ELF::SHN_LORESERVE - 3;
  ASSERT_TRUE(setSectionHeaderCount(Obj.Bytes, InputCount));

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  Trampoline T;
  T.Bytes.assign(8, 0);
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      View->growWithTrampolines({T}, SNop,
                                ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Out, nullptr);

  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Out->getBufferStart(), sizeof(Header));
  EXPECT_EQ(Header.e_shnum, llvm::ELF::SHN_LORESERVE - 1);
  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  EXPECT_TRUE(OutView->trampolinePoolVAddr().has_value());
}

TEST(ElfView, GrowWithTrampolinesRejectsSectionHeaderExtendedNumbering) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());
  constexpr uint16_t InputCount = llvm::ELF::SHN_LORESERVE - 2;
  ASSERT_TRUE(setSectionHeaderCount(Obj.Bytes, InputCount));

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  Trampoline T;
  T.Bytes.assign(8, 0);
  const uint8_t SNop[4] = {};
  EXPECT_EQ(View->growWithTrampolines(
                {T}, SNop, ExecutablePoolTargetState::Neutral),
            nullptr);
}

TEST(ElfView, GrowWithTrampolinesDropsRelocatedPtPhdr) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.EmitPhdrSegment = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);
  llvm::ELF::Elf64_Ehdr Before{};
  std::memcpy(&Before, Obj.Bytes.data(), sizeof(Before));
  ASSERT_GE(Before.e_phnum, 2u);
  llvm::ELF::Elf64_Phdr OriginalPhdr{};
  std::memcpy(&OriginalPhdr, Obj.Bytes.data() + Before.e_phoff,
              sizeof(OriginalPhdr));
  ASSERT_EQ(OriginalPhdr.p_type, llvm::ELF::PT_PHDR);

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  Trampoline T;
  T.Bytes.assign(8, 0);
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      View->growWithTrampolines({T}, SNop,
                                ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Out, nullptr);

  llvm::ELF::Elf64_Ehdr After{};
  std::memcpy(&After, Out->getBufferStart(), sizeof(After));
  EXPECT_NE(After.e_phoff, Before.e_phoff);
  EXPECT_EQ(After.e_phnum, Before.e_phnum + 1);
  bool SawNull = false;
  bool SawPhdr = false;
  for (uint16_t I = 0; I != After.e_phnum; ++I) {
    llvm::ELF::Elf64_Phdr Entry{};
    std::memcpy(&Entry,
                Out->getBufferStart() + After.e_phoff +
                    static_cast<uint64_t>(I) * After.e_phentsize,
                sizeof(Entry));
    SawNull |= Entry.p_type == llvm::ELF::PT_NULL;
    SawPhdr |= Entry.p_type == llvm::ELF::PT_PHDR;
  }
  EXPECT_TRUE(SawNull);
  EXPECT_FALSE(SawPhdr);
}

TEST(ElfView, ExecutablePoolProvenanceIsTargetSpecificAndFailClosed) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);
  llvm::Expected<ElfView> SourceView = createElfView(Obj);
  ASSERT_TRUE((bool)SourceView) << llvm::toString(SourceView.takeError());
  const uint64_t TextAddr = SourceView->textAddr();
  const uint64_t TextOffset = SourceView->textOffset();
  const uint8_t SNop[4] = {};
  Trampoline T;
  T.Bytes.assign(8, 0);
  std::vector<Trampoline> Trampolines{T};

  auto Grow = [&](ExecutablePoolTargetState State) {
    llvm::Expected<ElfView> View = createElfView(Obj);
    EXPECT_TRUE((bool)View);
    if (!View) {
      llvm::consumeError(View.takeError());
      return std::unique_ptr<llvm::WritableMemoryBuffer>();
    }
    return View->growWithTrampolines(Trampolines, SNop, State);
  };
  auto ExpectCompatibility = [](llvm::WritableMemoryBuffer &Buffer,
                                ExecutablePoolTargetState State,
                                bool Expected) {
    uint8_t *Data = reinterpret_cast<uint8_t *>(Buffer.getBufferStart());
    llvm::Expected<ElfView> View =
        ElfView::create(Data, Buffer.getBufferSize());
    ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
    std::optional<bool> Compatible =
        View->executableCodeOutsideTextIsCompatibleWith(State);
    ASSERT_TRUE(Compatible.has_value());
    EXPECT_EQ(*Compatible, Expected);
  };

  std::unique_ptr<llvm::WritableMemoryBuffer> Neutral =
      Grow(ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Neutral, nullptr);
  ExpectCompatibility(*Neutral, ExecutablePoolTargetState::A0, true);
  ExpectCompatibility(*Neutral, ExecutablePoolTargetState::B0, true);

  std::unique_ptr<llvm::WritableMemoryBuffer> A0 =
      Grow(ExecutablePoolTargetState::A0);
  ASSERT_NE(A0, nullptr);
  ExpectCompatibility(*A0, ExecutablePoolTargetState::A0, true);
  ExpectCompatibility(*A0, ExecutablePoolTargetState::B0, false);

  std::unique_ptr<llvm::WritableMemoryBuffer> B0 =
      Grow(ExecutablePoolTargetState::B0);
  ASSERT_NE(B0, nullptr);
  ExpectCompatibility(*B0, ExecutablePoolTargetState::B0, true);
  ExpectCompatibility(*B0, ExecutablePoolTargetState::A0, false);
  EXPECT_EQ(Grow(static_cast<ExecutablePoolTargetState>(99)), nullptr);

  uint8_t *A0Data = reinterpret_cast<uint8_t *>(A0->getBufferStart());
  llvm::Expected<ElfView> A0View = ElfView::create(A0Data, A0->getBufferSize());
  ASSERT_TRUE((bool)A0View) << llvm::toString(A0View.takeError());
  std::unique_ptr<llvm::WritableMemoryBuffer> TwoPools =
      A0View->growWithTrampolines(Trampolines, SNop,
                                  ExecutablePoolTargetState::A0);
  ASSERT_NE(TwoPools, nullptr);
  ExpectCompatibility(*TwoPools, ExecutablePoolTargetState::A0, true);

  using ELFT = llvm::object::ELF64LE;
  auto FindPoolNoteSection = [](llvm::WritableMemoryBuffer &Buffer)
      -> std::optional<std::pair<unsigned, ELFT::Shdr>> {
    const uint8_t *Data =
        reinterpret_cast<const uint8_t *>(Buffer.getBufferStart());
    llvm::Expected<llvm::object::ELFFile<ELFT>> File =
        llvm::object::ELFFile<ELFT>::create(llvm::StringRef(
            reinterpret_cast<const char *>(Data), Buffer.getBufferSize()));
    if (!File)
      return std::nullopt;
    llvm::Expected<ELFT::ShdrRange> Sections = File->sections();
    if (!Sections)
      return std::nullopt;
    for (unsigned I = 0; I != Sections->size(); ++I)
      if ((*Sections)[I].sh_type == llvm::ELF::SHT_NOTE &&
          !((*Sections)[I].sh_flags & llvm::ELF::SHF_ALLOC))
        return std::pair<unsigned, ELFT::Shdr>{I, (*Sections)[I]};
    return std::nullopt;
  };
  auto CloneBuffer = [](llvm::WritableMemoryBuffer &Buffer) {
    std::unique_ptr<llvm::WritableMemoryBuffer> Clone =
        llvm::WritableMemoryBuffer::getNewMemBuffer(Buffer.getBufferSize());
    if (Clone)
      std::memcpy(Clone->getBufferStart(), Buffer.getBufferStart(),
                  Buffer.getBufferSize());
    return Clone;
  };
  auto FindPoolSection = [TextAddr](llvm::WritableMemoryBuffer &Buffer)
      -> std::optional<std::pair<unsigned, ELFT::Shdr>> {
    const uint8_t *Data =
        reinterpret_cast<const uint8_t *>(Buffer.getBufferStart());
    llvm::Expected<llvm::object::ELFFile<ELFT>> File =
        llvm::object::ELFFile<ELFT>::create(llvm::StringRef(
            reinterpret_cast<const char *>(Data), Buffer.getBufferSize()));
    if (!File)
      return std::nullopt;
    llvm::Expected<ELFT::ShdrRange> Sections = File->sections();
    if (!Sections)
      return std::nullopt;
    for (unsigned I = 0; I != Sections->size(); ++I) {
      const ELFT::Shdr &Shdr = (*Sections)[I];
      if (Shdr.sh_type == llvm::ELF::SHT_PROGBITS &&
          (Shdr.sh_flags & (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR)) ==
              (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR) &&
          Shdr.sh_addr != TextAddr)
        return std::pair<unsigned, ELFT::Shdr>{I, Shdr};
    }
    return std::nullopt;
  };
  enum class Mutation {
    UnsupportedVersion,
    Unmarked,
    DuplicateNote,
    Align8Note,
    MissingNameNul,
    EnlargedTextLoad,
    PoolFileAliasesText,
    PoolVAddrOverlapsText,
  };
  auto Mutate = [&](llvm::WritableMemoryBuffer &Buffer, Mutation Kind) {
    llvm::ELF::Elf64_Ehdr Header{};
    std::memcpy(&Header, Buffer.getBufferStart(), sizeof(Header));
    auto WriteShdr = [&](unsigned Index, const ELFT::Shdr &Shdr) {
      std::memcpy(Buffer.getBufferStart() + Header.e_shoff +
                      static_cast<uint64_t>(Index) * Header.e_shentsize,
                  &Shdr, sizeof(Shdr));
    };
    auto RewritePhdr = [&](auto Predicate, auto Rewrite) {
      for (unsigned I = 0; I != Header.e_phnum; ++I) {
        llvm::ELF::Elf64_Phdr Phdr{};
        char *PhdrData = Buffer.getBufferStart() + Header.e_phoff +
                         static_cast<uint64_t>(I) * Header.e_phentsize;
        std::memcpy(&Phdr, PhdrData, sizeof(Phdr));
        if (!Predicate(Phdr))
          continue;
        Rewrite(Phdr);
        std::memcpy(PhdrData, &Phdr, sizeof(Phdr));
        return true;
      }
      return false;
    };

    std::optional<std::pair<unsigned, ELFT::Shdr>> Note =
        FindPoolNoteSection(Buffer);
    if (!Note)
      return false;
    if (Kind == Mutation::UnsupportedVersion) {
      uint32_t Version = 2;
      std::memcpy(Buffer.getBufferStart() + Note->second.sh_offset + 20,
                  &Version, sizeof(Version));
      return true;
    }
    if (Kind == Mutation::Unmarked) {
      ELFT::Shdr Shdr = Note->second;
      Shdr.sh_type = llvm::ELF::SHT_PROGBITS;
      WriteShdr(Note->first, Shdr);
      return true;
    }
    if (Kind == Mutation::DuplicateNote) {
      for (unsigned I = 0; I != Header.e_shnum; ++I) {
        ELFT::Shdr Shdr{};
        std::memcpy(&Shdr,
                    Buffer.getBufferStart() + Header.e_shoff +
                        static_cast<uint64_t>(I) * Header.e_shentsize,
                    sizeof(Shdr));
        if (Shdr.sh_type == llvm::ELF::SHT_PROGBITS &&
            !(Shdr.sh_flags & llvm::ELF::SHF_EXECINSTR)) {
          WriteShdr(I, Note->second);
          return true;
        }
      }
      return false;
    }
    if (Kind == Mutation::Align8Note) {
      ELFT::Shdr Shdr = Note->second;
      Shdr.sh_addralign = 8;
      WriteShdr(Note->first, Shdr);
      return true;
    }
    if (Kind == Mutation::MissingNameNul) {
      uint32_t NameSize = 6;
      std::memcpy(Buffer.getBufferStart() + Note->second.sh_offset, &NameSize,
                  sizeof(NameSize));
      return true;
    }
    if (Kind == Mutation::EnlargedTextLoad)
      return RewritePhdr(
          [&](const llvm::ELF::Elf64_Phdr &Phdr) {
            return Phdr.p_type == llvm::ELF::PT_LOAD &&
                   Phdr.p_vaddr == TextAddr && Phdr.p_offset == TextOffset;
          },
          [](llvm::ELF::Elf64_Phdr &Phdr) {
            ++Phdr.p_filesz;
            ++Phdr.p_memsz;
          });

    std::optional<std::pair<unsigned, ELFT::Shdr>> Pool =
        FindPoolSection(Buffer);
    if (!Pool)
      return false;
    if (Kind == Mutation::PoolFileAliasesText) {
      ELFT::Shdr Shdr = Pool->second;
      Shdr.sh_offset = TextOffset;
      WriteShdr(Pool->first, Shdr);
      return RewritePhdr(
          [&](const llvm::ELF::Elf64_Phdr &Phdr) {
            return Phdr.p_type == llvm::ELF::PT_LOAD &&
                   Phdr.p_vaddr == Pool->second.sh_addr &&
                   Phdr.p_filesz == Pool->second.sh_size;
          },
          [&](llvm::ELF::Elf64_Phdr &Phdr) { Phdr.p_offset = TextOffset; });
    }
    ELFT::Shdr Shdr = Pool->second;
    Shdr.sh_addr = TextAddr;
    WriteShdr(Pool->first, Shdr);
    bool Rewritten = RewritePhdr(
        [&](const llvm::ELF::Elf64_Phdr &Phdr) {
          return Phdr.p_type == llvm::ELF::PT_LOAD &&
                 Phdr.p_vaddr == Pool->second.sh_addr &&
                 Phdr.p_filesz == Pool->second.sh_size;
        },
        [&](llvm::ELF::Elf64_Phdr &Phdr) {
          Phdr.p_vaddr = TextAddr;
          Phdr.p_paddr = TextAddr;
        });
    // Nhdr (12) + padded "AMDGPU\0" (8) + version/state (8).
    std::memcpy(Buffer.getBufferStart() + Note->second.sh_offset + 28,
                &TextAddr, sizeof(TextAddr));
    return Rewritten;
  };

  struct MutationCase {
    const char *Name;
    Mutation Kind;
    bool IsMalformed;
  };
  const MutationCase Cases[] = {
      {"unsupported version", Mutation::UnsupportedVersion, true},
      {"unmarked pool", Mutation::Unmarked, false},
      {"duplicate note", Mutation::DuplicateNote, true},
      {"align-8 note", Mutation::Align8Note, true},
      {"missing name NUL", Mutation::MissingNameNul, true},
      {"enlarged text load", Mutation::EnlargedTextLoad, false},
      {"pool file alias", Mutation::PoolFileAliasesText, false},
      {"pool vaddr overlap", Mutation::PoolVAddrOverlapsText, false},
  };
  for (const MutationCase &Case : Cases) {
    SCOPED_TRACE(Case.Name);
    std::unique_ptr<llvm::WritableMemoryBuffer> Buffer = CloneBuffer(*A0);
    ASSERT_NE(Buffer, nullptr);
    ASSERT_TRUE(Mutate(*Buffer, Case.Kind));

    uint8_t *Data = reinterpret_cast<uint8_t *>(Buffer->getBufferStart());
    llvm::Expected<ElfView> View =
        ElfView::create(Data, Buffer->getBufferSize());
    ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
    std::optional<bool> Compatible =
        View->executableCodeOutsideTextIsCompatibleWith(
            ExecutablePoolTargetState::A0);
    if (Case.IsMalformed) {
      EXPECT_FALSE(Compatible.has_value());
    } else {
      ASSERT_TRUE(Compatible.has_value());
      EXPECT_FALSE(*Compatible);
    }
  }
}

// gfx1250 "address of a global" idiom, as emitted for e.g. `x++` on a
// __managed__/__device__ global and observed in GPU_func of the
// Unit_hipModuleGetGlobal_Functional reproducer:
//
//   s_get_pc_i64 s[0:1]                         ; s[0:1] = addr of next insn
//   s_add_nc_u64 s[0:1], s[0:1], lit64(delta)   ; s[0:1] = that addr + delta
//
// The 64-bit literal is baked at link time (no relocation) and encodes the
// distance from the s_add instruction to the referenced symbol. Reading it back
// out of .text is exactly how the hardware resolves the global's address, so it
// is the ground truth any rewrite must keep consistent with the symbol table.
namespace {
constexpr uint8_t SGetPcI64SS01[4] = {0xBE, 0x80, 0x47, 0x00};
constexpr uint8_t SAddNcU64Lit[4] = {0xA9, 0x80, 0xFE, 0x00};
constexpr size_t GetPcOffset = 0;
constexpr size_t AddOpOffset = 4; // s_get_pc_i64 is one dword
constexpr size_t Lit64Offset = 8; // + s_add_nc_u64 opcode dword
constexpr size_t RefSeqSize = 16; // + 8-byte lit64

// Build a .text image containing the reference idiom. The literal is computed
// so that, loaded at TextAddr, the ISA resolves the reference to TargetVAddr.
std::vector<uint8_t> makeTextReferencing(uint64_t TextAddr,
                                         uint64_t TargetVAddr) {
  std::vector<uint8_t> Text(RefSeqSize, 0);
  std::memcpy(Text.data() + GetPcOffset, SGetPcI64SS01, sizeof(SGetPcI64SS01));
  std::memcpy(Text.data() + AddOpOffset, SAddNcU64Lit, sizeof(SAddNcU64Lit));
  // s_get_pc_i64 returns the address of the *following* instruction (the
  // s_add), so the PC base the add works from is TextAddr + AddOpOffset.
  const uint64_t PcBase = TextAddr + AddOpOffset;
  const uint64_t Lit = TargetVAddr - PcBase; // two's-complement; forward here
  std::memcpy(Text.data() + Lit64Offset, &Lit, sizeof(Lit));
  return Text;
}

// Decode the reference idiom out of a .text image loaded at TextAddr and return
// the virtual address the ISA resolves it to.
uint64_t decodeReferencedVAddr(const uint8_t *Text, uint64_t TextAddr) {
  uint64_t Lit = 0;
  std::memcpy(&Lit, Text + Lit64Offset, sizeof(Lit));
  return TextAddr + AddOpOffset + Lit;
}
} // namespace

// The real invariant: after appending trampolines, the address the *ISA*
// resolves a global reference to (decoded from the PC-relative literal in
// .text) must equal the address the *symbol table* reports for that global.
//
// This is the ELF-layer reproduction of Unit_hipModuleGetGlobal_Functional: the
// entry-trampoline rewrite grew .text, which shifted the referenced symbol by
// the trampoline size while leaving the baked literal pointing at the old
// location, so the ISA and the symbol table disagreed and the kernel
// dereferenced the wrong address. Here the descriptor symbol lives in .rodata
// (after .text), standing in for a global in a post-.text data section.
TEST(ElfView, GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol) {
  // One 256-byte entry stub, matching the real
  // Unit_hipModuleGetGlobal_Functional reproducer's 0x100 shift.
  static constexpr uint64_t GrowthBytes = 0x100;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  // .text (at TextAddr) references the descriptor symbol in .rodata (at
  // RodataAddr), which sits after .text -- like a kernel referencing a global
  // in a post-.text data section.
  std::vector<uint8_t> Text =
      makeTextReferencing(Opts.TextAddr, Opts.RodataAddr);
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // Sanity: before the rewrite, the ISA reference and the symbol agree.
  ASSERT_EQ(decodeReferencedVAddr(ViewOrErr->textData(), ViewOrErr->textAddr()),
            Obj.RodataAddr);

  Trampoline T;
  T.Bytes.assign(GrowthBytes, 0);
  std::vector<Trampoline> Trampolines{T};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(
          Trampolines, SNop, ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  // Decode whatever is now in .text into the address the ISA resolves the
  // reference to...
  const uint64_t IsaResolved =
      decodeReferencedVAddr(OutView->textData(), OutView->textAddr());
  // ...vs. the address the symbol table now reports for the same global.
  llvm::ArrayRef<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  const uint64_t SymbolVAddr = KDs[0].VAddr;

  EXPECT_EQ(IsaResolved, SymbolVAddr);
}

// A fully-linked code object's DWARF encodes *absolute* virtual addresses of
// code and data (DW_AT_low_pc, DW_OP_addr, the .debug_addr pool, .debug_line
// set_address) with no relocations. The old growWithTrampolines shifted
// post-.text virtual addresses and symbols but left .debug_* contents
// untouched, so any such address went stale by the trampoline size -- the
// debugger would resolve a global to the pre-shift location. This is the
// debug-info analogue of the ISA-literal corruption above; the no-shift model
// keeps both in agreement.
namespace {

// Minimal ET_DYN AMDGPU object:
//   [1] .text        (alloc, exec)  at TextAddr
//   [2] .data        (alloc, write) at DataAddr -- holds global "g"
//   [3] .debug_info  (non-alloc)    -- 8-byte absolute address DWARF would
//                                       encode for "g" (stands in for a
//                                       DW_AT_low_pc / DW_OP_addr / .debug_addr
//                                       entry)
//   [4] .symtab  [5] .strtab  [6] .shstrtab
// Both .data and .debug_info follow .text.
struct DwarfRefElf {
  std::vector<uint8_t> Bytes;
  uint64_t DataAddr = 0;
  unsigned DebugSectionIndex = 3;
  unsigned SymtabSectionIndex = 4;
  unsigned GlobalSymIndex = 1;
};

DwarfRefElf makeDwarfRefElf(uint64_t TextAddr = 0x1000,
                            uint64_t DataAddr = 0x2000) {
  using namespace llvm::ELF;
  constexpr uint64_t TextSize = 16;
  constexpr unsigned SymCount = 2; // null + "g"

  static const char ShStr[] =
      "\0.text\0.data\0.debug_info\0.symtab\0.strtab\0.shstrtab\0";
  constexpr uint32_t NameText = 1;
  constexpr uint32_t NameData = 7;
  constexpr uint32_t NameDebug = 13;
  constexpr uint32_t NameSymtab = 25;
  constexpr uint32_t NameStrtab = 33;
  constexpr uint32_t NameShstrtab = 41;

  static const char Str[] = "\0g\0";
  constexpr uint32_t GNameOff = 1;

  constexpr unsigned ShNum = 7;
  const uint64_t PhOff = sizeof(Elf64_Ehdr);
  const uint64_t ShOff =
      comgr_test::alignTo8(PhOff + sizeof(Elf64_Phdr));
  const uint64_t TextOff =
      comgr_test::alignTo8(ShOff + ShNum * sizeof(Elf64_Shdr));
  const uint64_t DataOff = comgr_test::alignTo8(TextOff + TextSize);
  const uint64_t DebugOff = comgr_test::alignTo8(DataOff + 8);
  const uint64_t StrOff = comgr_test::alignTo8(DebugOff + 8);
  const uint64_t SymOff = comgr_test::alignTo8(StrOff + sizeof(Str));
  const uint64_t ShStrOff =
      comgr_test::alignTo8(SymOff + SymCount * sizeof(Elf64_Sym));
  const uint64_t BufSize = comgr_test::alignTo8(ShStrOff + sizeof(ShStr) + 64);

  DwarfRefElf R;
  R.Bytes.assign(BufSize, 0);
  R.DataAddr = DataAddr;
  uint8_t *B = R.Bytes.data();

  Elf64_Ehdr Ehdr = comgr_test::makeElf64Ehdr(EM_AMDGPU);
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_DYN;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_phoff = PhOff;
  Ehdr.e_phentsize = sizeof(Elf64_Phdr);
  Ehdr.e_phnum = 1;
  Ehdr.e_shoff = ShOff;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = ShNum;
  Ehdr.e_shstrndx = 6;
  std::memcpy(B, &Ehdr, sizeof(Ehdr));

  Elf64_Phdr TextLoad{};
  TextLoad.p_type = PT_LOAD;
  TextLoad.p_flags = PF_R | PF_X;
  TextLoad.p_offset = TextOff;
  TextLoad.p_vaddr = TextAddr;
  TextLoad.p_paddr = TextAddr;
  TextLoad.p_filesz = TextSize;
  TextLoad.p_memsz = TextSize;
  TextLoad.p_align = 4;
  std::memcpy(B + PhOff, &TextLoad, sizeof(TextLoad));

  std::memcpy(B + StrOff, Str, sizeof(Str));
  std::memcpy(B + ShStrOff, ShStr, sizeof(ShStr));

  // The absolute address DWARF encodes for "g".
  const uint64_t DebugAddr = DataAddr;
  std::memcpy(B + DebugOff, &DebugAddr, sizeof(DebugAddr));

  auto writeShdr = [&](unsigned Idx, const Elf64_Shdr &Sh) {
    std::memcpy(B + ShOff + Idx * sizeof(Elf64_Shdr), &Sh, sizeof(Sh));
  };

  Elf64_Shdr Text{};
  Text.sh_name = NameText;
  Text.sh_type = SHT_PROGBITS;
  Text.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  Text.sh_offset = TextOff;
  Text.sh_addr = TextAddr;
  Text.sh_size = TextSize;
  Text.sh_addralign = 4;
  writeShdr(1, Text);

  Elf64_Shdr Data{};
  Data.sh_name = NameData;
  Data.sh_type = SHT_PROGBITS;
  Data.sh_flags = SHF_ALLOC | SHF_WRITE;
  Data.sh_offset = DataOff;
  Data.sh_addr = DataAddr;
  Data.sh_size = 8;
  Data.sh_addralign = 8;
  writeShdr(2, Data);

  Elf64_Shdr Debug{};
  Debug.sh_name = NameDebug;
  Debug.sh_type = SHT_PROGBITS;
  Debug.sh_flags = 0; // non-alloc, like real .debug_* sections
  Debug.sh_offset = DebugOff;
  Debug.sh_size = 8;
  Debug.sh_addralign = 1;
  writeShdr(3, Debug);

  Elf64_Shdr Symtab{};
  Symtab.sh_name = NameSymtab;
  Symtab.sh_type = SHT_SYMTAB;
  Symtab.sh_offset = SymOff;
  Symtab.sh_size = SymCount * sizeof(Elf64_Sym);
  Symtab.sh_link = 5; // .strtab
  Symtab.sh_info = 1;
  Symtab.sh_entsize = sizeof(Elf64_Sym);
  writeShdr(4, Symtab);

  Elf64_Shdr Strtab{};
  Strtab.sh_name = NameStrtab;
  Strtab.sh_type = SHT_STRTAB;
  Strtab.sh_offset = StrOff;
  Strtab.sh_size = sizeof(Str);
  writeShdr(5, Strtab);

  Elf64_Shdr Shstr{};
  Shstr.sh_name = NameShstrtab;
  Shstr.sh_type = SHT_STRTAB;
  Shstr.sh_offset = ShStrOff;
  Shstr.sh_size = sizeof(ShStr);
  writeShdr(6, Shstr);

  Elf64_Sym G{};
  G.st_name = GNameOff;
  G.setBindingAndType(STB_GLOBAL, STT_OBJECT);
  G.st_shndx = 2; // .data
  G.st_value = DataAddr;
  G.st_size = 8;
  std::memcpy(B + SymOff + 1 * sizeof(Elf64_Sym), &G, sizeof(G));

  return R;
}

// Read an 8-byte value from section [Idx] at intra-section byte offset Off.
uint64_t readSectionU64(const ElfView &V, unsigned Idx, uint64_t Off) {
  uint64_t Val = 0;
  std::memcpy(
      &Val, V.data() + static_cast<uint64_t>(V.sections()[Idx].sh_offset) + Off,
      sizeof(Val));
  return Val;
}

// Read st_value of symbol SymIdx in the symbol table at section [SymtabIdx].
uint64_t readSymbolValue(const ElfView &V, unsigned SymtabIdx,
                         unsigned SymIdx) {
  llvm::ELF::Elf64_Sym Sym{};
  std::memcpy(&Sym,
              V.data() +
                  static_cast<uint64_t>(V.sections()[SymtabIdx].sh_offset) +
                  SymIdx * sizeof(llvm::ELF::Elf64_Sym),
              sizeof(Sym));
  return Sym.st_value;
}

} // namespace

// The invariant: after appending trampolines, the address the object's DWARF
// encodes for a global must still equal the address the symbol table reports
// for it. The no-shift model leaves both untouched, so they agree (the old
// shift moved the symbol but not the DWARF address). Debug-info analogue of
// GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol.
TEST(ElfView, GrowWithTrampolinesKeepsDwarfConsistentWithSymbol) {
  static constexpr uint64_t GrowthBytes = 0x100;

  DwarfRefElf Obj = makeDwarfRefElf();

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // Sanity: before the rewrite, DWARF and the symbol agree.
  ASSERT_EQ(readSectionU64(*ViewOrErr, Obj.DebugSectionIndex, 0), Obj.DataAddr);
  ASSERT_EQ(
      readSymbolValue(*ViewOrErr, Obj.SymtabSectionIndex, Obj.GlobalSymIndex),
      Obj.DataAddr);

  Trampoline T;
  T.Bytes.assign(GrowthBytes, 0);
  std::vector<Trampoline> Trampolines{T};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(
          Trampolines, SNop, ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  // Address DWARF still encodes for "g" ...
  const uint64_t DwarfAddr = readSectionU64(*OutView, Obj.DebugSectionIndex, 0);
  // ... vs. the address the symbol table now reports for "g".
  const uint64_t SymbolVAddr =
      readSymbolValue(*OutView, Obj.SymtabSectionIndex, Obj.GlobalSymIndex);

  EXPECT_EQ(DwarfAddr, SymbolVAddr);
}

// Stub symbols must describe the same entry as the rewritten kernel descriptor
// and the appended executable pool. Growing the non-allocating symbol tables
// must not move or invalidate that pool's PT_LOAD mapping.
TEST(ElfView, AddKernelEntryTrampolineSymbolMatchesDescriptorAndPool) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr = createElfView(Obj);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::optional<uint64_t> PoolVAddr = ViewOrErr->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddr.has_value());

  Trampoline Stub;
  Stub.Bytes.assign(KernelEntryStubStride, 0xa5);
  std::vector<Trampoline> Growth{Stub};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      ViewOrErr->growWithTrampolines(
          Growth, SNop, ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Grown, nullptr);

  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"entry_kernel", /*StubTextOffset=*/0, /*RequiredSgprs=*/10,
       /*InstPrefLines=*/0},
  };
  uint8_t *GrownData =
      reinterpret_cast<uint8_t *>(Grown->getBufferStart());
  llvm::Expected<ElfView> GrownView =
      ElfView::create(GrownData, Grown->getBufferSize());
  ASSERT_TRUE((bool)GrownView) << llvm::toString(GrownView.takeError());
  std::optional<uint64_t> KdVAddr =
      GrownView->getKernelDescriptorVAddr("entry_kernel");
  ASSERT_TRUE(KdVAddr.has_value());
  ASSERT_GE(*PoolVAddr, *KdVAddr);
  ASSERT_LE(*PoolVAddr - *KdVAddr,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  ASSERT_TRUE(GrownView->updateKernelDescriptorEntryOffset(
      "entry_kernel", static_cast<int64_t>(*PoolVAddr - *KdVAddr)));

  using ELFT = llvm::object::ELF64LE;
  llvm::ELF::Elf64_Ehdr GrownHeader{};
  std::memcpy(&GrownHeader, Grown->getBufferStart(), sizeof(GrownHeader));
  std::unique_ptr<llvm::WritableMemoryBuffer> WithSyms =
      addKernelEntryTrampolineSymbols(*Grown, *PoolVAddr, Fixups);
  ASSERT_NE(WithSyms, nullptr);

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
  std::optional<unsigned> PoolSectionIndex;
  for (unsigned I = 0; I != SecsOrErr->size(); ++I) {
    const ELFT::Shdr &S = (*SecsOrErr)[I];
    if (S.sh_type == llvm::ELF::SHT_SYMTAB) {
      SymtabShdr = &S;
      continue;
    }
    if (S.sh_type == llvm::ELF::SHT_PROGBITS &&
        (S.sh_flags & (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR)) ==
            (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR) &&
        S.sh_addr == *PoolVAddr && S.sh_size >= KernelEntryStubStride)
      PoolSectionIndex = I;
  }
  ASSERT_NE(SymtabShdr, nullptr);
  ASSERT_TRUE(PoolSectionIndex.has_value());
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

  const ELFT::Sym *Sym = FindSym("entry_kernel.stub");
  ASSERT_NE(Sym, nullptr);
  EXPECT_EQ(static_cast<unsigned>(Sym->getType()),
            static_cast<unsigned>(llvm::ELF::STT_FUNC));
  EXPECT_EQ(Sym->st_shndx, *PoolSectionIndex);
  EXPECT_EQ(Sym->st_value, *PoolVAddr);
  EXPECT_EQ(Sym->st_size, KernelEntryStubStride);

  uint8_t *WithSymsData =
      reinterpret_cast<uint8_t *>(WithSyms->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(WithSymsData, WithSyms->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  llvm::ArrayRef<KernelDescriptorInfo> Descriptors =
      OutView->kernelDescriptors();
  ASSERT_EQ(Descriptors.size(), 1u);
  ASSERT_GE(Descriptors.front().EntryOffset, 0);
  const uint64_t DescriptorEntry =
      Descriptors.front().VAddr +
      static_cast<uint64_t>(Descriptors.front().EntryOffset);
  EXPECT_EQ(Sym->st_value, DescriptorEntry);

  llvm::ELF::Elf64_Ehdr OutputHeader{};
  std::memcpy(&OutputHeader, Data, sizeof(OutputHeader));
  EXPECT_EQ(OutputHeader.e_phoff, GrownHeader.e_phoff);
  EXPECT_EQ(OutputHeader.e_phnum, GrownHeader.e_phnum);
  EXPECT_EQ(OutputHeader.e_phentsize, GrownHeader.e_phentsize);
  const uint64_t PhdrBytes =
      static_cast<uint64_t>(OutputHeader.e_phnum) * OutputHeader.e_phentsize;
  ASSERT_LE(OutputHeader.e_phoff, WithSyms->getBufferSize());
  ASSERT_LE(PhdrBytes, WithSyms->getBufferSize() - OutputHeader.e_phoff);
  ASSERT_LE(GrownHeader.e_phoff, Grown->getBufferSize());
  ASSERT_LE(PhdrBytes, Grown->getBufferSize() - GrownHeader.e_phoff);
  EXPECT_EQ(std::memcmp(Data + OutputHeader.e_phoff,
                        Grown->getBufferStart() + GrownHeader.e_phoff,
                        PhdrBytes),
            0);

  llvm::Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  ASSERT_TRUE((bool)PhdrsOrErr) << llvm::toString(PhdrsOrErr.takeError());
  const ELFT::Phdr *PoolPhdr = nullptr;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr)
    if (Phdr.p_type == llvm::ELF::PT_LOAD &&
        (Phdr.p_flags & (llvm::ELF::PF_R | llvm::ELF::PF_X)) ==
            (llvm::ELF::PF_R | llvm::ELF::PF_X) &&
        Phdr.p_vaddr == *PoolVAddr &&
        Phdr.p_filesz >= KernelEntryStubStride) {
      PoolPhdr = &Phdr;
      break;
    }
  ASSERT_NE(PoolPhdr, nullptr);
  ASSERT_LE(PoolPhdr->p_offset, WithSyms->getBufferSize());
  ASSERT_LE(KernelEntryStubStride,
            WithSyms->getBufferSize() - PoolPhdr->p_offset);
  EXPECT_EQ(std::memcmp(Data + PoolPhdr->p_offset, Stub.Bytes.data(),
                        KernelEntryStubStride),
            0);
}

// A zero-sized allocatable section may share the trampoline pool's base.
// Stub symbols must name the real executable pool section that fully contains
// the stub, not the empty section at the same address.
TEST(ElfView, AddKernelEntryTrampolineSymbolsSkipsZeroSizedSectionAtPoolVAddr) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.ExtraAllocSectionAddr = 0x3000;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  std::optional<uint64_t> PoolVAddr = View->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddr.has_value());
  ASSERT_EQ(*PoolVAddr, 0x3000u);

  Trampoline Stub;
  Stub.Bytes.assign(KernelEntryStubStride, 0xaa);
  std::vector<Trampoline> Growth{Stub};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      View->growWithTrampolines(
          Growth, SNop, ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Grown, nullptr);

  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"entry_kernel", /*StubTextOffset=*/0, /*RequiredSgprs=*/10,
       /*InstPrefLines=*/0}};
  std::unique_ptr<llvm::WritableMemoryBuffer> WithSyms =
      addKernelEntryTrampolineSymbols(*Grown, *PoolVAddr, Fixups);
  ASSERT_NE(WithSyms, nullptr);

  using ELFT = llvm::object::ELF64LE;
  llvm::Expected<llvm::object::ELFFile<ELFT>> FileOrErr =
      llvm::object::ELFFile<ELFT>::create(llvm::StringRef(
          WithSyms->getBufferStart(), WithSyms->getBufferSize()));
  ASSERT_TRUE((bool)FileOrErr) << llvm::toString(FileOrErr.takeError());
  llvm::Expected<ELFT::ShdrRange> Sections = FileOrErr->sections();
  ASSERT_TRUE((bool)Sections) << llvm::toString(Sections.takeError());

  const ELFT::Shdr *Symtab = nullptr;
  for (const ELFT::Shdr &Section : *Sections)
    if (Section.sh_type == llvm::ELF::SHT_SYMTAB) {
      Symtab = &Section;
      break;
    }
  ASSERT_NE(Symtab, nullptr);
  llvm::Expected<ELFT::SymRange> Symbols = FileOrErr->symbols(Symtab);
  ASSERT_TRUE((bool)Symbols) << llvm::toString(Symbols.takeError());
  llvm::Expected<llvm::StringRef> Strings =
      FileOrErr->getStringTableForSymtab(*Symtab);
  ASSERT_TRUE((bool)Strings) << llvm::toString(Strings.takeError());

  const ELFT::Sym *StubSym = nullptr;
  for (const ELFT::Sym &Sym : *Symbols) {
    llvm::Expected<llvm::StringRef> Name = Sym.getName(*Strings);
    ASSERT_TRUE((bool)Name) << llvm::toString(Name.takeError());
    if (*Name == "entry_kernel.stub") {
      StubSym = &Sym;
      break;
    }
  }
  ASSERT_NE(StubSym, nullptr);
  ASSERT_LT(StubSym->st_shndx, Sections->size());
  const ELFT::Shdr &StubSection = (*Sections)[StubSym->st_shndx];
  EXPECT_EQ(StubSym->st_value, *PoolVAddr);
  EXPECT_NE(StubSection.sh_size, 0u);
  EXPECT_EQ(StubSection.sh_type, llvm::ELF::SHT_PROGBITS);
  EXPECT_EQ(StubSection.sh_flags &
                (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR),
            llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR);
  EXPECT_GE(StubSym->st_value, StubSection.sh_addr);
  EXPECT_LE(StubSym->st_value + StubSym->st_size,
            StubSection.sh_addr + StubSection.sh_size);
}

TEST(ElfView, AddKernelEntryTrampolineSymbolsRejectsLinkedShndxTable) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);
  llvm::Expected<ElfView> View = createElfView(Obj);
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  std::optional<uint64_t> PoolVAddr = View->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddr.has_value());

  Trampoline Stub;
  Stub.Bytes.assign(KernelEntryStubStride, 0);
  std::vector<Trampoline> Growth{Stub};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      View->growWithTrampolines(Growth, SNop,
                                ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Grown, nullptr);

  llvm::ELF::Elf64_Ehdr Header{};
  std::memcpy(&Header, Grown->getBufferStart(), sizeof(Header));
  std::optional<unsigned> SymtabIndex;
  std::optional<unsigned> NoteIndex;
  for (unsigned I = 0; I != Header.e_shnum; ++I) {
    llvm::ELF::Elf64_Shdr Shdr{};
    std::memcpy(&Shdr,
                Grown->getBufferStart() + Header.e_shoff +
                    static_cast<uint64_t>(I) * Header.e_shentsize,
                sizeof(Shdr));
    if (Shdr.sh_type == llvm::ELF::SHT_SYMTAB)
      SymtabIndex = I;
    else if (Shdr.sh_type == llvm::ELF::SHT_NOTE)
      NoteIndex = I;
  }
  ASSERT_TRUE(SymtabIndex.has_value());
  ASSERT_TRUE(NoteIndex.has_value());
  llvm::ELF::Elf64_Shdr Shndx{};
  std::memcpy(&Shndx,
              Grown->getBufferStart() + Header.e_shoff +
                  static_cast<uint64_t>(*NoteIndex) * Header.e_shentsize,
              sizeof(Shndx));
  Shndx.sh_type = llvm::ELF::SHT_SYMTAB_SHNDX;
  Shndx.sh_link = *SymtabIndex;
  std::memcpy(Grown->getBufferStart() + Header.e_shoff +
                  static_cast<uint64_t>(*NoteIndex) * Header.e_shentsize,
              &Shndx, sizeof(Shndx));

  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"entry_kernel", /*StubTextOffset=*/0, /*RequiredSgprs=*/8,
       /*InstPrefLines=*/0},
  };
  EXPECT_EQ(addKernelEntryTrampolineSymbols(*Grown, *PoolVAddr, Fixups),
            nullptr);
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

  // Prime the metadata cache before the in-place update.
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 8u);
  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/true));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 10u);
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
}

TEST(ElfView, ReadsWorkgroupCapacityMetadata) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataVgprCount = 120;
  Opts.MetadataMaxFlatWorkgroupSize = 1024;
  Opts.MetadataWavefrontSize = 32;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelMetadataVgprCount("entry_kernel"), 120u);
  EXPECT_EQ(ViewOrErr->getKernelMaxFlatWorkgroupSize("entry_kernel"), 1024u);
  EXPECT_EQ(ViewOrErr->getKernelWavefrontSize("entry_kernel"), 32u);
}

TEST(ElfView, UpdateKernelDescriptorVgprCountIsChecked) {
  static constexpr unsigned Granule = 16;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(readReservedVgprs(Obj.Bytes, Obj.KernelDescriptorOffset, Granule),
            16u);
  ASSERT_TRUE(
      ViewOrErr->updateKernelDescriptorVgprCount("entry_kernel", 17, Granule));
  EXPECT_EQ(readReservedVgprs(Obj.Bytes, Obj.KernelDescriptorOffset, Granule),
            32u);
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorVgprCount("entry_kernel", 4096,
                                                          Granule));
  EXPECT_EQ(readReservedVgprs(Obj.Bytes, Obj.KernelDescriptorOffset, Granule),
            32u);
}

TEST(ElfView, UpdateKernelMetadataVgprCountsUpdatesInPlace) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataVgprCount = 9;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  llvm::StringMap<unsigned> RequiredVgprs;
  RequiredVgprs.try_emplace("entry_kernel", 10u);
  ASSERT_TRUE(ViewOrErr->updateKernelMetadataVgprCounts(RequiredVgprs));
  EXPECT_EQ(ViewOrErr->getKernelMetadataVgprCount("entry_kernel"), 10u);
}

TEST(ElfView, UpdateKernelMetadataVgprCountsRejectsMissingCount) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataOmitVgprCount = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  llvm::StringMap<unsigned> RequiredVgprs;
  RequiredVgprs.try_emplace("entry_kernel", 10u);
  EXPECT_FALSE(ViewOrErr->updateKernelMetadataVgprCounts(RequiredVgprs));
}

TEST(ElfView, UpdateGfx1250RevisionMetadataRetagsKernelInPlace) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 8;
  Opts.MetadataGfx1250Revision = "B0";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::StringRef Before(reinterpret_cast<const char *>(Obj.Bytes.data()),
                         Obj.Bytes.size());
  EXPECT_NE(Before.find("B0"), llvm::StringRef::npos);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ASSERT_TRUE(ViewOrErr->updateGfx1250RevisionMetadata("A0"));

  llvm::StringRef After(reinterpret_cast<const char *>(Obj.Bytes.data()),
                        Obj.Bytes.size());
  EXPECT_EQ(After.find("B0"), llvm::StringRef::npos);
  EXPECT_NE(After.find("A0"), llvm::StringRef::npos);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountCanUpdateMetadataOnly) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr = createElfView(Obj);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  const unsigned DescriptorSgprsBefore =
      readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset);

  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/false));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 10u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset),
            DescriptorSgprsBefore);
}

TEST(ElfView, UpdateKernelMetadataSgprCountsKeepsPrimedCacheCoherent) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr = createElfView(Obj);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 8u);

  llvm::StringMap<unsigned> RequiredSgprs;
  RequiredSgprs.try_emplace("entry_kernel", 10u);
  ASSERT_TRUE(ViewOrErr->updateKernelMetadataSgprCounts(RequiredSgprs));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 10u);
}

TEST(ElfView, UpdateKernelMetadataSgprCountsBatchesMixedRequirements) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataKernels = {{"needs_update", 8}, {"already_enough", 16}};
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  llvm::StringMap<unsigned> RequiredSgprs;
  RequiredSgprs.try_emplace("needs_update", 10u);
  RequiredSgprs.try_emplace("already_enough", 12u);
  ASSERT_TRUE(ViewOrErr->updateKernelMetadataSgprCounts(RequiredSgprs));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("needs_update"), 10u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("already_enough"), 16u);
}

TEST(ElfView, UpdateKernelMetadataSgprCountsRejectsAbsentKernelAtomically) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataKernels = {{"needs_update", 8}, {"already_enough", 16}};
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  llvm::StringMap<unsigned> RequiredSgprs;
  RequiredSgprs.try_emplace("needs_update", 10u);
  RequiredSgprs.try_emplace("already_enough", 12u);
  RequiredSgprs.try_emplace("absent_kernel", 4u);
  EXPECT_FALSE(ViewOrErr->updateKernelMetadataSgprCounts(RequiredSgprs));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("needs_update"), 8u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("already_enough"), 16u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("absent_kernel"), std::nullopt);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountMetadataOnlyRequiresMetadata) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr = createElfView(Obj);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  const unsigned DescriptorSgprsBefore =
      readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset);

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/false));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset),
            DescriptorSgprsBefore);
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

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/true));
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
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/true));
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
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/true));
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

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 128, /*UpdateDescriptor=*/true));
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

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 100000, /*UpdateDescriptor=*/true));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 200u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

// -- kernel-descriptor cache dedup --------------------------------------------

// A kernel name is the key used by metadata and descriptor fixups, so it must
// resolve to one location. Repeated identical symbols are harmless, but a
// second location for the same name makes the cache incomplete and prevents a
// rewrite from updating only one of the descriptors.
TEST(ElfView, KernelDescriptorCacheRejectsAmbiguousNameAcrossOrdering) {
  comgr_test::MultiKernelDescriptorElfOptions Opts;
  Opts.TextAddr = 0x1000;
  Opts.TextSize = 0x400;
  Opts.RodataAddr = 0x2000;
  // Distinct descriptor locations for the two "A" instances, with a "B" in
  // between and an identical repeat of the first A at the end.
  Opts.Kernels = {
      {"kern_a", 0x1000, 0x2000, /*EntryOffset=*/-0x1000},
      {"kern_b", 0x1100, 0x2100, /*EntryOffset=*/-0x1000},
      {"kern_a", 0x1200, 0x2200, /*EntryOffset=*/-0x1000},
      {"kern_a", 0x1000, 0x2000, /*EntryOffset=*/-0x1000}, // dup of first
  };
  std::vector<uint8_t> Bytes = comgr_test::makeMultiKernelDescriptorElf(Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Bytes.data(), Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  llvm::ArrayRef<KernelDescriptorInfo> KDs = ViewOrErr->kernelDescriptors();
  EXPECT_FALSE(ViewOrErr->kernelDescriptorCacheIsComplete());
  ASSERT_EQ(KDs.size(), 2u);
  EXPECT_EQ(KDs[0].KernelName, "kern_a");
  EXPECT_EQ(KDs[0].VAddr, 0x2000u);
  EXPECT_EQ(KDs[1].KernelName, "kern_b");
  EXPECT_EQ(KDs[1].VAddr, 0x2100u);
}
