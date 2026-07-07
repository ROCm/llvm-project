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

// -- findNearestSled ----------------------------------------------------------

TEST(FindNearestSled, SkipsSledsOutsideInstructionFunctionRange) {
  std::vector<NopSled> Sleds;
  Sleds.push_back({0, 32, 0, 0, 32});
  Sleds.push_back({96, 128, 96, 96, 160});

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

TEST(ElfView, AppendExecutableSegmentPreservesAllocSectionSymbols) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::vector<KernelDescriptorInfo> InputKDs = ViewOrErr->kernelDescriptors();
  ASSERT_EQ(InputKDs.size(), 1u);

  llvm::Expected<ElfView::ELFT::PhdrRange> InputPhdrsOrErr =
      ViewOrErr->file().program_headers();
  ASSERT_TRUE((bool)InputPhdrsOrErr)
      << llvm::toString(InputPhdrsOrErr.takeError());
  unsigned InputLoadCount = 0;
  for (const ElfView::ELFT::Phdr &Phdr : *InputPhdrsOrErr) {
    if (Phdr.p_type == llvm::ELF::PT_LOAD)
      ++InputLoadCount;
  }

  std::optional<ExecutableSegmentPlan> Plan = ViewOrErr->planExecutableSegment(
      KernelEntryStubStride, KernelEntryStubSegmentAlign, 0);
  ASSERT_TRUE(Plan.has_value());
  llvm::SmallVector<uint8_t> Payload(KernelEntryStubStride, 0xcc);
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->appendExecutableSegment(Payload, *Plan, ".hotswap.entry");
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  EXPECT_EQ(OutView->textAddr(), ViewOrErr->textAddr());
  EXPECT_EQ(OutView->textSize(), ViewOrErr->textSize());
  std::vector<KernelDescriptorInfo> OutputKDs = OutView->kernelDescriptors();
  ASSERT_EQ(OutputKDs.size(), 1u);
  EXPECT_EQ(OutputKDs[0].VAddr, InputKDs[0].VAddr);
  EXPECT_EQ(OutputKDs[0].EntryOffset, InputKDs[0].EntryOffset);

  llvm::Expected<ElfView::ELFT::PhdrRange> OutputPhdrsOrErr =
      OutView->file().program_headers();
  ASSERT_TRUE((bool)OutputPhdrsOrErr)
      << llvm::toString(OutputPhdrsOrErr.takeError());
  unsigned OutputLoadCount = 0;
  bool FoundNewLoad = false;
  for (const ElfView::ELFT::Phdr &Phdr : *OutputPhdrsOrErr) {
    if (Phdr.p_type != llvm::ELF::PT_LOAD)
      continue;
    ++OutputLoadCount;
    if (Phdr.p_vaddr != Plan->SegmentVAddr)
      continue;
    FoundNewLoad = true;
    EXPECT_EQ(Phdr.p_flags, llvm::ELF::PF_R | llvm::ELF::PF_X);
    EXPECT_GE(Phdr.p_filesz, Plan->PayloadOffset + Payload.size());
  }
  EXPECT_EQ(OutputLoadCount, InputLoadCount + 1);
  EXPECT_TRUE(FoundNewLoad);

  bool FoundPayloadSection = false;
  for (const ElfView::ELFT::Shdr &Shdr : OutView->sections()) {
    llvm::Expected<llvm::StringRef> NameOrErr =
        OutView->file().getSectionName(Shdr);
    ASSERT_TRUE((bool)NameOrErr) << llvm::toString(NameOrErr.takeError());
    if (*NameOrErr != ".hotswap.entry")
      continue;
    FoundPayloadSection = true;
    EXPECT_EQ(Shdr.sh_type, llvm::ELF::SHT_PROGBITS);
    EXPECT_EQ(Shdr.sh_flags, llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR);
    EXPECT_EQ(Shdr.sh_addr, Plan->PayloadVAddr);
    EXPECT_EQ(Shdr.sh_size, Payload.size());
  }
  EXPECT_TRUE(FoundPayloadSection);
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
