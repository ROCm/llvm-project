//===- HotswapToolTest.cpp - Unit tests for the HSA_TOOLS_LIB tool --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Covers the pure device-detection helpers the hotswap tool uses to decide
// whether a board is gfx1250 A0 (rewrite armed) or not. The rest of the tool
// is glue over the HSA loader API and is exercised on real silicon.

#include "hotswap/comgr-hotswap-tool-detect.h"
#include "gtest/gtest.h"

using namespace COMGR::hotswap;

// -- extractGfxTarget ---------------------------------------------------------

TEST(ExtractGfxTarget, PlainTargetIsReturnedUnchanged) {
  EXPECT_EQ(extractGfxTarget("gfx1250"), "gfx1250");
}

TEST(ExtractGfxTarget, FullIsaNameYieldsBareTarget) {
  EXPECT_EQ(extractGfxTarget("amdgcn-amd-amdhsa--gfx1250"), "gfx1250");
}

TEST(ExtractGfxTarget, FeatureSuffixIsDropped) {
  EXPECT_EQ(extractGfxTarget("amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-"),
            "gfx1250");
}

TEST(ExtractGfxTarget, MissingTargetYieldsEmpty) {
  EXPECT_TRUE(extractGfxTarget("amdgcn-amd-amdhsa--").empty());
  EXPECT_TRUE(extractGfxTarget("").empty());
}

// -- gateAllowsHotswap --------------------------------------------------------

TEST(GateAllowsHotswap, ArmsOnGfx1250RevisionZero) {
  EXPECT_TRUE(gateAllowsHotswap("gfx1250", /*Revision=*/0,
                                /*RevisionValid=*/true));
}

TEST(GateAllowsHotswap, DoesNotArmOnNonZeroRevision) {
  EXPECT_FALSE(gateAllowsHotswap("gfx1250", /*Revision=*/1,
                                 /*RevisionValid=*/true));
}

TEST(GateAllowsHotswap, DoesNotArmOnOtherTargets) {
  EXPECT_FALSE(gateAllowsHotswap("gfx950", /*Revision=*/0,
                                 /*RevisionValid=*/true));
}

TEST(GateAllowsHotswap, FailedRevisionQueryIsNotTreatedAsA0) {
  // RevisionValid=false must never arm, even though Revision reads 0.
  EXPECT_FALSE(gateAllowsHotswap("gfx1250", /*Revision=*/0,
                                 /*RevisionValid=*/false));
}

// -- isGfx1250CodeObject ------------------------------------------------------

// Minimal 64-bit ELF header with the given machine + e_flags.
static llvm::ELF::Elf64_Ehdr makeEhdr(uint16_t Machine, uint32_t Flags) {
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

TEST(IsGfx1250CodeObject, NullDataIsRejected) {
  EXPECT_FALSE(isGfx1250CodeObject(nullptr, 4096));
}

TEST(IsGfx1250CodeObject, TruncatedInputIsRejected) {
  llvm::ELF::Elf64_Ehdr Ehdr =
      makeEhdr(llvm::ELF::EM_AMDGPU, llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  EXPECT_FALSE(isGfx1250CodeObject(&Ehdr, sizeof(Ehdr) - 1));
}

TEST(IsGfx1250CodeObject, NonAmdgpuElfIsRejected) {
  // Same mach bits, wrong e_machine: must not be misidentified as gfx1250.
  llvm::ELF::Elf64_Ehdr Ehdr =
      makeEhdr(llvm::ELF::EM_X86_64, llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  EXPECT_FALSE(isGfx1250CodeObject(&Ehdr, sizeof(Ehdr)));
}

TEST(IsGfx1250CodeObject, WrongMachIsRejected) {
  llvm::ELF::Elf64_Ehdr Ehdr =
      makeEhdr(llvm::ELF::EM_AMDGPU, llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100);
  EXPECT_FALSE(isGfx1250CodeObject(&Ehdr, sizeof(Ehdr)));
}

TEST(IsGfx1250CodeObject, Gfx1250IsAccepted) {
  llvm::ELF::Elf64_Ehdr Ehdr =
      makeEhdr(llvm::ELF::EM_AMDGPU, llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  EXPECT_TRUE(isGfx1250CodeObject(&Ehdr, sizeof(Ehdr)));
}
