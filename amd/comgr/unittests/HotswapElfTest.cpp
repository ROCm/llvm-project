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

static constexpr uint32_t kTestBranchGFX9 = 0xBF820000u;
static constexpr uint32_t kTestBranchGFX12 = 0xBFA00000u;
static constexpr uint32_t kTestNopOpcode = 0xBF800000u;

// ── EncodeSBranch ────────────────────────────────────────────────────────────

TEST(EncodeSBranch, ForwardBranchGFX9) {
  uint8_t out[kMinInstSize] = {};
  ASSERT_TRUE(EncodeSBranch(0, 8, out, kTestBranchGFX9));
  uint32_t encoded;
  std::memcpy(&encoded, out, sizeof(encoded));
  EXPECT_EQ(encoded, 0xBF820001u);
}

TEST(EncodeSBranch, BackwardBranchGFX9) {
  uint8_t out[kMinInstSize] = {};
  ASSERT_TRUE(EncodeSBranch(16, 0, out, kTestBranchGFX9));
  uint32_t encoded;
  std::memcpy(&encoded, out, sizeof(encoded));
  EXPECT_EQ(encoded, 0xBF82FFFBu);
}

TEST(EncodeSBranch, ForwardBranchGFX12) {
  uint8_t out[kMinInstSize] = {};
  ASSERT_TRUE(EncodeSBranch(0, 8, out, kTestBranchGFX12));
  uint32_t encoded;
  std::memcpy(&encoded, out, sizeof(encoded));
  EXPECT_EQ(encoded, 0xBFA00001u);
}

TEST(EncodeSBranch, UnalignedDeltaFails) {
  uint8_t out[kMinInstSize] = {};
  EXPECT_FALSE(EncodeSBranch(0, 7, out, kTestBranchGFX9));
}

TEST(EncodeSBranch, OutOfRangeFails) {
  uint8_t out[kMinInstSize] = {};
  EXPECT_FALSE(EncodeSBranch(0, 500000, out, kTestBranchGFX9));
}

TEST(EncodeSBranch, ZeroOffsetBranch) {
  uint8_t out[kMinInstSize] = {};
  ASSERT_TRUE(EncodeSBranch(0, kMinInstSize, out, kTestBranchGFX9));
  uint32_t encoded;
  std::memcpy(&encoded, out, sizeof(encoded));
  EXPECT_EQ(encoded, kTestBranchGFX9);
}

// ── EncodeSNop ───────────────────────────────────────────────────────────────

TEST(EncodeSNop, ProducesCorrectEncoding) {
  uint8_t out[kMinInstSize] = {};
  EncodeSNop(out, kTestNopOpcode);
  uint32_t encoded;
  std::memcpy(&encoded, out, sizeof(encoded));
  EXPECT_EQ(encoded, kTestNopOpcode);
}

// ── ExtractCPU ───────────────────────────────────────────────────────────────

TEST(ExtractCPU, FullISAName) {
  EXPECT_EQ(ExtractCPU("amdgcn-amd-amdhsa--gfx1250"), "gfx1250");
}

TEST(ExtractCPU, NoGFXPrefix) {
  EXPECT_EQ(ExtractCPU("amdgcn-amd-amdhsa"), "");
}

TEST(ExtractCPU, StopsAtNonAlphanumeric) {
  EXPECT_EQ(ExtractCPU("amdgcn-amd-amdhsa--gfx1250:sramecc+"), "gfx1250");
}

// ── MallocBuffer ─────────────────────────────────────────────────────────────

TEST(MallocBuffer, AllocAndMove) {
  MallocBuffer a(64);
  ASSERT_TRUE(static_cast<bool>(a));
  EXPECT_EQ(a.size, 64u);

  uint8_t *orig = a.get();
  MallocBuffer b(std::move(a));
  EXPECT_EQ(b.get(), orig);
  EXPECT_EQ(a.get(), nullptr);
  EXPECT_EQ(a.size, 0u);
}

TEST(MallocBuffer, Release) {
  MallocBuffer buf(64);
  uint8_t *p = buf.release();
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(buf.get(), nullptr);
  EXPECT_EQ(buf.size, 0u);
  std::free(p);
}

// ── ParseElfInfo ─────────────────────────────────────────────────────────────

TEST(ParseElfInfo, RejectsTruncatedInput) {
  uint8_t garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  ElfInfo info;
  EXPECT_FALSE(ParseElfInfo(garbage, sizeof(garbage), info));
}

TEST(ParseElfInfo, RejectsNonElfInput) {
  uint8_t not_elf[64] = {};
  ElfInfo info;
  EXPECT_FALSE(ParseElfInfo(not_elf, sizeof(not_elf), info));
}
