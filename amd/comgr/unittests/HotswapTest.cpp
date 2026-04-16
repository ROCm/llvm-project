//===- HotswapTest.cpp - Unit tests for HotSwap internals -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "gtest/gtest.h"
#include <cstring>
#include <vector>

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

// ── WMMA hazard helpers ─────────────────────────────────────────────────────

TEST(ClassifyWmmaNops, DistinguishesSteppingRequirements) {
  WmmaNopReq f8x64 = ClassifyWmmaNops("v_wmma_f32_16x16x64_fp8_fp8");
  EXPECT_EQ(f8x64.b0_nops, 1);
  EXPECT_EQ(f8x64.a0_nops, 4);

  WmmaNopReq f8x128 = ClassifyWmmaNops("v_wmma_f32_16x16x128_fp8_bf8");
  EXPECT_EQ(f8x128.b0_nops, 3);
  EXPECT_EQ(f8x128.a0_nops, 4);

  WmmaNopReq f8f6f4 = ClassifyWmmaNops("v_swmmac_f32_16x16x128_f8f6f4");
  EXPECT_EQ(f8f6f4.b0_nops, 1);
  EXPECT_EQ(f8f6f4.a0_nops, 4);

  WmmaNopReq f16 = ClassifyWmmaNops("v_wmma_f32_16x16x32_f16");
  EXPECT_EQ(f16.b0_nops, 4);
  EXPECT_EQ(f16.a0_nops, 4);

  WmmaNopReq iu8 = ClassifyWmmaNops("v_wmma_i32_16x16x64_iu8");
  EXPECT_EQ(iu8.b0_nops, 8);
  EXPECT_EQ(iu8.a0_nops, 4);

  WmmaNopReq fallback = ClassifyWmmaNops("v_fma_f32");
  EXPECT_EQ(fallback.b0_nops, 4);
  EXPECT_EQ(fallback.a0_nops, 4);
}

TEST(IsValuInst, ExcludesNonValuCases) {
  EXPECT_TRUE(IsValuInst("v_fma_f32"));
  EXPECT_FALSE(IsValuInst("v_nop"));
  EXPECT_FALSE(IsValuInst("v_wmma_f32_16x16x64_fp8_fp8"));
  EXPECT_FALSE(IsValuInst("v_swmmac_f32_16x16x128_fp8_fp8"));
  EXPECT_FALSE(IsValuInst("s_add_u32"));
}

TEST(CheckVgprOverlap, UsesValuDestinationForWarHazards) {
  LLVMState llvm_state = InitLLVMCached("amdgcn-amd-amdhsa--gfx1250");
  ASSERT_TRUE(llvm_state.valid);

  auto wmma_bytes = AssembleSingleInst(
      "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7]",
      llvm_state);
  ASSERT_FALSE(wmma_bytes.empty());

  auto hazard_bytes = AssembleSingleInst("v_fma_f32 v0, v0, v1, v2",
                                         llvm_state);
  ASSERT_FALSE(hazard_bytes.empty());

  auto safe_bytes = AssembleSingleInst("v_fma_f32 v40, v0, v1, v2",
                                       llvm_state);
  ASSERT_FALSE(safe_bytes.empty());

  std::vector<InternalDecodedInst> wmma_decoded;
  ASSERT_TRUE(DecodeTextSection(wmma_bytes.data(), wmma_bytes.size(),
                                llvm_state, wmma_decoded));
  ASSERT_EQ(wmma_decoded.size(), 1u);

  std::vector<InternalDecodedInst> hazard_decoded;
  ASSERT_TRUE(DecodeTextSection(hazard_bytes.data(), hazard_bytes.size(),
                                llvm_state, hazard_decoded));
  ASSERT_EQ(hazard_decoded.size(), 1u);

  std::vector<InternalDecodedInst> safe_decoded;
  ASSERT_TRUE(DecodeTextSection(safe_bytes.data(), safe_bytes.size(),
                                llvm_state, safe_decoded));
  ASSERT_EQ(safe_decoded.size(), 1u);

  EXPECT_TRUE(CheckVgprOverlap(wmma_decoded[0].inst, hazard_decoded[0].inst,
                               *llvm_state.MRI));
  EXPECT_FALSE(CheckVgprOverlap(wmma_decoded[0].inst, safe_decoded[0].inst,
                                *llvm_state.MRI));
}

TEST(ApplyWmmaHazardPatch, InsertsVNopsViaNearbySled) {
  LLVMState llvm_state = InitLLVMCached("amdgcn-amd-amdhsa--gfx1250");
  ASSERT_TRUE(llvm_state.valid);

  auto wmma_bytes = AssembleSingleInst(
      "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7]",
      llvm_state);
  ASSERT_FALSE(wmma_bytes.empty());

  auto valu_bytes = AssembleSingleInst("v_fma_f32 v0, v0, v1, v2",
                                       llvm_state);
  ASSERT_FALSE(valu_bytes.empty());

  auto vnop_bytes = AssembleSingleInst("v_nop", llvm_state);
  ASSERT_EQ(vnop_bytes.size(), static_cast<size_t>(kMinInstSize));

  std::vector<uint8_t> text;
  text.insert(text.end(), wmma_bytes.begin(), wmma_bytes.end());
  text.insert(text.end(), valu_bytes.begin(), valu_bytes.end());

  uint64_t sled_start = text.size();
  static constexpr int kSledNops = 16;
  for (int nop_index = 0; nop_index < kSledNops; ++nop_index) {
    uint8_t snop[kMinInstSize];
    EncodeSNop(snop, kTestNopOpcode);
    text.insert(text.end(), snop, snop + kMinInstSize);
  }

  std::vector<InternalDecodedInst> decoded;
  ASSERT_TRUE(DecodeTextSection(text.data(), text.size(), llvm_state, decoded));
  ASSERT_GE(decoded.size(), 2u);

  std::vector<Trampoline> trampolines;
  std::vector<NopSled> nop_sleds{
      {sled_start, sled_start + kSledNops * kMinInstSize, sled_start}};
  RewriteConfig config{"amdgcn-amd-amdhsa--gfx1250",
                       "amdgcn-amd-amdhsa--gfx1250",
                       "gfx1250",
                       kTestBranchGFX12,
                       kTestNopOpcode,
                       256};
  ElfInfo elf_info;
  LivenessInfo liveness;
  llvm::StringMap<KernelPatchStats> kernel_stats;
  std::vector<ScratchPatchInfo> scratch_patches;
  PatchContext ctx{config,         decoded,        text.data(),
                   text.size(),    llvm_state,     trampolines,
                   nop_sleds,      text.data(),    text.size(),
                   elf_info,       liveness,       kernel_stats,
                   scratch_patches};

  const auto &valu_inst = decoded[1];
  uint64_t original_write_pos = nop_sleds[0].write_pos;
  uint32_t patched = ApplyWmmaHazardPatch(ctx);

  EXPECT_EQ(patched, 1u);
  EXPECT_TRUE(trampolines.empty());
  EXPECT_EQ(nop_sleds[0].write_pos,
            original_write_pos + 4 * kMinInstSize + valu_inst.size +
                kMinInstSize);

  for (int nop_index = 0; nop_index < 4; ++nop_index) {
    EXPECT_EQ(std::memcmp(text.data() + original_write_pos +
                              nop_index * kMinInstSize,
                          vnop_bytes.data(), kMinInstSize),
              0);
  }

  EXPECT_EQ(std::memcmp(text.data() + original_write_pos + 4 * kMinInstSize,
                        valu_bytes.data(), valu_bytes.size()),
            0);
  EXPECT_NE(std::memcmp(text.data() + valu_inst.offset, valu_bytes.data(),
                        std::min<size_t>(valu_bytes.size(), kMinInstSize)),
            0);
}
