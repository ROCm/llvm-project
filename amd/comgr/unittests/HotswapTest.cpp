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

static constexpr uint32_t kTestBranchGFX12 = 0xBFA00000u;
static constexpr uint32_t kTestNopOpcode = 0xBF800000u;
static constexpr int kTestMaxVGPRs = 256;

static RewriteConfig MakeTestConfig() {
  return {"amdgcn-amd-amdhsa--gfx1250",
          "amdgcn-amd-amdhsa--gfx1250",
          "gfx1250",
          kTestBranchGFX12,
          kTestNopOpcode,
          kTestMaxVGPRs};
}

static void AppendSNopBytes(std::vector<uint8_t> &text, int nop_count) {
  for (int nop_index = 0; nop_index < nop_count; ++nop_index) {
    uint8_t snop[kMinInstSize];
    EncodeSNop(snop, kTestNopOpcode);
    text.insert(text.end(), snop, snop + kMinInstSize);
  }
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

TEST(CheckVgprOverlap, DetectsReadAndWriteHazards) {
  LLVMState llvm_state = InitLLVMCached("amdgcn-amd-amdhsa--gfx1250");
  ASSERT_TRUE(llvm_state.valid);

  auto wmma_bytes = AssembleSingleInst(
      "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7]",
      llvm_state);
  ASSERT_FALSE(wmma_bytes.empty());

  auto write_hazard_bytes =
      AssembleSingleInst("v_fma_f32 v0, v0, v1, v2", llvm_state);
  ASSERT_FALSE(write_hazard_bytes.empty());

  auto read_hazard_bytes =
      AssembleSingleInst("v_fma_f32 v40, v0, v1, v2", llvm_state);
  ASSERT_FALSE(read_hazard_bytes.empty());

  auto safe_bytes =
      AssembleSingleInst("v_fma_f32 v40, v32, v33, v34", llvm_state);
  ASSERT_FALSE(safe_bytes.empty());

  std::vector<InternalDecodedInst> wmma_decoded;
  ASSERT_TRUE(DecodeTextSection(wmma_bytes.data(), wmma_bytes.size(),
                                llvm_state, wmma_decoded));
  ASSERT_EQ(wmma_decoded.size(), 1u);

  std::vector<InternalDecodedInst> write_hazard_decoded;
  ASSERT_TRUE(DecodeTextSection(write_hazard_bytes.data(),
                                write_hazard_bytes.size(), llvm_state,
                                write_hazard_decoded));
  ASSERT_EQ(write_hazard_decoded.size(), 1u);

  std::vector<InternalDecodedInst> read_hazard_decoded;
  ASSERT_TRUE(DecodeTextSection(read_hazard_bytes.data(),
                                read_hazard_bytes.size(), llvm_state,
                                read_hazard_decoded));
  ASSERT_EQ(read_hazard_decoded.size(), 1u);

  std::vector<InternalDecodedInst> safe_decoded;
  ASSERT_TRUE(DecodeTextSection(safe_bytes.data(), safe_bytes.size(),
                                llvm_state, safe_decoded));
  ASSERT_EQ(safe_decoded.size(), 1u);

  EXPECT_TRUE(CheckVgprOverlap(wmma_decoded[0].inst,
                               write_hazard_decoded[0].inst, *llvm_state.MCII,
                               *llvm_state.MRI));
  EXPECT_TRUE(CheckVgprOverlap(wmma_decoded[0].inst,
                               read_hazard_decoded[0].inst, *llvm_state.MCII,
                               *llvm_state.MRI));
  EXPECT_FALSE(CheckVgprOverlap(wmma_decoded[0].inst, safe_decoded[0].inst,
                                *llvm_state.MCII, *llvm_state.MRI));
}

TEST(ApplyWmmaHazardPatch, InsertsVNopsViaNearbySled) {
  LLVMState llvm_state = InitLLVMCached("amdgcn-amd-amdhsa--gfx1250");
  ASSERT_TRUE(llvm_state.valid);

  auto wmma_bytes = AssembleSingleInst(
      "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7]",
      llvm_state);
  ASSERT_FALSE(wmma_bytes.empty());

  auto valu_bytes = AssembleSingleInst("v_fma_f32 v0, v0, v1, v2", llvm_state);
  ASSERT_FALSE(valu_bytes.empty());

  auto vnop_bytes = AssembleSingleInst("v_nop", llvm_state);
  ASSERT_EQ(vnop_bytes.size(), static_cast<size_t>(kMinInstSize));

  std::vector<uint8_t> text;
  text.insert(text.end(), wmma_bytes.begin(), wmma_bytes.end());
  text.insert(text.end(), valu_bytes.begin(), valu_bytes.end());

  uint64_t sled_start = text.size();
  static constexpr int kSledNops = 16;
  AppendSNopBytes(text, kSledNops);

  std::vector<InternalDecodedInst> decoded;
  ASSERT_TRUE(DecodeTextSection(text.data(), text.size(), llvm_state, decoded));
  ASSERT_GE(decoded.size(), 2u);

  std::vector<Trampoline> trampolines;
  std::vector<NopSled> nop_sleds{
      {sled_start, sled_start + kSledNops * kMinInstSize, sled_start}};
  RewriteConfig config = MakeTestConfig();
  ElfInfo elf_info;
  LivenessInfo liveness;
  llvm::StringMap<KernelPatchStats> kernel_stats;
  std::vector<ScratchPatchInfo> scratch_patches;
  PatchContext ctx{config,         decoded,     text.data(), text.size(),
                   llvm_state,     trampolines, nop_sleds,   text.data(),
                   text.size(),    elf_info,    liveness,    kernel_stats,
                   scratch_patches};

  const auto &valu_inst = decoded[1];
  uint64_t original_write_pos = nop_sleds[0].write_pos;
  uint32_t patched = ApplyWmmaHazardPatch(ctx);

  EXPECT_EQ(patched, 1u);
  EXPECT_TRUE(trampolines.empty());
  EXPECT_EQ(nop_sleds[0].write_pos, original_write_pos + 4 * kMinInstSize +
                                        valu_inst.size + kMinInstSize);

  for (int nop_index = 0; nop_index < 4; ++nop_index) {
    EXPECT_EQ(
        std::memcmp(text.data() + original_write_pos + nop_index * kMinInstSize,
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

TEST(ApplyWmmaHazardPatch, CountsExistingSafeSlotsBeforeHazard) {
  LLVMState llvm_state = InitLLVMCached("amdgcn-amd-amdhsa--gfx1250");
  ASSERT_TRUE(llvm_state.valid);

  auto wmma_bytes = AssembleSingleInst(
      "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7]",
      llvm_state);
  ASSERT_FALSE(wmma_bytes.empty());

  auto salu_bytes = AssembleSingleInst("s_mov_b32 s0, s1", llvm_state);
  ASSERT_FALSE(salu_bytes.empty());

  auto safe_valu_bytes =
      AssembleSingleInst("v_fma_f32 v40, v32, v33, v34", llvm_state);
  ASSERT_FALSE(safe_valu_bytes.empty());

  auto hazard_valu_bytes =
      AssembleSingleInst("v_fma_f32 v0, v0, v1, v2", llvm_state);
  ASSERT_FALSE(hazard_valu_bytes.empty());

  auto vnop_bytes = AssembleSingleInst("v_nop", llvm_state);
  ASSERT_EQ(vnop_bytes.size(), static_cast<size_t>(kMinInstSize));

  std::vector<uint8_t> text;
  text.insert(text.end(), wmma_bytes.begin(), wmma_bytes.end());
  text.insert(text.end(), salu_bytes.begin(), salu_bytes.end());
  text.insert(text.end(), safe_valu_bytes.begin(), safe_valu_bytes.end());
  text.insert(text.end(), vnop_bytes.begin(), vnop_bytes.end());
  text.insert(text.end(), hazard_valu_bytes.begin(), hazard_valu_bytes.end());

  uint64_t sled_start = text.size();
  static constexpr int kSledNops = 16;
  AppendSNopBytes(text, kSledNops);

  std::vector<InternalDecodedInst> decoded;
  ASSERT_TRUE(DecodeTextSection(text.data(), text.size(), llvm_state, decoded));
  ASSERT_GE(decoded.size(), 5u);

  std::vector<Trampoline> trampolines;
  std::vector<NopSled> nop_sleds{
      {sled_start, sled_start + kSledNops * kMinInstSize, sled_start}};
  RewriteConfig config = MakeTestConfig();
  ElfInfo elf_info;
  LivenessInfo liveness;
  llvm::StringMap<KernelPatchStats> kernel_stats;
  std::vector<ScratchPatchInfo> scratch_patches;
  PatchContext ctx{config,         decoded,     text.data(), text.size(),
                   llvm_state,     trampolines, nop_sleds,   text.data(),
                   text.size(),    elf_info,    liveness,    kernel_stats,
                   scratch_patches};

  const auto &hazard_inst = decoded[4];
  uint64_t original_write_pos = nop_sleds[0].write_pos;
  uint32_t patched = ApplyWmmaHazardPatch(ctx);

  EXPECT_EQ(patched, 1u);
  EXPECT_TRUE(trampolines.empty());
  EXPECT_EQ(nop_sleds[0].write_pos, original_write_pos + 2 * kMinInstSize +
                                        hazard_inst.size + kMinInstSize);

  for (int nop_index = 0; nop_index < 2; ++nop_index) {
    EXPECT_EQ(
        std::memcmp(text.data() + original_write_pos + nop_index * kMinInstSize,
                    vnop_bytes.data(), kMinInstSize),
        0);
  }

  EXPECT_EQ(std::memcmp(text.data() + original_write_pos + 2 * kMinInstSize,
                        hazard_valu_bytes.data(), hazard_valu_bytes.size()),
            0);
}

TEST(ApplyWmmaHazardPatch, StopsScanningAtTerminatingSalu) {
  LLVMState llvm_state = InitLLVMCached("amdgcn-amd-amdhsa--gfx1250");
  ASSERT_TRUE(llvm_state.valid);

  auto wmma_bytes = AssembleSingleInst(
      "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7]",
      llvm_state);
  ASSERT_FALSE(wmma_bytes.empty());

  auto endpgm_bytes = AssembleSingleInst("s_endpgm", llvm_state);
  ASSERT_FALSE(endpgm_bytes.empty());

  auto hazard_valu_bytes =
      AssembleSingleInst("v_fma_f32 v0, v0, v1, v2", llvm_state);
  ASSERT_FALSE(hazard_valu_bytes.empty());

  std::vector<uint8_t> text;
  text.insert(text.end(), wmma_bytes.begin(), wmma_bytes.end());
  text.insert(text.end(), endpgm_bytes.begin(), endpgm_bytes.end());
  text.insert(text.end(), hazard_valu_bytes.begin(), hazard_valu_bytes.end());

  uint64_t sled_start = text.size();
  static constexpr int kSledNops = 16;
  AppendSNopBytes(text, kSledNops);

  std::vector<InternalDecodedInst> decoded;
  ASSERT_TRUE(DecodeTextSection(text.data(), text.size(), llvm_state, decoded));
  ASSERT_GE(decoded.size(), 3u);

  std::vector<Trampoline> trampolines;
  std::vector<NopSled> nop_sleds{
      {sled_start, sled_start + kSledNops * kMinInstSize, sled_start}};
  RewriteConfig config = MakeTestConfig();
  ElfInfo elf_info;
  LivenessInfo liveness;
  llvm::StringMap<KernelPatchStats> kernel_stats;
  std::vector<ScratchPatchInfo> scratch_patches;
  PatchContext ctx{config,         decoded,     text.data(), text.size(),
                   llvm_state,     trampolines, nop_sleds,   text.data(),
                   text.size(),    elf_info,    liveness,    kernel_stats,
                   scratch_patches};

  uint64_t original_write_pos = nop_sleds[0].write_pos;
  uint32_t patched = ApplyWmmaHazardPatch(ctx);

  EXPECT_EQ(patched, 0u);
  EXPECT_TRUE(trampolines.empty());
  EXPECT_EQ(nop_sleds[0].write_pos, original_write_pos);
}

TEST(ApplyWmmaHazardPatch, FallsBackToTrampolineWhenNoSledExists) {
  LLVMState llvm_state = InitLLVMCached("amdgcn-amd-amdhsa--gfx1250");
  ASSERT_TRUE(llvm_state.valid);

  auto wmma_bytes = AssembleSingleInst(
      "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7]",
      llvm_state);
  ASSERT_FALSE(wmma_bytes.empty());

  auto hazard_valu_bytes =
      AssembleSingleInst("v_fma_f32 v0, v0, v1, v2", llvm_state);
  ASSERT_FALSE(hazard_valu_bytes.empty());

  auto vnop_bytes = AssembleSingleInst("v_nop", llvm_state);
  ASSERT_EQ(vnop_bytes.size(), static_cast<size_t>(kMinInstSize));

  std::vector<uint8_t> text;
  text.insert(text.end(), wmma_bytes.begin(), wmma_bytes.end());
  text.insert(text.end(), hazard_valu_bytes.begin(), hazard_valu_bytes.end());

  std::vector<InternalDecodedInst> decoded;
  ASSERT_TRUE(DecodeTextSection(text.data(), text.size(), llvm_state, decoded));
  ASSERT_GE(decoded.size(), 2u);

  std::vector<Trampoline> trampolines;
  std::vector<NopSled> nop_sleds;
  RewriteConfig config = MakeTestConfig();
  ElfInfo elf_info;
  LivenessInfo liveness;
  llvm::StringMap<KernelPatchStats> kernel_stats;
  std::vector<ScratchPatchInfo> scratch_patches;
  PatchContext ctx{config,         decoded,     text.data(), text.size(),
                   llvm_state,     trampolines, nop_sleds,   text.data(),
                   text.size(),    elf_info,    liveness,    kernel_stats,
                   scratch_patches};

  const auto &hazard_inst = decoded[1];
  uint32_t patched = ApplyWmmaHazardPatch(ctx);

  ASSERT_EQ(patched, 1u);
  ASSERT_EQ(trampolines.size(), 1u);

  const auto &trampoline = trampolines[0];
  EXPECT_EQ(trampoline.original_offset, hazard_inst.offset);
  EXPECT_EQ(trampoline.original_size, hazard_inst.size);
  EXPECT_EQ(
      trampoline.bytes.size(),
      static_cast<size_t>(4 * kMinInstSize + hazard_inst.size + kMinInstSize));

  for (int nop_index = 0; nop_index < 4; ++nop_index) {
    EXPECT_EQ(std::memcmp(trampoline.bytes.data() + nop_index * kMinInstSize,
                          vnop_bytes.data(), kMinInstSize),
              0);
  }

  EXPECT_EQ(std::memcmp(trampoline.bytes.data() + 4 * kMinInstSize,
                        hazard_valu_bytes.data(), hazard_valu_bytes.size()),
            0);
}
