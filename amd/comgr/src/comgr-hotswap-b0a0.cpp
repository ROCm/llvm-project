//===- comgr-hotswap-b0a0.cpp - GFX1250 B0-to-A0 patch dispatcher --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Dispatcher for B0-to-A0 silicon stepping patches and the
/// RetargetCodeObjectB0A0 orchestrator that drives the full pipeline:
/// decode -> patch -> trampoline growth -> DWARF update.
///
/// Patch entry points are declared as weak symbols returning 0. Each
/// comgr-hotswap-patch-*.cpp file provides a strong override, allowing
/// patches to land as independent PRs with no merge conflicts.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "Utils/AMDGPUBaseInfo.h"

// ── GFX1250 B0-to-A0 constants ──────────────────────────────────────────────

static constexpr uint32_t kSBranchGFX12 = 0xBFA00000u;
static constexpr uint32_t kSNopOpcode = 0xBF800000u;

static RewriteConfig MakeGfx1250B0A0Config(const LLVMState &state) {
  unsigned max_vgprs =
      llvm::AMDGPU::IsaInfo::getTotalNumVGPRs(state.STI.get());
  return {"amdgcn-amd-amdhsa--gfx1250",
          "amdgcn-amd-amdhsa--gfx1250",
          "gfx1250",
          kSBranchGFX12,
          kSNopOpcode,
          max_vgprs};
}

// ── Weak-symbol patch stubs ──────────────────────────────────────────────────
//
// The linker picks strong definitions from patch .cpp files when present;
// otherwise these no-op stubs keep the build green.

__attribute__((weak)) uint32_t ApplyInPlacePatches(PatchContext &, size_t) {
  return 0;
}
__attribute__((weak)) uint32_t ApplyTrampolinePatches(PatchContext &, size_t) {
  return 0;
}
__attribute__((weak)) uint32_t ApplyWmmaHazardPatch(PatchContext &) {
  return 0;
}
__attribute__((weak)) uint32_t ApplyWmmaSplitPatches(PatchContext &, size_t) {
  return 0;
}
__attribute__((weak)) uint32_t ApplyScratchPatches(PatchContext &, size_t) {
  return 0;
}

// ── Weak-symbol liveness stubs ───────────────────────────────────────────────
//
// Conservative defaults: all VGPRs reported live. ScratchAllocator
// will allocate above KD count (correct but suboptimal until
// comgr-hotswap-liveness.cpp lands with real analysis).

__attribute__((weak)) CFG
BuildCFG(const std::vector<InternalDecodedInst> &decoded,
         const llvm::MCInstrInfo &) {
  (void)decoded;
  return CFG{};
}

__attribute__((weak)) LivenessInfo
ComputeLiveness(const std::vector<InternalDecodedInst> &decoded,
                const CFG &, const llvm::MCInstrInfo &,
                const llvm::MCRegisterInfo &,
                unsigned max_vgprs) {
  LivenessInfo info;
  llvm::BitVector all_live(max_vgprs);
  all_live.set(0, max_vgprs);
  info.live_before.resize(decoded.size(), all_live);
  info.live_after.resize(decoded.size(), all_live);
  info.converged = true;
  return info;
}

__attribute__((weak)) RegDefUse
GetInstRegDefUse(const llvm::MCInst &, const llvm::MCInstrInfo &,
                 const llvm::MCRegisterInfo &) {
  return {};
}

__attribute__((weak)) int64_t GetBranchImm(const llvm::MCInst &) { return 0; }

__attribute__((weak)) bool
VerifyPatchCorrectness(const uint8_t *, uint64_t, const LLVMState &,
                       const std::vector<ScratchPatchInfo> &,
                       unsigned) {
  return true;
}

// ── Weak-symbol DWARF stubs ──────────────────────────────────────────────────

__attribute__((weak)) uint8_t *FindSectionHeader(uint8_t *, size_t,
                                                 const char *, int *) {
  return nullptr;
}
__attribute__((weak)) bool
AddTrampolineSymbols(MallocBuffer &, const std::vector<Trampoline> &,
                     uint64_t, int) {
  return true;
}
__attribute__((weak)) bool
PatchDebugLine(MallocBuffer &, const std::vector<Trampoline> &, uint64_t,
               uint64_t) {
  return true;
}
__attribute__((weak)) void PatchDebugRanges(uint8_t *, size_t, uint64_t,
                                            uint64_t, uint64_t) {}
__attribute__((weak)) void PatchDebugInfo(uint8_t *, size_t, uint64_t,
                                          uint64_t, uint64_t) {}
__attribute__((weak)) void PatchDebugFrame(uint8_t *, size_t, uint64_t,
                                           uint64_t, uint64_t) {}

// ── NOP sled scanning ────────────────────────────────────────────────────────

static std::vector<NopSled>
BuildNopSledMap(const std::vector<InternalDecodedInst> &decoded) {
  std::vector<NopSled> sleds;
  size_t i = 0;
  while (i < decoded.size()) {
    if (decoded[i].mnemonic == "s_nop") {
      uint64_t start = decoded[i].offset;
      uint64_t end = start;
      while (i < decoded.size() && decoded[i].mnemonic == "s_nop") {
        end = decoded[i].offset + decoded[i].size;
        ++i;
      }
      if (end - start >= kMinNopSledSize)
        sleds.push_back({start, end, start});
    } else {
      ++i;
    }
  }
  return sleds;
}

// ── Sled-or-trampoline code emission ─────────────────────────────────────────

[[nodiscard]] static bool EmitToNopSled(
    PatchContext &ctx, NopSled &sled, uint64_t inst_offset, uint32_t inst_size,
    const std::vector<uint8_t> &replacement) {
  const auto &cfg = ctx.config;
  std::memcpy(ctx.text + sled.write_pos, replacement.data(),
              replacement.size());

  uint8_t br_back[kMinInstSize];
  if (!EncodeSBranch(sled.write_pos + replacement.size(),
                     inst_offset + inst_size, br_back, cfg.s_branch_opcode))
    return false;
  std::memcpy(ctx.text + sled.write_pos + replacement.size(), br_back,
              kMinInstSize);

  uint8_t br_fwd[kMinInstSize];
  if (!EncodeSBranch(inst_offset, sled.write_pos, br_fwd,
                     cfg.s_branch_opcode))
    return false;
  std::memcpy(ctx.text + inst_offset, br_fwd, kMinInstSize);

  for (uint32_t i = kMinInstSize; i < inst_size; i += kMinInstSize) {
    uint8_t nop[kMinInstSize];
    EncodeSNop(nop, cfg.s_nop_opcode);
    std::memcpy(ctx.text + inst_offset + i, nop, kMinInstSize);
  }
  sled.write_pos += replacement.size() + kMinInstSize;
  return true;
}

[[nodiscard]] static bool EmitToTrampoline(
    PatchContext &ctx, uint64_t inst_offset, uint32_t inst_size,
    const std::vector<uint8_t> &replacement) {
  uint64_t tramp_offset = ctx.text_size;
  for (auto &t : ctx.out_trampolines)
    tramp_offset += t.bytes.size();

  Trampoline t;
  t.original_offset = inst_offset;
  t.original_size = inst_size;
  t.bytes.insert(t.bytes.end(), replacement.begin(), replacement.end());

  uint8_t br_back[kMinInstSize];
  if (!EncodeSBranch(tramp_offset + t.bytes.size(), inst_offset + inst_size,
                     br_back, ctx.config.s_branch_opcode))
    return false;
  t.bytes.insert(t.bytes.end(), br_back, br_back + kMinInstSize);

  ctx.out_trampolines.push_back(std::move(t));
  return true;
}

[[nodiscard]] static bool EmitReplacementCode(
    PatchContext &ctx, uint64_t inst_offset, uint32_t inst_size,
    const std::vector<uint8_t> &replacement, const char *desc = nullptr) {
  uint64_t needed = replacement.size() + kMinInstSize;
  NopSled *sled = FindNearestSled(ctx.nop_sleds, inst_offset, needed);

  if (sled && replacement.size() + kMinInstSize <= sled->end - sled->write_pos)
    return EmitToNopSled(ctx, *sled, inst_offset, inst_size, replacement);

  return EmitToTrampoline(ctx, inst_offset, inst_size, replacement);
}

// ── ApplyGfx1250B0toA0Rules ──────────────────────────────────────────────────

static uint32_t
ApplyGfx1250B0toA0Rules(std::vector<InternalDecodedInst> &decoded,
                        uint8_t *text, uint64_t text_size,
                        const LLVMState &llvm_state,
                        std::vector<Trampoline> &out_trampolines,
                        uint8_t *elf_data, size_t elf_size,
                        const ElfInfo &elf_info,
                        std::vector<ScratchPatchInfo> &out_scratch_patches,
                        const RewriteConfig &config) {
  uint32_t patched = 0;
  std::vector<NopSled> nop_sleds = BuildNopSledMap(decoded);

  CFG cfg = BuildCFG(decoded, *llvm_state.MCII);
  LivenessInfo liveness = ComputeLiveness(
      decoded, cfg, *llvm_state.MCII, *llvm_state.MRI, config.max_vgprs);

  if (!liveness.converged) {
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WARNING: liveness analysis did not converge, "
           "using conservative all-VGPRs-live fallback\n";
    llvm::BitVector all_vgprs(config.max_vgprs);
    all_vgprs.set(0, config.max_vgprs);
    for (size_t i = 0; i < liveness.live_before.size(); ++i) {
      liveness.live_before[i] = all_vgprs;
      liveness.live_after[i] = all_vgprs;
    }
  }

  llvm::StringMap<KernelPatchStats> kernel_stats;

  PatchContext ctx{config,       decoded,     text,
                   text_size,    llvm_state,
                   out_trampolines, nop_sleds,
                   elf_data,     elf_size,
                   elf_info,     liveness,
                   kernel_stats, out_scratch_patches};

  for (size_t idx = 0; idx < decoded.size(); ++idx) {
    auto &di = decoded[idx];
    if (di.mnemonic == "<unknown>")
      continue;

    uint32_t p = 0;
    p += ApplyInPlacePatches(ctx, idx);
    if (p) { patched += p; continue; }
    p += ApplyTrampolinePatches(ctx, idx);
    if (p) { patched += p; continue; }
    p += ApplyWmmaSplitPatches(ctx, idx);
    if (p) { patched += p; continue; }
    p += ApplyScratchPatches(ctx, idx);
    if (p) { patched += p; continue; }
  }

  patched += ApplyWmmaHazardPatch(ctx);

  for (const auto &kv : kernel_stats) {
    llvm::StringRef kname = kv.first();
    const auto &stats = kv.second;
    if (kname.empty())
      continue;
    std::string kname_str = kname.str();
    int vgprs_before =
        GetKernelVgprCount(elf_data, elf_size, elf_info, kname_str);
    if (stats.extra_vgprs > 0)
      UpdateKernelDescriptor(elf_data, elf_size, elf_info, kname_str,
                             stats.extra_vgprs, 0);
    int vgprs_after =
        GetKernelVgprCount(elf_data, elf_size, elf_info, kname_str);
    HotswapLog(HotswapLogLevel::Info)
        << "hotswap: liveness: kernel " << kname
        << ": vgprs_before=" << vgprs_before
        << ", vgprs_after=" << vgprs_after
        << ", scratch_reused=" << stats.scratch_reused
        << ", scratch_above_kd=" << stats.scratch_above_kd << "\n";
  }

  return patched;
}

// ── RetargetCodeObjectB0A0 — helpers ─────────────────────────────────────────

static void FixupTrampolineBranches(
    std::vector<Trampoline> &trampolines, uint8_t *text,
    uint64_t text_size, const RewriteConfig &config) {
  uint64_t tramp_text_offset = text_size;
  for (auto &t : trampolines) {
    uint64_t tp = tramp_text_offset;
    tramp_text_offset += t.bytes.size();

    uint8_t br_back[kMinInstSize];
    if (!EncodeSBranch(tp + t.bytes.size() - kMinInstSize,
                       t.original_offset + t.original_size, br_back,
                       config.s_branch_opcode))
      continue;
    std::memcpy(t.bytes.data() + t.bytes.size() - kMinInstSize, br_back,
                kMinInstSize);

    uint8_t br_fwd[kMinInstSize];
    if (!EncodeSBranch(t.original_offset, tp, br_fwd,
                       config.s_branch_opcode))
      continue;
    std::memcpy(text + t.original_offset, br_fwd, kMinInstSize);
    for (uint32_t i = kMinInstSize; i < t.original_size; i += kMinInstSize) {
      uint8_t nop[kMinInstSize];
      EncodeSNop(nop, config.s_nop_opcode);
      std::memcpy(text + t.original_offset + i, nop, kMinInstSize);
    }
  }
}

static void PatchDebugSections(MallocBuffer &elf_buf,
                               const std::vector<Trampoline> &trampolines,
                               const ElfInfo &elf_info,
                               size_t tramp_total) {
  if (!AddTrampolineSymbols(elf_buf, trampolines, elf_info.text_size,
                            elf_info.text_section_idx))
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WARNING: AddTrampolineSymbols failed\n";
  PatchDebugRanges(elf_buf.get(), elf_buf.size, elf_info.text_addr,
                   elf_info.text_size, tramp_total);
  PatchDebugInfo(elf_buf.get(), elf_buf.size, elf_info.text_addr,
                 elf_info.text_size, tramp_total);
  PatchDebugFrame(elf_buf.get(), elf_buf.size, elf_info.text_addr,
                  elf_info.text_size, tramp_total);
  if (!PatchDebugLine(elf_buf, trampolines, elf_info.text_size,
                      elf_info.text_addr))
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WARNING: PatchDebugLine failed\n";
}

static void RunScratchVerification(
    const void *out_data, size_t out_size, const LLVMState &llvm_state,
    const std::vector<ScratchPatchInfo> &scratch_patches,
    unsigned max_vgprs) {
  ElfInfo verify_elf_info;
  const uint8_t *verify_elf = static_cast<const uint8_t *>(out_data);
  if (!ParseElfInfo(verify_elf, out_size, verify_elf_info) ||
      verify_elf_info.text_size == 0)
    return;
  if (!VerifyPatchCorrectness(verify_elf + verify_elf_info.text_offset,
                              verify_elf_info.text_size, llvm_state,
                              scratch_patches, max_vgprs))
    HotswapLog(HotswapLogLevel::Error)
        << "hotswap: WARNING: post-patch verification detected "
           "possible scratch conflicts\n";
}

// ── RetargetCodeObjectB0A0 ───────────────────────────────────────────────────

amd_comgr_status_t RetargetCodeObjectB0A0(const void *elf_data,
                                          size_t elf_size, void **out_data,
                                          size_t *out_size) {
  ElfInfo elf_info;
  const uint8_t *elf = static_cast<const uint8_t *>(elf_data);
  if (!ParseElfInfo(elf, elf_size, elf_info) || elf_info.text_size == 0) {
    MallocBuffer copy(elf_size);
    if (!copy) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(copy.get(), elf_data, elf_size);
    *out_size = elf_size;
    *out_data = copy.release();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  const std::string isa = "amdgcn-amd-amdhsa--gfx1250";
  LLVMState llvm_state = InitLLVMCached(isa);
  if (!llvm_state.valid) return AMD_COMGR_STATUS_ERROR;

  RewriteConfig config = MakeGfx1250B0A0Config(llvm_state);

  std::vector<uint8_t> buf(elf, elf + elf_size);
  uint8_t *text = buf.data() + elf_info.text_offset;

  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, llvm_state, decoded))
    return AMD_COMGR_STATUS_ERROR;

  std::vector<Trampoline> deferred;
  std::vector<ScratchPatchInfo> scratch_patches;
  uint32_t count =
      ApplyGfx1250B0toA0Rules(decoded, text, elf_info.text_size, llvm_state,
                              deferred, buf.data(), buf.size(), elf_info,
                              scratch_patches, config);

  HotswapLog(HotswapLogLevel::Info)
      << "hotswap: applied " << count << " patches\n";

  if (!deferred.empty()) {
    FixupTrampolineBranches(deferred, text, elf_info.text_size, config);

    MallocBuffer new_buf =
        GrowElfWithTrampolines(buf.data(), elf_size, elf_info, deferred);
    if (!new_buf) return AMD_COMGR_STATUS_ERROR;

    size_t tramp_total = 0;
    for (auto &t : deferred) tramp_total += t.bytes.size();

    PatchDebugSections(new_buf, deferred, elf_info, tramp_total);

    *out_size = new_buf.size;
    *out_data = new_buf.release();
  } else {
    MallocBuffer out(elf_size);
    if (!out) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(out.get(), buf.data(), elf_size);
    *out_data = out.release();
    *out_size = elf_size;
  }

  if (!scratch_patches.empty())
    RunScratchVerification(*out_data, *out_size, llvm_state, scratch_patches,
                           config.max_vgprs);

  return AMD_COMGR_STATUS_SUCCESS;
}
