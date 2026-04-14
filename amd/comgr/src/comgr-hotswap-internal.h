//===- comgr-hotswap-internal.h - HotSwap internal types and declarations -===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal header for the HotSwap ISA rewriting subsystem. Shared by all
/// comgr-hotswap-*.cpp compilation units. Not part of the public COMGR API.
///
/// Module structure:
///   comgr-hotswap-elf.cpp       — ELF parsing, binary helpers, trampoline growth
///   comgr-hotswap-dwarf.cpp     — DWARF debug section patching
///   comgr-hotswap-llvm.cpp      — LLVM MC infrastructure (disasm/asm/encode)
///   comgr-hotswap-liveness.cpp  — CFG, backward liveness, scratch allocator
///   comgr-hotswap-b0a0.cpp      — ISA rewrite policy (e.g., GFX1250 B0-to-A0)
///   comgr-hotswap.cpp           — Public C API entry points
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_INTERNAL_H
#define COMGR_HOTSWAP_INTERNAL_H

#include "amd_comgr.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/BitVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/raw_ostream.h"

// ── MallocBuffer ─────────────────────────────────────────────────────────────

struct FreeDeleter {
  void operator()(uint8_t *p) const { std::free(p); }
};

struct MallocBuffer {
  std::unique_ptr<uint8_t[], FreeDeleter> data;
  size_t size = 0;

  MallocBuffer() = default;
  explicit MallocBuffer(size_t n)
      : data(static_cast<uint8_t *>(std::malloc(n))), size(data ? n : 0) {}

  MallocBuffer(MallocBuffer &&o) noexcept
      : data(std::move(o.data)), size(std::exchange(o.size, 0)) {}
  MallocBuffer &operator=(MallocBuffer &&o) noexcept {
    data = std::move(o.data);
    size = std::exchange(o.size, 0);
    return *this;
  }

  explicit operator bool() const { return data != nullptr; }
  uint8_t *get() const { return data.get(); }
  uint8_t *release() { size = 0; return data.release(); }
};

// ── Logging ──────────────────────────────────────────────────────────────────

enum class HotswapLogLevel : int { Silent = 0, Error = 1, Info = 2, Debug = 3 };

inline HotswapLogLevel GetHotswapLogLevel() {
  static HotswapLogLevel level = []() {
    const char *env = std::getenv("HSA_HOTSWAP_LOG_LEVEL");
    if (env) {
      int v = std::atoi(env);
      if (v >= 0 && v <= 3)
        return static_cast<HotswapLogLevel>(v);
    }
    return HotswapLogLevel::Info;
  }();
  return level;
}

inline llvm::raw_ostream &HotswapLog(HotswapLogLevel level) {
  if (static_cast<int>(level) <= static_cast<int>(GetHotswapLogLevel()))
    return llvm::errs();
  return llvm::nulls();
}

// ── RewriteConfig ────────────────────────────────────────────────────────────
//
// ISA-specific parameters that drive the generic rewriting infrastructure.
// Constructed by the policy layer (e.g., b0a0.cpp for GFX1250) and threaded
// through PatchContext so infrastructure code has zero ISA assumptions.

struct RewriteConfig {
  std::string source_isa;
  std::string target_isa;
  std::string target_cpu;
  uint32_t s_branch_opcode;
  uint32_t s_nop_opcode;
  unsigned max_vgprs;
};

// ── ELF types ────────────────────────────────────────────────────────────────

struct ElfSection {
  uint32_t name_idx;
  std::string name;
  uint32_t type;
  uint64_t offset;
  uint64_t size;
  uint64_t addr;
};

struct ElfSymbol {
  std::string name;
  uint64_t value;
  uint64_t size;
  uint8_t info;
  uint16_t shndx;
};

struct ElfInfo {
  std::vector<ElfSection> sections;
  std::vector<ElfSymbol> symbols;
  int text_section_idx = -1;
  int text_idx = -1;
  uint64_t text_offset = 0;
  uint64_t text_size = 0;
  uint64_t text_addr = 0;
};

// ── Trampoline ───────────────────────────────────────────────────────────────

struct Trampoline {
  uint64_t original_offset;
  uint32_t original_size;
  std::vector<uint8_t> bytes;
};

// ── NOP sled ─────────────────────────────────────────────────────────────────

struct NopSled {
  uint64_t start;
  uint64_t end;
  uint64_t write_pos;
};

// ── Rewrite-rule types ───────────────────────────────────────────────────────

struct RewriteRule {
  std::string replace_mnemonic;
  std::vector<uint8_t> replace_bytes;
};

// ── ELF / KD named constants ─────────────────────────────────────────────────

static constexpr uint64_t kMinElfSize = sizeof(llvm::ELF::Elf64_Ehdr);
static constexpr uint64_t kKdSize = 64;
static constexpr uint64_t kKdRsrc1Offset = 48;
static constexpr uint32_t kKdRsrc1VgprMask = 0x3Fu;
static constexpr uint32_t kKdRsrc1VgprMaxGranule = 63;
static constexpr uint32_t kKdRsrc1SgprShift = 6;
static constexpr uint32_t kKdRsrc1SgprMask = 0xFu;
static constexpr uint32_t kKdRsrc1SgprMaxGranule = 15;
static constexpr uint32_t kVgprGranularity = 8;
static constexpr uint32_t kVgprGranuleSize = 4;
static constexpr uint32_t kSgprGranuleSize = 8;
static constexpr int64_t kMaxSledDistance = 131072;
static constexpr uint64_t kMinNopSledSize = 8;
static constexpr uint32_t kMinInstSize = 4;

// AMDGPU code-object-v3 ISA version note type (not in llvm::ELF enum)
static constexpr uint32_t kNoteTypeIsaVersion = 27;

// ELF symbol type extraction mask
static constexpr uint8_t kElfStTypeMask = 0xf;

// ── DWARF types ──────────────────────────────────────────────────────────────

struct DebugLineRow {
  uint64_t address;
  uint32_t file;
  int32_t line;
};

// ── LLVM MC Context ──────────────────────────────────────────────────────────

struct LLVMState {
  const llvm::Target *target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCObjectFileInfo> MOFI;
  std::unique_ptr<llvm::MCDisassembler> disasm;
  std::unique_ptr<llvm::MCInstPrinter> printer;
  std::unique_ptr<llvm::MCCodeEmitter> CE;
  std::string cpu;
  bool valid = false;
};

// ── Decoded instruction ──────────────────────────────────────────────────────

struct InternalDecodedInst {
  uint64_t offset;
  uint32_t size;
  llvm::MCInst inst;
  std::string mnemonic;
};

// ── VGPR liveness types ──────────────────────────────────────────────────────

struct RegDefUse {
  llvm::BitVector defs;
  llvm::BitVector uses;
};

struct BasicBlock {
  uint64_t start_offset = 0;
  uint64_t end_offset = 0;
  std::vector<size_t> inst_indices;
  std::vector<int> successors;
  std::vector<int> predecessors;
};

struct CFG {
  std::vector<BasicBlock> blocks;
  std::unordered_map<uint64_t, int> offset_to_block;
};

struct LivenessInfo {
  std::vector<llvm::BitVector> live_before;
  std::vector<llvm::BitVector> live_after;
  bool converged = false;
};

struct ScratchAllocator {
  llvm::BitVector live_at_point;
  int kd_allocated_vgprs;
  int next_above_kd;
  int max_vgprs;
  int extra_allocated = 0;

  ScratchAllocator(const llvm::BitVector &live, int kd_vgprs, int max)
      : live_at_point(live), kd_allocated_vgprs(kd_vgprs),
        next_above_kd(kd_vgprs), max_vgprs(max) {}

  int Alloc() {
    for (int v = kd_allocated_vgprs - 1; v >= 0; --v) {
      if (!live_at_point.test(v)) {
        live_at_point.set(v);
        return v;
      }
    }
    if (next_above_kd >= max_vgprs)
      return -1;
    int v = next_above_kd++;
    extra_allocated++;
    live_at_point.set(v);
    return v;
  }

  int ExtraVgprsNeeded() const { return extra_allocated; }
};

struct ScratchPatchInfo {
  uint64_t offset;
  llvm::BitVector scratch_regs;
};

// ── Patch types ──────────────────────────────────────────────────────────────

struct WmmaNopReq {
  int b0_nops;
  int a0_nops;
};

struct WmmaHazard {
  size_t wmma_idx;
  size_t valu_idx;
  int existing_nops;
  int needed_nops;
  int deficit;
};

struct KernelPatchStats {
  int extra_vgprs = 0;
  int scratch_reused = 0;
  int scratch_above_kd = 0;
};

struct PatchContext {
  const RewriteConfig &config;
  std::vector<InternalDecodedInst> &decoded;
  uint8_t *text;
  uint64_t text_size;
  const LLVMState &llvm_state;
  std::vector<Trampoline> &out_trampolines;
  std::vector<NopSled> &nop_sleds;
  uint8_t *elf_data;
  size_t elf_size;
  const ElfInfo &elf_info;
  const LivenessInfo &liveness;
  std::unordered_map<std::string, KernelPatchStats> &kernel_stats;
  std::vector<ScratchPatchInfo> &out_scratch_patches;
};

// ── Function declarations ────────────────────────────────────────────────────

// elf
[[nodiscard]] bool EncodeSBranch(uint64_t from_offset, uint64_t to_offset,
                                 uint8_t out_bytes[4],
                                 uint32_t s_branch_opcode);
void EncodeSNop(uint8_t out_bytes[4], uint32_t s_nop_opcode);
std::string ExtractCPU(const std::string &isa_name);
[[nodiscard]] bool ParseElfInfo(const uint8_t *elf, size_t elf_size,
                                ElfInfo &info);
std::string FindKernelAtOffset(const ElfInfo &elf_info, uint64_t text_offset);
[[nodiscard]] bool ApplyByteReplace(const RewriteRule &rule,
                                    uint64_t inst_offset, uint32_t inst_size,
                                    uint8_t *text, uint64_t text_size,
                                    uint32_t s_nop_opcode);
void UpdateKernelDescriptor(uint8_t *elf_data, size_t elf_size,
                            const ElfInfo &elf_info,
                            const std::string &kernel_name,
                            int32_t extra_vgprs, int32_t extra_sgprs);
NopSled *FindNearestSled(std::vector<NopSled> &sleds, uint64_t offset,
                         uint64_t needed);
MallocBuffer GrowElfWithTrampolines(const uint8_t *elf, size_t elf_size,
                                    const ElfInfo &elf_info,
                                    const std::vector<Trampoline> &trampolines);
bool PatchElfIsa(uint8_t *elf, size_t elf_size, const std::string &target_cpu);
int GetKernelVgprCount(const uint8_t *elf_data, size_t elf_size,
                       const ElfInfo &elf_info,
                       const std::string &kernel_name);

// dwarf
uint8_t *FindSectionHeader(uint8_t *elf, size_t elf_size, const char *name,
                           int *out_idx = nullptr);
[[nodiscard]] bool AddTrampolineSymbols(
    MallocBuffer &elf_buf, const std::vector<Trampoline> &trampolines,
    uint64_t text_size_before, int text_section_idx);
[[nodiscard]] bool PatchDebugLine(MallocBuffer &elf_buf,
                                  const std::vector<Trampoline> &trampolines,
                                  uint64_t text_size_before,
                                  uint64_t text_addr);
void PatchDebugRanges(uint8_t *elf, size_t elf_size, uint64_t text_addr,
                      uint64_t text_size_before, uint64_t tramp_total);
void PatchDebugInfo(uint8_t *elf, size_t elf_size, uint64_t text_addr,
                    uint64_t text_size_before, uint64_t tramp_total);
void PatchDebugFrame(uint8_t *elf, size_t elf_size, uint64_t text_addr,
                     uint64_t text_size_before, uint64_t tramp_total);

// llvm
LLVMState InitLLVMImpl(const std::string &isa_name,
                       const llvm::Target *cached_target = nullptr);
LLVMState InitLLVMCached(const std::string &isa_name);
[[nodiscard]] bool DecodeTextSection(const uint8_t *text, uint64_t text_size,
                                     const LLVMState &llvm_state,
                                     std::vector<InternalDecodedInst> &decoded);
std::vector<uint8_t> AssembleSingleInst(const std::string &asm_str,
                                        const LLVMState &llvm_state);
[[nodiscard]] bool ApplyMnemonicSwap(const RewriteRule &rule,
                                     InternalDecodedInst &inst, uint8_t *text,
                                     const LLVMState &llvm_state);
Trampoline BuildTrampoline(const std::vector<std::string> &asm_lines,
                           uint64_t original_offset, uint32_t original_size,
                           uint64_t trampoline_text_offset,
                           const RewriteConfig &config,
                           const LLVMState &llvm_state);
std::string PrintInst(const InternalDecodedInst &di,
                      const LLVMState &llvm_state);
int GetVgprNum(unsigned reg, const llvm::MCRegisterInfo &MRI);
std::pair<int, int> GetVgprRange(unsigned reg,
                                 const llvm::MCRegisterInfo &MRI);
std::pair<int, int> GetOperandVgprRange(const llvm::MCInst &inst,
                                        unsigned op_idx,
                                        const llvm::MCRegisterInfo &MRI);
bool RangesOverlap(int base1, int count1, int base2, int count2);
bool CheckVgprOverlap(const llvm::MCInst &wmma_inst,
                      const llvm::MCInst &valu_inst,
                      const llvm::MCRegisterInfo &MRI);

// liveness
RegDefUse GetInstRegDefUse(const llvm::MCInst &inst,
                           const llvm::MCInstrInfo &MCII,
                           const llvm::MCRegisterInfo &MRI);
int64_t GetBranchImm(const llvm::MCInst &inst);
CFG BuildCFG(const std::vector<InternalDecodedInst> &decoded,
             const llvm::MCInstrInfo &MCII);
LivenessInfo ComputeLiveness(const std::vector<InternalDecodedInst> &decoded,
                             const CFG &cfg, const llvm::MCInstrInfo &MCII,
                             const llvm::MCRegisterInfo &MRI,
                             unsigned max_vgprs);
[[nodiscard]] bool VerifyPatchCorrectness(
    const uint8_t *text, uint64_t text_size, const LLVMState &llvm_state,
    const std::vector<ScratchPatchInfo> &scratch_patches,
    unsigned max_vgprs);

// policy — dispatcher
amd_comgr_status_t RetargetCodeObjectB0A0(const void *elf_data,
                                          size_t elf_size, void **out_data,
                                          size_t *out_size);

// policy — patch entry points (weak stubs in b0a0.cpp)
//
// Each group is defined as a weak symbol in comgr-hotswap-b0a0.cpp returning 0.
// Patch .cpp files provide strong definitions that override the stubs at link
// time, allowing patches to land as independent PRs.

/// Per-instruction patches that rewrite in place without changing code size:
///  - cluster_load -> global_load mnemonic swap
///  - s_clause -> s_nop byte overwrite
uint32_t ApplyInPlacePatches(PatchContext &ctx, size_t idx);

/// Per-instruction patches that expand one instruction into multiple via NOP
/// sled or trampoline: ds_2addr stride64 expansion and tensor_load_to_lds
/// s_pack_hh prepend.
uint32_t ApplyTrampolinePatches(PatchContext &ctx, size_t idx);

/// Whole-kernel pass: detect WMMA/SWMMAC followed by VALU with overlapping
/// VGPR ranges and insert v_nop instructions to resolve the co-execution
/// hazard. Runs after all per-instruction patches.
uint32_t ApplyWmmaHazardPatch(PatchContext &ctx);

/// Per-instruction decomposition of unsupported WMMA variants into narrower
/// operations: split 16x16x128 FP8/BF8 WMMA into two 16x16x64 halves and
/// split 32x16x128_f4 WMMA into two 16x16x128 f8f6f4 WMMAs with FP4 format
/// modifiers.
uint32_t ApplyWmmaSplitPatches(PatchContext &ctx, size_t idx);

/// Per-instruction patches requiring dynamically allocated VGPRs via
/// ScratchAllocator backed by backward liveness analysis: CVT E5M3 CLAMP
/// emulation (4 scratch VGPRs) and Scale16 decomposition (2 scratch VGPRs).
uint32_t ApplyScratchPatches(PatchContext &ctx, size_t idx);

#endif // COMGR_HOTSWAP_INTERNAL_H
