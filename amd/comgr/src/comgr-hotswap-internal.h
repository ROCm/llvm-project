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
///   comgr-hotswap-llvm.cpp      — LLVM MC infrastructure (disasm/asm/encode)
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
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
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
  llvm::SmallVector<uint8_t, 16> bytes;
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
  llvm::SmallVector<uint8_t, 16> replace_bytes;
};

// ── Named constants ──────────────────────────────────────────────────────────

// ELF
static constexpr uint64_t kMinElfSize = sizeof(llvm::ELF::Elf64_Ehdr);

// Kernel descriptor — sizes and offsets from AMDHSAKernelDescriptor.h
static constexpr uint64_t kKdSize = sizeof(llvm::amdhsa::kernel_descriptor_t);
static constexpr uint64_t kKdRsrc1Offset =
    llvm::amdhsa::COMPUTE_PGM_RSRC1_OFFSET;

// VGPR/SGPR granularity for KD RSRC1 fields
static constexpr uint32_t kVgprGranularity = 8;
static constexpr uint32_t kVgprGranuleSize = 4;
static constexpr uint32_t kSgprGranuleSize = 8;

// Infrastructure limits
static constexpr int64_t kMaxSledDistance = 131072;
static constexpr uint64_t kMinNopSledSize = 8;
static constexpr uint32_t kMinInstSize = 4;

// AMDGPU ELF note — type 27 is the ISA version note used by code object v3+
// (including v5). The numeric value is not in llvm::ELF but is stable across
// code object versions.
static constexpr uint32_t kNoteTypeIsaVersion = 27;

// ELF symbol type extraction mask
static constexpr uint8_t kElfStTypeMask = 0xf;

// s_branch encoding limits (16-bit signed dword offset field)
static constexpr int64_t kBranchOffsetMin = -32768;
static constexpr int64_t kBranchOffsetMax = 32767;
static constexpr uint32_t kBranchOffsetMask = 0xFFFF;

// AMDGPU ELF note owner
static constexpr const char *kAmdgpuNoteOwner = "AMDGPU";
static constexpr size_t kAmdgpuNoteOwnerLen = 6;

// ELF note alignment
static constexpr uint32_t kNoteAlign = 4;

// ── Function declarations (ELF layer) ────────────────────────────────────────

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

#endif // COMGR_HOTSWAP_INTERNAL_H
