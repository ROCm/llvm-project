//===- comgr-hotswap-elf.cpp - ELF types, parsing, and binary helpers -----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/TargetParser.h"

// ── s_branch / s_nop encoding ────────────────────────────────────────────────

[[nodiscard]] bool EncodeSBranch(uint64_t from_offset, uint64_t to_offset,
                                 uint8_t out_bytes[4],
                                 uint32_t s_branch_opcode) {
  int64_t byte_delta = static_cast<int64_t>(to_offset) -
                       static_cast<int64_t>(from_offset) - kMinInstSize;
  if (byte_delta % kMinInstSize != 0) return false;
  int64_t dword_offset = byte_delta / kMinInstSize;
  if (dword_offset < kBranchOffsetMin || dword_offset > kBranchOffsetMax)
    return false;
  uint32_t encoded = s_branch_opcode |
                     (static_cast<uint16_t>(dword_offset) & kBranchOffsetMask);
  std::memcpy(out_bytes, &encoded, kMinInstSize);
  return true;
}

void EncodeSNop(uint8_t out_bytes[4], uint32_t s_nop_opcode) {
  std::memcpy(out_bytes, &s_nop_opcode, kMinInstSize);
}

// ── ExtractCPU ───────────────────────────────────────────────────────────────

std::string ExtractCPU(const std::string &isa_name) {
  size_t pos = isa_name.rfind("gfx");
  if (pos != std::string::npos) {
    std::string cpu;
    for (size_t i = pos; i < isa_name.size(); ++i) {
      if (llvm::isAlnum(isa_name[i]))
        cpu += isa_name[i];
      else
        break;
    }
    return cpu;
  }
  return "";
}

// ── ELF parsing ──────────────────────────────────────────────────────────────

[[nodiscard]] bool ParseElfInfo(const uint8_t *elf, size_t elf_size,
                                ElfInfo &info) {
  using ELFT = llvm::object::ELF64LE;
  auto elf_or_err = llvm::object::ELFFile<ELFT>::create(
      llvm::StringRef(reinterpret_cast<const char *>(elf), elf_size));
  if (!elf_or_err) {
    llvm::consumeError(elf_or_err.takeError());
    return false;
  }
  const auto &elf_file = *elf_or_err;

  auto sections_or_err = elf_file.sections();
  if (!sections_or_err) {
    llvm::consumeError(sections_or_err.takeError());
    return false;
  }
  auto shdrs = *sections_or_err;

  for (const auto &shdr : shdrs) {
    ElfSection sec;
    sec.type = shdr.sh_type;
    sec.offset = shdr.sh_offset;
    sec.size = shdr.sh_size;
    sec.addr = shdr.sh_addr;
    sec.name_idx = shdr.sh_name;

    auto name_or_err = elf_file.getSectionName(shdr);
    if (name_or_err)
      sec.name = name_or_err->str();
    else
      llvm::consumeError(name_or_err.takeError());

    if (sec.name == ".text" && sec.offset + sec.size <= elf_size) {
      info.text_section_idx = static_cast<int>(info.sections.size());
      info.text_idx = info.text_section_idx;
      info.text_offset = sec.offset;
      info.text_size = sec.size;
      info.text_addr = sec.addr;
    }

    info.sections.push_back(std::move(sec));
  }

  size_t num_sections = info.sections.size();
  for (size_t i = 0; i < num_sections; ++i) {
    if (info.sections[i].type != llvm::ELF::SHT_SYMTAB &&
        info.sections[i].type != llvm::ELF::SHT_DYNSYM)
      continue;

    const auto &sym_shdr = *(shdrs.begin() + i);

    auto syms_or_err = elf_file.symbols(&sym_shdr);
    if (!syms_or_err) {
      llvm::consumeError(syms_or_err.takeError());
      continue;
    }

    auto strtab_or_err = elf_file.getStringTableForSymtab(sym_shdr, shdrs);
    if (!strtab_or_err) {
      llvm::consumeError(strtab_or_err.takeError());
      continue;
    }

    for (const auto &sym : *syms_or_err) {
      ElfSymbol esym;
      esym.info = sym.st_info;
      esym.shndx = sym.st_shndx;
      esym.value = sym.st_value;
      esym.size = sym.st_size;

      auto sym_name_or_err = sym.getName(*strtab_or_err);
      if (sym_name_or_err)
        esym.name = sym_name_or_err->str();
      else
        llvm::consumeError(sym_name_or_err.takeError());

      info.symbols.push_back(std::move(esym));
    }
  }

  return info.text_section_idx >= 0;
}

std::string FindKernelAtOffset(const ElfInfo &elf_info,
                               uint64_t text_offset) {
  for (auto &sym : elf_info.symbols) {
    uint8_t sym_type = sym.info & kElfStTypeMask;
    if (sym_type != llvm::ELF::STT_FUNC && sym_type != llvm::ELF::STT_GNU_IFUNC)
      continue;
    if (sym.shndx != static_cast<uint16_t>(elf_info.text_section_idx))
      continue;
    uint64_t sym_start = sym.value;
    uint64_t sym_end = sym.value + sym.size;
    if (text_offset >= sym_start && text_offset < sym_end)
      return sym.name;
  }
  return "";
}

// ── ApplyByteReplace ─────────────────────────────────────────────────────────

[[nodiscard]] bool ApplyByteReplace(const RewriteRule &rule,
                                    uint64_t inst_offset, uint32_t inst_size,
                                    uint8_t *text, uint64_t text_size,
                                    uint32_t s_nop_opcode) {
  if (inst_offset + inst_size > text_size) return false;
  if (rule.replace_bytes.size() > inst_size) return false;
  std::memcpy(text + inst_offset, rule.replace_bytes.data(),
              rule.replace_bytes.size());
  uint32_t remaining =
      inst_size - static_cast<uint32_t>(rule.replace_bytes.size());
  uint64_t pad_offset = inst_offset + rule.replace_bytes.size();
  while (remaining >= kMinInstSize) {
    uint8_t nop[kMinInstSize];
    EncodeSNop(nop, s_nop_opcode);
    std::memcpy(text + pad_offset, nop, kMinInstSize);
    pad_offset += kMinInstSize;
    remaining -= kMinInstSize;
  }
  return true;
}

// ── Kernel descriptor lookup ─────────────────────────────────────────────────

static uint8_t *FindKernelDescriptor(uint8_t *elf_data, size_t elf_size,
                                     const ElfInfo &elf_info,
                                     const std::string &kernel_name) {
  std::string kd_name = kernel_name + ".kd";
  for (const auto &sym : elf_info.symbols) {
    if (sym.name != kd_name) continue;
    if (sym.shndx >= elf_info.sections.size()) continue;
    const auto &sec = elf_info.sections[sym.shndx];
    if (sym.value < sec.addr) continue;
    uint64_t kd_file_offset = sec.offset + (sym.value - sec.addr);
    if (kd_file_offset + kKdSize > elf_size) continue;
    return elf_data + kd_file_offset;
  }
  return nullptr;
}

// ── UpdateKernelDescriptor ───────────────────────────────────────────────────

void UpdateKernelDescriptor(uint8_t *elf_data, size_t elf_size,
                            const ElfInfo &elf_info,
                            const std::string &kernel_name,
                            int32_t extra_vgprs, int32_t extra_sgprs) {
  using namespace llvm::amdhsa;
  uint8_t *kd = FindKernelDescriptor(elf_data, elf_size, elf_info, kernel_name);
  if (!kd) return;

  uint32_t rsrc1;
  std::memcpy(&rsrc1, kd + kKdRsrc1Offset, sizeof(rsrc1));
  if (extra_vgprs > 0) {
    uint32_t current = AMDHSA_BITS_GET(
        rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
    uint32_t extra_granules =
        (static_cast<uint32_t>(extra_vgprs) + kVgprGranuleSize - 1) /
        kVgprGranuleSize;
    uint32_t max_vgpr_gran = static_cast<uint32_t>(
        COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT >>
        COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT);
    AMDHSA_BITS_SET(rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    std::min(current + extra_granules, max_vgpr_gran));
  }
  if (extra_sgprs > 0) {
    uint32_t current = AMDHSA_BITS_GET(
        rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
    uint32_t extra_granules =
        (static_cast<uint32_t>(extra_sgprs) + kSgprGranuleSize - 1) /
        kSgprGranuleSize;
    uint32_t max_sgpr_gran = static_cast<uint32_t>(
        COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT >>
        COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT);
    AMDHSA_BITS_SET(rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    std::min(current + extra_granules, max_sgpr_gran));
  }
  std::memcpy(kd + kKdRsrc1Offset, &rsrc1, sizeof(rsrc1));
}

// ── NOP sled management ──────────────────────────────────────────────────────

NopSled *FindNearestSled(std::vector<NopSled> &sleds, uint64_t offset,
                         uint64_t needed) {
  NopSled *best = nullptr;
  int64_t best_dist = INT64_MAX;
  for (auto &sled : sleds) {
    if (sled.write_pos + needed > sled.end) continue;
    int64_t dist = std::abs(static_cast<int64_t>(sled.write_pos) -
                            static_cast<int64_t>(offset));
    if (dist < kMaxSledDistance && dist < best_dist) {
      best = &sled;
      best_dist = dist;
    }
  }
  return best;
}

// ── GrowElfWithTrampolines ──────────────────────────────────────────────────

static void AdjustSectionHeaders(uint8_t *elf, size_t elf_size,
                                 uint64_t text_offset, uint64_t text_size,
                                 size_t tramp_total) {
  using Ehdr = llvm::ELF::Elf64_Ehdr;
  using Shdr = llvm::ELF::Elf64_Shdr;

  if (elf_size < sizeof(Ehdr)) return;

  uint64_t text_end = text_offset + text_size;
  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum;
  std::memcpy(&e_shoff, elf + offsetof(Ehdr, e_shoff), sizeof(e_shoff));
  std::memcpy(&e_shentsize, elf + offsetof(Ehdr, e_shentsize),
              sizeof(e_shentsize));

  if (e_shoff >= text_end) {
    uint64_t new_shoff = e_shoff + tramp_total;
    std::memcpy(elf + offsetof(Ehdr, e_shoff), &new_shoff, sizeof(new_shoff));
    e_shoff = new_shoff;
  }

  std::memcpy(&e_shnum, elf + offsetof(Ehdr, e_shnum), sizeof(e_shnum));

  for (uint16_t i = 0; i < e_shnum; ++i) {
    uint64_t sh_pos = e_shoff + static_cast<uint64_t>(i) * e_shentsize;
    if (sh_pos + sizeof(Shdr) > elf_size) break;
    uint8_t *sh = elf + sh_pos;
    uint64_t sh_offset;
    std::memcpy(&sh_offset, sh + offsetof(Shdr, sh_offset), sizeof(sh_offset));

    if (sh_offset == text_offset) {
      uint64_t new_text_size = text_size + tramp_total;
      std::memcpy(sh + offsetof(Shdr, sh_size), &new_text_size,
                  sizeof(new_text_size));
    } else if (sh_offset > text_offset) {
      uint64_t new_offset = sh_offset + tramp_total;
      std::memcpy(sh + offsetof(Shdr, sh_offset), &new_offset,
                  sizeof(new_offset));
    }
  }
}

static void AdjustProgramHeaders(uint8_t *elf, size_t elf_size,
                                 uint64_t text_offset, uint64_t text_size,
                                 size_t tramp_total) {
  using Ehdr = llvm::ELF::Elf64_Ehdr;
  using Phdr = llvm::ELF::Elf64_Phdr;

  if (elf_size < sizeof(Ehdr)) return;

  uint64_t text_end = text_offset + text_size;
  uint64_t e_phoff;
  uint16_t e_phentsize, e_phnum;
  std::memcpy(&e_phoff, elf + offsetof(Ehdr, e_phoff), sizeof(e_phoff));
  std::memcpy(&e_phentsize, elf + offsetof(Ehdr, e_phentsize),
              sizeof(e_phentsize));
  std::memcpy(&e_phnum, elf + offsetof(Ehdr, e_phnum), sizeof(e_phnum));

  for (uint16_t i = 0; i < e_phnum; ++i) {
    uint64_t ph_pos = e_phoff + static_cast<uint64_t>(i) * e_phentsize;
    if (ph_pos + sizeof(Phdr) > elf_size) break;
    uint8_t *ph = elf + ph_pos;
    uint64_t p_offset, p_filesz, p_memsz;
    std::memcpy(&p_offset, ph + offsetof(Phdr, p_offset), sizeof(p_offset));
    std::memcpy(&p_filesz, ph + offsetof(Phdr, p_filesz), sizeof(p_filesz));
    std::memcpy(&p_memsz, ph + offsetof(Phdr, p_memsz), sizeof(p_memsz));

    if (p_offset <= text_offset && p_offset + p_filesz >= text_end) {
      p_filesz += tramp_total;
      p_memsz += tramp_total;
      std::memcpy(ph + offsetof(Phdr, p_filesz), &p_filesz, sizeof(p_filesz));
      std::memcpy(ph + offsetof(Phdr, p_memsz), &p_memsz, sizeof(p_memsz));
    } else if (p_offset > text_offset) {
      p_offset += tramp_total;
      std::memcpy(ph + offsetof(Phdr, p_offset), &p_offset, sizeof(p_offset));
    }
  }
}

MallocBuffer GrowElfWithTrampolines(const uint8_t *elf, size_t elf_size,
                                    const ElfInfo &elf_info,
                                    const std::vector<Trampoline> &trampolines) {
  size_t tramp_total = 0;
  for (auto &t : trampolines)
    tramp_total += t.bytes.size();
  if (tramp_total == 0 || tramp_total > SIZE_MAX - elf_size)
    return {};

  MallocBuffer buf(elf_size + tramp_total);
  if (!buf) return {};

  uint8_t *out = buf.get();
  uint64_t text_end = elf_info.text_offset + elf_info.text_size;

  std::memcpy(out, elf, text_end);
  uint64_t pos = text_end;
  for (auto &t : trampolines) {
    std::memcpy(out + pos, t.bytes.data(), t.bytes.size());
    pos += t.bytes.size();
  }
  if (text_end < elf_size)
    std::memcpy(out + pos, elf + text_end, elf_size - text_end);

  size_t new_elf_size = elf_size + tramp_total;
  AdjustSectionHeaders(out, new_elf_size, elf_info.text_offset,
                        elf_info.text_size, tramp_total);
  AdjustProgramHeaders(out, new_elf_size, elf_info.text_offset,
                        elf_info.text_size, tramp_total);
  return buf;
}

// ── PatchElfIsa ──────────────────────────────────────────────────────────────

static constexpr uint32_t kEFlagsMachMask = 0xFFu;

static void PatchIsaInNote(uint8_t *desc, uint32_t descsz,
                           const std::string &target_cpu) {
  std::string orig_isa(reinterpret_cast<const char *>(desc), descsz);
  size_t gfx_pos = orig_isa.find("gfx");
  if (gfx_pos == std::string::npos) return;

  size_t gfx_end = gfx_pos;
  while (gfx_end < orig_isa.size() && orig_isa[gfx_end] != ':' &&
         orig_isa[gfx_end] != '\0')
    ++gfx_end;

  size_t orig_len = gfx_end - gfx_pos;
  if (target_cpu.size() > orig_len) return;

  std::memcpy(desc + gfx_pos, target_cpu.c_str(), target_cpu.size());
  for (size_t j = target_cpu.size(); j < orig_len; ++j)
    desc[gfx_pos + j] = '\0';
}

bool PatchElfIsa(uint8_t *elf, size_t elf_size,
                 const std::string &target_cpu) {
  using ELFT = llvm::object::ELF64LE;
  auto elf_or_err = llvm::object::ELFFile<ELFT>::create(
      llvm::StringRef(reinterpret_cast<const char *>(elf), elf_size));
  if (!elf_or_err) {
    llvm::consumeError(elf_or_err.takeError());
    return false;
  }
  const auto &elf_file = *elf_or_err;

  auto ak = llvm::AMDGPU::parseArchAMDGCN(target_cpu);
  if (ak == llvm::AMDGPU::GPUKind::GK_NONE) return false;
  unsigned target_mach = llvm::ELF::EF_AMDGPU_MACH_NONE;
#define X(VAL, ENUM, NAME)                                                     \
  if (llvm::StringRef(NAME) == target_cpu)                                     \
    target_mach = llvm::ELF::ENUM;
  AMDGPU_MACH_LIST(X)
#undef X
  if (target_mach == llvm::ELF::EF_AMDGPU_MACH_NONE) return false;

  using Ehdr = llvm::ELF::Elf64_Ehdr;

  uint32_t e_flags;
  std::memcpy(&e_flags, elf + offsetof(Ehdr, e_flags), sizeof(e_flags));
  e_flags = (e_flags & ~kEFlagsMachMask) | (target_mach & kEFlagsMachMask);
  std::memcpy(elf + offsetof(Ehdr, e_flags), &e_flags, sizeof(e_flags));

  auto sections_or_err = elf_file.sections();
  if (!sections_or_err) {
    llvm::consumeError(sections_or_err.takeError());
    return true;
  }

  for (const auto &shdr : *sections_or_err) {
    if (shdr.sh_type != llvm::ELF::SHT_NOTE) continue;

    llvm::Error err = llvm::Error::success();
    for (const auto &note : elf_file.notes(shdr, err)) {
      if (note.getType() != kNoteTypeIsaVersion) continue;
      if (note.getName() != kAmdgpuNoteOwner) continue;

      auto desc = note.getDesc(kNoteAlign);
      if (desc.empty()) continue;

      uint64_t desc_offset =
          reinterpret_cast<const uint8_t *>(desc.data()) - elf;
      PatchIsaInNote(elf + desc_offset, desc.size(), target_cpu);
    }
    if (err) {
      llvm::consumeError(std::move(err));
      continue;
    }
  }
  return true;
}

// ── GetKernelVgprCount ───────────────────────────────────────────────────────

int GetKernelVgprCount(const uint8_t *elf_data, size_t elf_size,
                       const ElfInfo &elf_info,
                       const std::string &kernel_name) {
  using namespace llvm::amdhsa;
  const uint8_t *kd = FindKernelDescriptor(
      const_cast<uint8_t *>(elf_data), elf_size, elf_info, kernel_name);
  if (!kd) return -1;
  uint32_t rsrc1;
  std::memcpy(&rsrc1, kd + kKdRsrc1Offset, sizeof(rsrc1));
  uint32_t granulated = AMDHSA_BITS_GET(
      rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  return static_cast<int>((granulated + 1) * kVgprGranularity);
}
