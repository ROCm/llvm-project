//===- comgr-hotswap-entry-trampoline-fast.cpp - B0->B0 fast path ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// B0-on-B0 kernel-entry trampoline FAST PATH. Mirrors the ROCr loader
/// trampoline: no LLVM MC layer (no initLLVM, no assembler, no disassembler).
/// The stub is emitted from a pre-encoded gfx1250 byte template with only the
/// two PC-relative delta immediates patched in, using a fixed s[100:101]
/// scratch pair (so no per-kernel SGPR read and no descriptor SGPR-reservation
/// update). Idempotency and the compile-time-workaround skip are decided by raw
/// byte comparison rather than decoding.
///
/// This path is selected automatically for pure B0->B0 entry-only rewrites
/// (no B0->A0 instruction patches, no mask workaround). The MC-based path in
/// comgr-hotswap-entry-trampoline.cpp handles A0 and any rewrite that needs
/// instruction patches.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <cstring>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// Pre-encoded gfx1250 stub template (fixed s[100:101]). Ground-truth encodings
// from llvm-mc -mcpu=gfx1250 (little-endian). The body is 40 bytes and is
// padded to KernelEntryStubStride (256) with s_code_end. s_get_pc_i64 loads the
// address of the instruction after it (s_add, at StubVAddr +
// FastEntryPcBaseOffset), which is the base for the PC-relative delta.
// clang-format off
static constexpr uint8_t StubTemplate[FastEntryStubBodyBytes] = {
    0x7c, 0x00, 0x0b, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // global_wb
    0x00, 0x00, 0x00, 0x7e,                                                 // v_nop
    0x00, 0x47, 0xe4, 0xbe,                                                 // s_get_pc_i64 s[100:101]
    0x64, 0xff, 0x64, 0x80, 0x00, 0x00, 0x00, 0x00,                         // s_add_co_u32 s100,s100,imm32 (imm@24)
    0x65, 0xff, 0x65, 0x82, 0x00, 0x00, 0x00, 0x00,                         // s_add_co_ci_u32 s101,s101,imm32 (imm@32)
    0x64, 0x48, 0x80, 0xbe,                                                 // s_set_pc_i64 s[100:101]
};

static constexpr uint8_t SCodeEnd[4] = {0x00, 0x00, 0x9f, 0xbf};
static constexpr uint8_t SNop[4] = {0x00, 0x00, 0x80, 0xbf};

// global_wb; v_nop prefix (16 bytes), for raw idempotency / workaround detection.
static constexpr uint8_t EntryPrefix[FastEntryPrefixBytes] = {
    0x7c, 0x00, 0x0b, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x7e,
};
// clang-format on

SmallVector<uint8_t> buildKernelEntryTrampolineFast(uint64_t StubVAddr,
                                                    uint64_t EntryVAddr) {
  SmallVector<uint8_t> Bytes;
  Bytes.resize(KernelEntryStubStride);
  std::memcpy(Bytes.data(), StubTemplate, FastEntryStubBodyBytes);

  // Materialize EntryVAddr relative to the s_get_pc base. Two's complement, so
  // back-jumps are handled.
  const uint64_t PcBase = StubVAddr + FastEntryPcBaseOffset;
  const uint64_t Delta = EntryVAddr - PcBase;
  llvm::support::endian::write32le(Bytes.data() + FastEntryDeltaLoOffset,
                                   static_cast<uint32_t>(Delta));
  llvm::support::endian::write32le(Bytes.data() + FastEntryDeltaHiOffset,
                                   static_cast<uint32_t>(Delta >> 32));

  // Pad to stride with s_code_end (prefetch-safe; never executed).
  for (uint64_t Off = FastEntryStubBodyBytes; Off < KernelEntryStubStride;
       Off += sizeof(SCodeEnd))
    std::memcpy(Bytes.data() + Off, SCodeEnd, sizeof(SCodeEnd));
  return Bytes;
}

// Raw byte check: does the descriptor's current entry already begin with
// global_wb; v_nop (either a hotswap fast stub already installed, or the
// compile-time unclaused-VMEM workaround prologue)? Both mean "do not add a
// trampoline".
static std::optional<bool>
entryHasWorkaroundPrefixFast(const ElfView &Elf,
                             const KernelDescriptorInfo &KD) {
  std::optional<uint64_t> Entry = entryVAddr(KD);
  if (!Entry)
    return std::nullopt;
  const uint8_t *EntryBytes = Elf.dataAtVAddr(*Entry, FastEntryPrefixBytes);
  if (!EntryBytes)
    return false;
  return std::memcmp(EntryBytes, EntryPrefix, FastEntryPrefixBytes) == 0;
}

static bool appendPaddingFast(std::vector<Trampoline> &Out, uint64_t PadBytes) {
  if (PadBytes == 0)
    return true;
  if (PadBytes % sizeof(SNop) != 0) {
    log() << "hotswap: error: fast entry-stub padding " << PadBytes
          << " is not a multiple of s_nop size.\n";
    return false;
  }
  Trampoline Pad;
  Pad.Bytes.reserve(PadBytes);
  while (static_cast<uint64_t>(Pad.Bytes.size()) < PadBytes)
    Pad.Bytes.append(SNop, SNop + sizeof(SNop));
  Out.push_back(std::move(Pad));
  return true;
}

std::optional<uint32_t> appendKernelEntryTrampolinesFast(
    const ElfView &Elf, StringRef TargetCpu, std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups) {
  ArrayRef<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
  if (Descriptors.empty())
    return 0;

  struct WorkItem {
    KernelDescriptorInfo KD;
    uint32_t StubInstPrefLines = 0;
  };
  std::vector<WorkItem> Work;
  uint32_t MaxStubInstPrefLines = 0;

  for (const KernelDescriptorInfo &KD : Descriptors) {
    // Skip if the entry already carries the workaround (already-installed fast
    // stub, or a compile-time global_wb; v_nop prologue). Raw byte check.
    std::optional<bool> HasPrefix = entryHasWorkaroundPrefixFast(Elf, KD);
    if (!HasPrefix)
      return std::nullopt;
    if (*HasPrefix) {
      log() << "hotswap: fast: kernel '" << KD.KernelName
            << "' entry already has global_wb; v_nop; skipping trampoline\n";
      continue;
    }
    std::optional<uint32_t> OriginalInstPrefLines =
        Elf.getKernelDescriptorInstPrefSize(KD.KernelName, TargetCpu);
    if (!OriginalInstPrefLines)
      return std::nullopt;
    uint32_t StubInstPrefLines =
        std::min(*OriginalInstPrefLines, KernelEntryStubInstPrefLines);
    MaxStubInstPrefLines = std::max(MaxStubInstPrefLines, StubInstPrefLines);
    Work.push_back({KD, StubInstPrefLines});
  }
  if (Work.empty())
    return 0;

  uint64_t AppendOffset = 0;
  for (const Trampoline &T : Growth)
    AppendOffset += T.Bytes.size();

  std::optional<uint64_t> PoolVAddrOr = Elf.trampolinePoolVAddr();
  if (!PoolVAddrOr)
    return std::nullopt;
  const uint64_t PoolVAddr = *PoolVAddrOr;

  std::optional<uint64_t> StubPoolBaseVAddr =
      checkedAddUint64(PoolVAddr, AppendOffset, "fast entry stub-pool base");
  if (!StubPoolBaseVAddr)
    return std::nullopt;
  std::optional<uint64_t> AlignedBase =
      checkedAlignTo(*StubPoolBaseVAddr, KernelEntryStubStride,
                     "fast entry trampoline aligned stub-pool base");
  if (!AlignedBase)
    return std::nullopt;
  const uint64_t StubStart = *AlignedBase - PoolVAddr;

  std::vector<Trampoline> LocalGrowth;
  std::vector<KernelEntryTrampolineFixup> LocalFixups;
  if (!appendPaddingFast(LocalGrowth, StubStart - AppendOffset))
    return std::nullopt;
  AppendOffset = StubStart;

  for (const WorkItem &Item : Work) {
    const KernelDescriptorInfo &KD = Item.KD;
    std::optional<uint64_t> StubVAddr = checkedAddUint64(
        PoolVAddr, AppendOffset,
        (Twine("fast entry trampoline vaddr for '") + KD.KernelName + "'")
            .str());
    if (!StubVAddr)
      return std::nullopt;
    std::optional<uint64_t> Entry = entryVAddr(KD);
    if (!Entry)
      return std::nullopt;

    Trampoline T;
    T.Bytes = buildKernelEntryTrampolineFast(*StubVAddr, *Entry);
    LocalGrowth.push_back(std::move(T));

    // Fixed s[100:101]: no per-kernel SGPR read, and SkipSgprReservation tells
    // the descriptor rewrite to leave the logical SGPR count unchanged.
    LocalFixups.push_back({KD.KernelName, AppendOffset, /*RequiredSgprs=*/0,
                           Item.StubInstPrefLines,
                           /*SkipSgprReservation=*/true});

    std::optional<uint64_t> NewAppendOffset = checkedAddUint64(
        AppendOffset, KernelEntryStubStride,
        (Twine("fast entry append offset after '") + KD.KernelName + "'")
            .str());
    if (!NewAppendOffset)
      return std::nullopt;
    AppendOffset = *NewAppendOffset;
  }

  // Prefetch guard sized like the MC path (shared helper).
  const uint64_t GuardBytes =
      computeKernelEntryPrefetchGuardBytes(MaxStubInstPrefLines);
  if (GuardBytes != 0) {
    Trampoline Guard;
    Guard.Bytes.reserve(GuardBytes);
    for (uint64_t Off = 0; Off < GuardBytes; Off += sizeof(SCodeEnd))
      Guard.Bytes.append(SCodeEnd, SCodeEnd + sizeof(SCodeEnd));
    LocalGrowth.push_back(std::move(Guard));
  }

  if (LocalFixups.empty())
    return 0;
  if (LocalFixups.size() > std::numeric_limits<uint32_t>::max()) {
    log() << "hotswap: error: fast kernel-entry trampoline count "
          << LocalFixups.size() << " exceeds uint32_t.\n";
    return std::nullopt;
  }

  for (Trampoline &T : LocalGrowth)
    Growth.push_back(std::move(T));
  OutFixups.insert(OutFixups.end(), LocalFixups.begin(), LocalFixups.end());

  log() << "hotswap: fast: installed " << LocalFixups.size()
        << " kernel-entry trampoline" << (LocalFixups.size() == 1 ? "" : "s")
        << " (no-disasm, fixed s[100:101]) with " << GuardBytes
        << " prefetch guard bytes\n";
  return static_cast<uint32_t>(LocalFixups.size());
}

} // namespace hotswap
} // namespace COMGR
