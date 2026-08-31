//===- user-sgpr-layout.h - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_USER_SGPR_LAYOUT_H
#define HOTSWAP_TRANSPILER_USER_SGPR_LAYOUT_H

#include "hotswap/common/kernel-meta.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace llvm {
class MCSubtargetInfo;
class raw_ostream;
} // namespace llvm

namespace COMGR::hotswap {

// What each SGPR contains at function entry on the source ISA, derived from the
// kernel descriptor (KernelMeta's KernelCodeProperties, KernargPreload,
// ComputePgmRsrc2). This is the single source of truth for the source-ISA SGPR
// ABI inside the raiser: the layout depends on kernarg-preload and
// kernel_code_properties, so handlers must consult it rather than assuming
// fixed SGPR indices.
//
// Entries is indexed by SGPR index, so Entries[i] describes the contents of
// SGPR i at kernel entry. The convenience `*Sgpr()` accessors return the SGPR
// index of the first dword of the corresponding source, or no value when the
// source is disabled, so handlers can identify a source by comparison rather
// than walking Entries.
//
// tryFromKernelMeta reports an inconsistent descriptor through llvm::Error; it
// never falls back to a hardcoded layout.
struct UserSgprLayout {
  enum class Source : uint8_t {
    Unset,
    PrivateSegmentBuffer, // 4 dwords (s[i:i+3]) -- shader resource descriptor
    DispatchPtr,          // 2 dwords -- pointer to the AQL dispatch packet
    QueuePtr,             // 2 dwords -- pointer to the HSA queue object
    KernargSegmentPtr,    // 2 dwords -- pointer to the kernarg segment
    DispatchId,           // 2 dwords -- dispatch identifier
    FlatScratchInit,      // 2 dwords -- flat scratch base/size init
    PrivateSegmentSize,   // 1 dword  -- size of private segment per work-item
    PreloadedKernarg,     // 1 dword  -- preloaded kernarg dword (gfx1250 ABI)
    WorkgroupIdX,         // 1 dword  -- system SGPR (compute_pgm_rsrc2 bit 7)
    WorkgroupIdY,         // 1 dword  -- system SGPR (compute_pgm_rsrc2 bit 8)
    WorkgroupIdZ,         // 1 dword  -- system SGPR (compute_pgm_rsrc2 bit 9)
    WorkgroupInfo,        // 1 dword  -- system SGPR (compute_pgm_rsrc2 bit 10)
  };

  // Number of consecutive SGPRs the source occupies.
  static unsigned dwordCount(Source Src);

  struct Entry {
    Source SrcKind = Source::Unset;
    // For multi-dword sources (DispatchPtr, KernargSegmentPtr, ...) this is
    // the dword index within the source: 0 = lo, 1 = hi for 2-dword
    // sources, 0..3 for the 4-dword private segment buffer descriptor.
    uint8_t SubDword = 0;
    // For PreloadedKernarg only: the byte offset within the kernarg segment
    // that this dword originated from. Computed as
    // `(kernarg_preload_offset + i) * 4` per the gfx1250 ABI. Used by the
    // raiser to look up the matching kernarg parameter and extract the
    // appropriate dword from it.
    uint16_t KernargByteOffset = 0;
  };

  llvm::SmallVector<Entry> Entries;
  uint8_t UserSgprCount = 0; // == Entries.size() at end of user-SGPR region

  // SGPR index that holds the low dword of the corresponding source, or no
  // value when the source is not enabled in `kernel_code_properties` /
  // `compute_pgm_rsrc2`. Handlers identify a source by comparing against these
  // rather than assuming a fixed SGPR index.
  std::optional<unsigned> kernargSegmentPtrSgpr() const {
    return KernargSegmentPtrSgpr;
  }
  std::optional<unsigned> dispatchPtrSgpr() const { return DispatchPtrSgpr; }
  std::optional<unsigned> queuePtrSgpr() const { return QueuePtrSgpr; }
  std::optional<unsigned> dispatchIdSgpr() const { return DispatchIdSgpr; }
  std::optional<unsigned> flatScratchInitSgpr() const {
    return FlatScratchInitSgpr;
  }
  std::optional<unsigned> privateSegmentBufferSgpr() const {
    return PrivateSegmentBufferSgpr;
  }
  std::optional<unsigned> privateSegmentSizeSgpr() const {
    return PrivateSegmentSizeSgpr;
  }
  std::optional<unsigned> workgroupIdXSgpr() const { return WorkgroupIdXSgpr; }
  std::optional<unsigned> workgroupIdYSgpr() const { return WorkgroupIdYSgpr; }
  std::optional<unsigned> workgroupIdZSgpr() const { return WorkgroupIdZSgpr; }
  std::optional<unsigned> workgroupInfoSgpr() const {
    return WorkgroupInfoSgpr;
  }

  // SGPR index of the first preloaded kernarg dword, or no value when no
  // kernarg preload is enabled.
  std::optional<unsigned> firstPreloadedKernargSgpr() const {
    return FirstPreloadedKernargSgpr;
  }
  uint8_t preloadedKernargLength() const { return PreloadedKernargLength; }
  uint16_t preloadedKernargByteOffset() const {
    return PreloadedKernargByteOffset;
  }

  // Build the layout from a parsed kernel descriptor. Returns llvm::Error
  // when the descriptor is missing or internally inconsistent.
  // `sourceSTI` selects ABI-versioned fields such as gfx125's 6-bit
  // compute_pgm_rsrc2.USER_SGPR_COUNT. `sourceISA` is used only in diagnostics.
  static llvm::Error tryFromKernelMeta(const KernelMeta &Meta,
                                       const llvm::MCSubtargetInfo &SourceSTI,
                                       llvm::StringRef SourceIsa,
                                       UserSgprLayout &Layout);

  // Stream a one-line debug summary useful for HSA_HOTSWAP_DEBUG output and
  // failure diagnostics, in the form:
  //   "user_sgpr_count=N s[0]=KernargSegmentPtr s[2]=PreloadedKernarg(off=0)"
  void print(llvm::raw_ostream &OS) const;

private:
  std::optional<unsigned> KernargSegmentPtrSgpr;
  std::optional<unsigned> DispatchPtrSgpr;
  std::optional<unsigned> QueuePtrSgpr;
  std::optional<unsigned> DispatchIdSgpr;
  std::optional<unsigned> FlatScratchInitSgpr;
  std::optional<unsigned> PrivateSegmentBufferSgpr;
  std::optional<unsigned> PrivateSegmentSizeSgpr;
  std::optional<unsigned> WorkgroupIdXSgpr;
  std::optional<unsigned> WorkgroupIdYSgpr;
  std::optional<unsigned> WorkgroupIdZSgpr;
  std::optional<unsigned> WorkgroupInfoSgpr;
  std::optional<unsigned> FirstPreloadedKernargSgpr;
  uint8_t PreloadedKernargLength = 0;
  uint16_t PreloadedKernargByteOffset = 0;
};

} // namespace COMGR::hotswap

#endif
