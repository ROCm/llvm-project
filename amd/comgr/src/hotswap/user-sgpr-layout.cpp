//===- user-sgpr-layout.cpp - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "user-sgpr-layout.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR::hotswap {

namespace {

// Append `count` Entry rows to `entries` describing a multi-dword source
// that occupies `count` consecutive SGPRs. Returns the SGPR index of the
// first (low) dword, which the factory stores into the convenience
// `*Sgpr` field for the corresponding source.
int appendSource(llvm::SmallVectorImpl<UserSgprLayout::Entry> &Entries,
                 UserSgprLayout::Source Src, unsigned Count) {
  int FirstIdx = static_cast<int>(Entries.size());
  for (unsigned I = 0; I < Count; ++I) {
    UserSgprLayout::Entry E;
    E.SrcKind = Src;
    E.SubDword = static_cast<uint8_t>(I);
    Entries.push_back(E);
  }
  return FirstIdx;
}

const char *sourceName(UserSgprLayout::Source S) {
  switch (S) {
  case UserSgprLayout::Source::Unset:                return "Unset";
  case UserSgprLayout::Source::PrivateSegmentBuffer: return "PrivateSegmentBuffer";
  case UserSgprLayout::Source::DispatchPtr:          return "DispatchPtr";
  case UserSgprLayout::Source::QueuePtr:             return "QueuePtr";
  case UserSgprLayout::Source::KernargSegmentPtr:    return "KernargSegmentPtr";
  case UserSgprLayout::Source::DispatchId:           return "DispatchId";
  case UserSgprLayout::Source::FlatScratchInit:      return "FlatScratchInit";
  case UserSgprLayout::Source::PrivateSegmentSize:   return "PrivateSegmentSize";
  case UserSgprLayout::Source::PreloadedKernarg:     return "PreloadedKernarg";
  case UserSgprLayout::Source::WorkgroupIdX:         return "WorkgroupIdX";
  case UserSgprLayout::Source::WorkgroupIdY:         return "WorkgroupIdY";
  case UserSgprLayout::Source::WorkgroupIdZ:         return "WorkgroupIdZ";
  case UserSgprLayout::Source::WorkgroupInfo:        return "WorkgroupInfo";
  }
  return "<invalid>";
}

unsigned userSgprCountFieldWidth(const ISAProfile &SourceProfile) {
  using namespace llvm::amdhsa;
  return SourceProfile.HasGfx125UserSgprCountField
             ? COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT_WIDTH
             : COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_WIDTH;
}

unsigned decodeUserSgprCount(uint32_t ComputePgmRsrc2,
                             const ISAProfile &SourceProfile) {
  using namespace llvm::amdhsa;
  const unsigned Width = userSgprCountFieldWidth(SourceProfile);
  return (ComputePgmRsrc2 >> COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT) &
         ((1u << Width) - 1u);
}

std::string formatMetadataMismatch(const KernelMeta &Meta,
                                   llvm::StringRef SourceIsa,
                                   const UserSgprLayout &Layout,
                                   unsigned DecodedUserSgprCount,
                                   unsigned UserSgprCountWidth,
                                   unsigned PreloadLen,
                                   unsigned PreloadOffsetDwords) {
  using namespace llvm::amdhsa;

  std::string Detail;
  llvm::raw_string_ostream Os(Detail);
  Os << "transpiler: UserSgprLayout::fromKernelMeta: kernel '" << Meta.Name
     << "' has compute_pgm_rsrc2.USER_SGPR_COUNT="
     << DecodedUserSgprCount << " (decoded as " << UserSgprCountWidth
     << "-bit field for source ISA '" << SourceIsa
     << "') but kernel_code_properties + kernarg_preload imply "
     << static_cast<unsigned>(Layout.UserSgprCount)
     << ". KD is inconsistent -- refusing to guess the layout. Raw KD fields:"
     << " compute_pgm_rsrc1=0x" << llvm::utohexstr(Meta.ComputePgmRsrc1)
     << " compute_pgm_rsrc2=0x" << llvm::utohexstr(Meta.ComputePgmRsrc2)
     << " kernel_code_properties=0x"
     << llvm::utohexstr(static_cast<unsigned>(Meta.KernelCodeProperties))
     << " kernarg_preload=0x"
     << llvm::utohexstr(static_cast<unsigned>(Meta.KernargPreload))
     << " kernarg_preload_length=" << PreloadLen
     << " kernarg_preload_offset_dwords=" << PreloadOffsetDwords
     << " kernarg_segment_size=" << Meta.KernargSegmentSize
     << " enabled_user_sgprs=[";

  bool First = true;
  auto Append = [&](llvm::StringRef Name, unsigned Count) {
    if (!First)
      Os << ",";
    First = false;
    Os << Name << ":" << Count;
  };

  const uint16_t Kcp = Meta.KernelCodeProperties;
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
    Append("private_segment_buffer", 4);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)
    Append("dispatch_ptr", 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)
    Append("queue_ptr", 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
    Append("kernarg_segment_ptr", 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)
    Append("dispatch_id", 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)
    Append("flat_scratch_init", 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
    Append("private_segment_size", 1);
  if (PreloadLen > 0)
    Append("kernarg_preload", PreloadLen);

  Os << "] system_sgprs=[";
  First = true;
  auto AppendSystem = [&](llvm::StringRef Name) {
    if (!First)
      Os << ",";
    First = false;
    Os << Name;
  };
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X)
    AppendSystem("workgroup_id_x");
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y)
    AppendSystem("workgroup_id_y");
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z)
    AppendSystem("workgroup_id_z");
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO)
    AppendSystem("workgroup_info");
  Os << "]";
  Os.flush();
  return Detail;
}

} // namespace

bool UserSgprLayout::tryFromKernelMeta(const KernelMeta &Meta,
                                       const ISAProfile &SourceProfile,
                                       llvm::StringRef SourceIsa,
                                       UserSgprLayout &Layout,
                                       std::string &FailureDetail) {
  Layout = UserSgprLayout();
  FailureDetail.clear();

  if (!Meta.HasKernelDescriptor) {
    FailureDetail =
        (llvm::Twine("transpiler: UserSgprLayout::fromKernelMeta: kernel '") +
         Meta.Name +
         "' has no parsed kernel descriptor. Cannot derive user-SGPR ABI; "
         "refuse the lift instead of guessing a hardcoded layout.")
            .str();
    return false;
  }

  using namespace llvm::amdhsa;

  const uint16_t Kcp = Meta.KernelCodeProperties;

  // Walk the canonical KERNEL_CODE_PROPERTY bit order.
  // Source: LLVM AMDHSAKernelDescriptor.h (bits 0..6 in ascending order).
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
    Layout.PrivateSegmentBufferSgpr =
        appendSource(Layout.Entries, Source::PrivateSegmentBuffer, 4);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)
    Layout.DispatchPtrSgpr =
        appendSource(Layout.Entries, Source::DispatchPtr, 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)
    Layout.QueuePtrSgpr = appendSource(Layout.Entries, Source::QueuePtr, 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
    Layout.KernargSegmentPtrSgpr =
        appendSource(Layout.Entries, Source::KernargSegmentPtr, 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)
    Layout.DispatchIdSgpr = appendSource(Layout.Entries, Source::DispatchId, 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)
    Layout.FlatScratchInitSgpr =
        appendSource(Layout.Entries, Source::FlatScratchInit, 2);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
    Layout.PrivateSegmentSizeSgpr =
        appendSource(Layout.Entries, Source::PrivateSegmentSize, 1);

  // Kernarg preload (gfx1250+): N dwords from kernarg memory at byte
  // offset (preloadOffset * 4) get loaded into the user SGPRs immediately
  // above the enable_sgpr_*-selected ones, before kernel entry. The
  // sequence is little-endian dword-aligned: dword i goes into
  // s[user_sgpr_count_so_far + i] and corresponds to kernarg bytes
  // [(offset+i)*4 .. (offset+i+1)*4 - 1].
  const uint8_t PreloadLen = static_cast<uint8_t>(
      (Meta.KernargPreload >> KERNARG_PRELOAD_SPEC_LENGTH_SHIFT) &
      ((1 << KERNARG_PRELOAD_SPEC_LENGTH_WIDTH) - 1));
  const uint16_t PreloadOffsetDwords = static_cast<uint16_t>(
      (Meta.KernargPreload >> KERNARG_PRELOAD_SPEC_OFFSET_SHIFT) &
      ((1 << KERNARG_PRELOAD_SPEC_OFFSET_WIDTH) - 1));
  Layout.PreloadedKernargLength = PreloadLen;
  Layout.PreloadedKernargByteOffset =
      static_cast<uint16_t>(PreloadOffsetDwords * 4);
  if (PreloadLen > 0) {
    Layout.FirstPreloadedKernargSgpr = static_cast<int>(Layout.Entries.size());
    for (unsigned I = 0; I < PreloadLen; ++I) {
      Entry E;
      E.SrcKind = Source::PreloadedKernarg;
      // Each preloaded dword is its own independent SGPR (no multi-dword
      // bundling from the KD's perspective -- the byte offset alone identifies
      // which kernarg slice it carries). subDword stays 0 so Phase 4's
      // "act on subDword==0 only" loop visits every preload entry.
      E.SubDword = 0;
      E.KernargByteOffset =
          static_cast<uint16_t>((PreloadOffsetDwords + I) * 4);
      Layout.Entries.push_back(E);
    }
  }

  Layout.UserSgprCount = static_cast<uint8_t>(Layout.Entries.size());

  // Sanity-check against compute_pgm_rsrc2.USER_SGPR_COUNT. gfx125 widens
  // this field to 6 bits; using the older 5-bit decode would read a valid
  // count of 32 as zero and falsely reject Triton gfx1250 kernels.
  const unsigned UserSgprCountWidth = userSgprCountFieldWidth(SourceProfile);
  const unsigned PgmRsrc2UserSgprCount =
      decodeUserSgprCount(Meta.ComputePgmRsrc2, SourceProfile);
  if (PgmRsrc2UserSgprCount != Layout.UserSgprCount) {
    FailureDetail =
        formatMetadataMismatch(Meta, SourceIsa, Layout, PgmRsrc2UserSgprCount,
                               UserSgprCountWidth, PreloadLen,
                               PreloadOffsetDwords);
    return false;
  }

  // Workgroup ID SGPRs sit immediately above the user-SGPR region.
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X)
    Layout.WorkgroupIdXSgpr =
        appendSource(Layout.Entries, Source::WorkgroupIdX, 1);
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y)
    Layout.WorkgroupIdYSgpr =
        appendSource(Layout.Entries, Source::WorkgroupIdY, 1);
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z)
    Layout.WorkgroupIdZSgpr =
        appendSource(Layout.Entries, Source::WorkgroupIdZ, 1);
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO)
    Layout.WorkgroupInfoSgpr =
        appendSource(Layout.Entries, Source::WorkgroupInfo, 1);

  return true;
}

UserSgprLayout UserSgprLayout::fromKernelMeta(const KernelMeta &Meta,
                                              const ISAProfile &SourceProfile,
                                              llvm::StringRef SourceIsa) {
  UserSgprLayout Layout;
  std::string FailureDetail;
  if (!tryFromKernelMeta(Meta, SourceProfile, SourceIsa, Layout, FailureDetail))
    llvm::report_fatal_error(llvm::StringRef(FailureDetail));
  return Layout;
}

std::string UserSgprLayout::toString() const {
  std::string Result;
  llvm::raw_string_ostream Os(Result);
  Os << "user_sgpr_count=" << static_cast<int>(UserSgprCount);
  for (size_t I = 0; I < Entries.size(); ++I) {
    const auto &E = Entries[I];
    Os << " s[" << I << "]=" << sourceName(E.SrcKind);
    if (E.SrcKind == Source::PreloadedKernarg)
      Os << "(off=" << E.KernargByteOffset << ")";
    else if (E.SubDword > 0)
      Os << "[" << static_cast<int>(E.SubDword) << "]";
  }
  return Os.str();
}

} // namespace COMGR::hotswap
