//===- user-sgpr-layout.cpp - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/user-sgpr-layout.h"

#include "hotswap/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR::hotswap {

namespace {

// Append one Entry row per dword the source occupies. Returns the SGPR index of
// the first (low) dword, which the caller stores into the convenience `*Sgpr`
// field for the corresponding source.
unsigned appendSource(llvm::SmallVectorImpl<UserSgprLayout::Entry> &Entries,
                      UserSgprLayout::Source Src) {
  unsigned FirstIdx = Entries.size();
  for (unsigned I = 0, E = UserSgprLayout::dwordCount(Src); I < E; ++I) {
    UserSgprLayout::Entry Row;
    Row.SrcKind = Src;
    Row.SubDword = static_cast<uint8_t>(I);
    Entries.push_back(Row);
  }
  return FirstIdx;
}

llvm::StringRef sourceName(UserSgprLayout::Source S) {
  switch (S) {
  case UserSgprLayout::Source::Unset:
    return "Unset";
  case UserSgprLayout::Source::PrivateSegmentBuffer:
    return "PrivateSegmentBuffer";
  case UserSgprLayout::Source::DispatchPtr:
    return "DispatchPtr";
  case UserSgprLayout::Source::QueuePtr:
    return "QueuePtr";
  case UserSgprLayout::Source::KernargSegmentPtr:
    return "KernargSegmentPtr";
  case UserSgprLayout::Source::DispatchId:
    return "DispatchId";
  case UserSgprLayout::Source::FlatScratchInit:
    return "FlatScratchInit";
  case UserSgprLayout::Source::PrivateSegmentSize:
    return "PrivateSegmentSize";
  case UserSgprLayout::Source::PreloadedKernarg:
    return "PreloadedKernarg";
  case UserSgprLayout::Source::WorkgroupIdX:
    return "WorkgroupIdX";
  case UserSgprLayout::Source::WorkgroupIdY:
    return "WorkgroupIdY";
  case UserSgprLayout::Source::WorkgroupIdZ:
    return "WorkgroupIdZ";
  case UserSgprLayout::Source::WorkgroupInfo:
    return "WorkgroupInfo";
  }
  return "<invalid>";
}

unsigned userSgprCountFieldWidth(const llvm::MCSubtargetInfo &SourceSTI) {
  using namespace llvm::amdhsa;
  return SourceSTI.hasFeature(llvm::AMDGPU::FeatureGFX1250Insts)
             ? COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT_WIDTH
             : COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_WIDTH;
}

unsigned decodeUserSgprCount(uint32_t ComputePgmRsrc2,
                             const llvm::MCSubtargetInfo &SourceSTI) {
  using namespace llvm::amdhsa;
  return SourceSTI.hasFeature(llvm::AMDGPU::FeatureGFX1250Insts)
             ? AMDHSA_BITS_GET(ComputePgmRsrc2,
                               COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT)
             : AMDHSA_BITS_GET(ComputePgmRsrc2,
                               COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT);
}

void formatMetadataMismatch(llvm::raw_ostream &Os, const KernelMeta &Meta,
                            llvm::StringRef SourceIsa,
                            const UserSgprLayout &Layout,
                            unsigned DecodedUserSgprCount,
                            unsigned UserSgprCountWidth, unsigned PreloadLen,
                            unsigned PreloadOffsetDwords) {
  using namespace llvm::amdhsa;

  Os << "transpiler: UserSgprLayout::fromKernelMeta: kernel '" << Meta.Name
     << "' has compute_pgm_rsrc2.USER_SGPR_COUNT=" << DecodedUserSgprCount
     << " (decoded as " << UserSgprCountWidth << "-bit field for source ISA '"
     << SourceIsa << "') but kernel_code_properties + kernarg_preload imply "
     << static_cast<unsigned>(Layout.UserSgprCount)
     << ". Kernel descriptor is inconsistent -- refusing to guess the layout. "
        "Raw descriptor fields:"
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

  using Source = UserSgprLayout::Source;
  const uint16_t Kcp = Meta.KernelCodeProperties;
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
    Append("private_segment_buffer",
           UserSgprLayout::dwordCount(Source::PrivateSegmentBuffer));
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)
    Append("dispatch_ptr", UserSgprLayout::dwordCount(Source::DispatchPtr));
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)
    Append("queue_ptr", UserSgprLayout::dwordCount(Source::QueuePtr));
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
    Append("kernarg_segment_ptr",
           UserSgprLayout::dwordCount(Source::KernargSegmentPtr));
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)
    Append("dispatch_id", UserSgprLayout::dwordCount(Source::DispatchId));
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)
    Append("flat_scratch_init",
           UserSgprLayout::dwordCount(Source::FlatScratchInit));
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
    Append("private_segment_size",
           UserSgprLayout::dwordCount(Source::PrivateSegmentSize));
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
}

} // namespace

unsigned UserSgprLayout::dwordCount(Source Src) {
  switch (Src) {
  case Source::PrivateSegmentBuffer:
    return 4;
  case Source::DispatchPtr:
  case Source::QueuePtr:
  case Source::KernargSegmentPtr:
  case Source::DispatchId:
  case Source::FlatScratchInit:
    return 2;
  case Source::PrivateSegmentSize:
  case Source::PreloadedKernarg:
  case Source::WorkgroupIdX:
  case Source::WorkgroupIdY:
  case Source::WorkgroupIdZ:
  case Source::WorkgroupInfo:
    return 1;
  case Source::Unset:
    return 0;
  }
  llvm_unreachable("unhandled UserSgprLayout::Source");
}

llvm::Error UserSgprLayout::tryFromKernelMeta(
    const KernelMeta &Meta, const llvm::MCSubtargetInfo &SourceSTI,
    llvm::StringRef SourceIsa, UserSgprLayout &Layout) {
  Layout = UserSgprLayout();

  using namespace llvm::amdhsa;

  const uint16_t Kcp = Meta.KernelCodeProperties;

  // The ABI assigns user SGPRs in ascending kernel-code-property bit order.
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
    Layout.PrivateSegmentBufferSgpr =
        appendSource(Layout.Entries, Source::PrivateSegmentBuffer);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)
    Layout.DispatchPtrSgpr = appendSource(Layout.Entries, Source::DispatchPtr);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)
    Layout.QueuePtrSgpr = appendSource(Layout.Entries, Source::QueuePtr);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
    Layout.KernargSegmentPtrSgpr =
        appendSource(Layout.Entries, Source::KernargSegmentPtr);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)
    Layout.DispatchIdSgpr = appendSource(Layout.Entries, Source::DispatchId);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)
    Layout.FlatScratchInitSgpr =
        appendSource(Layout.Entries, Source::FlatScratchInit);
  if (Kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
    Layout.PrivateSegmentSizeSgpr =
        appendSource(Layout.Entries, Source::PrivateSegmentSize);

  // Preloaded dwords follow the enable_sgpr_*-selected entries in the source
  // ABI.
  const uint8_t PreloadLen = static_cast<uint8_t>(
      AMDHSA_BITS_GET(Meta.KernargPreload, KERNARG_PRELOAD_SPEC_LENGTH));
  const uint16_t PreloadOffsetDwords = static_cast<uint16_t>(
      AMDHSA_BITS_GET(Meta.KernargPreload, KERNARG_PRELOAD_SPEC_OFFSET));
  if (Meta.KernargPreload != 0 &&
      !SourceSTI.hasFeature(llvm::AMDGPU::FeatureKernargPreload))
    return RaiseFailure::inKernel(RaiseFailureReason::UserSgprLayoutMismatch,
                                  Meta.Name,
                                  llvm::Twine("source ISA '") + SourceIsa +
                                      "' does not support kernarg preloading");
  Layout.PreloadedKernargLength = PreloadLen;
  Layout.PreloadedKernargByteOffset =
      static_cast<uint16_t>(PreloadOffsetDwords * 4);
  if (PreloadLen > 0) {
    const uint64_t PreloadEnd =
        (static_cast<uint64_t>(PreloadOffsetDwords) + PreloadLen) * 4;
    if (PreloadEnd > Meta.KernargSegmentSize)
      return RaiseFailure::inKernel(
          RaiseFailureReason::UserSgprLayoutMismatch, Meta.Name,
          llvm::Twine("kernarg preload ends at byte ") +
              llvm::Twine(PreloadEnd) + " beyond kernarg segment size " +
              llvm::Twine(Meta.KernargSegmentSize));

    Layout.FirstPreloadedKernargSgpr = Layout.Entries.size();
    for (unsigned I = 0; I < PreloadLen; ++I) {
      Entry Row;
      Row.SrcKind = Source::PreloadedKernarg;
      // Each preloaded dword is its own independent SGPR: the byte offset alone
      // identifies which kernarg slice it carries, so it is not bundled as a
      // multi-dword source. SubDword stays 0 for every preload entry.
      Row.SubDword = 0;
      Row.KernargByteOffset =
          static_cast<uint16_t>((PreloadOffsetDwords + I) * 4);
      Layout.Entries.push_back(Row);
    }
  }

  const unsigned ImpliedUserSgprCount = Layout.Entries.size();
  Layout.UserSgprCount = static_cast<uint8_t>(ImpliedUserSgprCount);

  // USER_SGPR_COUNT is 6 bits on gfx1250, where 32 is a valid count.
  const unsigned UserSgprCountWidth = userSgprCountFieldWidth(SourceSTI);
  const unsigned PgmRsrc2UserSgprCount =
      decodeUserSgprCount(Meta.ComputePgmRsrc2, SourceSTI);
  const unsigned MaxUserSgprs =
      SourceSTI.hasFeature(llvm::AMDGPU::FeatureGFX1250Insts) ? 32 : 16;
  if (PgmRsrc2UserSgprCount > MaxUserSgprs)
    return RaiseFailure::inKernel(
        RaiseFailureReason::UserSgprLayoutMismatch, Meta.Name,
        llvm::Twine("compute_pgm_rsrc2.USER_SGPR_COUNT=") +
            llvm::Twine(PgmRsrc2UserSgprCount) + " exceeds source ISA '" +
            SourceIsa + "' maximum of " + llvm::Twine(MaxUserSgprs));
  if (PgmRsrc2UserSgprCount < ImpliedUserSgprCount) {
    std::string Detail;
    llvm::raw_string_ostream Os(Detail);
    formatMetadataMismatch(Os, Meta, SourceIsa, Layout, PgmRsrc2UserSgprCount,
                           UserSgprCountWidth, PreloadLen, PreloadOffsetDwords);
    return RaiseFailure::inKernel(RaiseFailureReason::UserSgprLayoutMismatch,
                                  Meta.Name, Detail);
  }

  Layout.Entries.resize(PgmRsrc2UserSgprCount);
  Layout.UserSgprCount = static_cast<uint8_t>(PgmRsrc2UserSgprCount);

  // Architected workgroup IDs do not consume sequential SGPRs.
  if (!SourceSTI.hasFeature(llvm::AMDGPU::FeatureArchitectedSGPRs) &&
      Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X)
    Layout.WorkgroupIdXSgpr =
        appendSource(Layout.Entries, Source::WorkgroupIdX);
  if (!SourceSTI.hasFeature(llvm::AMDGPU::FeatureArchitectedSGPRs) &&
      Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y)
    Layout.WorkgroupIdYSgpr =
        appendSource(Layout.Entries, Source::WorkgroupIdY);
  if (!SourceSTI.hasFeature(llvm::AMDGPU::FeatureArchitectedSGPRs) &&
      Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z)
    Layout.WorkgroupIdZSgpr =
        appendSource(Layout.Entries, Source::WorkgroupIdZ);
  if (Meta.ComputePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO)
    Layout.WorkgroupInfoSgpr =
        appendSource(Layout.Entries, Source::WorkgroupInfo);

  return llvm::Error::success();
}

void UserSgprLayout::print(llvm::raw_ostream &OS) const {
  OS << "user_sgpr_count=" << static_cast<int>(UserSgprCount);
  for (size_t I = 0; I < Entries.size(); ++I) {
    const Entry &E = Entries[I];
    OS << " s[" << I << "]=" << sourceName(E.SrcKind);
    if (E.SrcKind == Source::PreloadedKernarg)
      OS << "(off=" << E.KernargByteOffset << ")";
    else if (E.SubDword > 0)
      OS << "[" << static_cast<int>(E.SubDword) << "]";
  }
}

} // namespace COMGR::hotswap
