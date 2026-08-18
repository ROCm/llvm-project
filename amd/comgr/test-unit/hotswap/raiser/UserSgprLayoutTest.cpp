//===- UserSgprLayoutTest.cpp - Hotswap user-SGPR layout tests ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Covers the user-SGPR layout the raiser seeds its entry registers from: the
// canonical entry order the kernel descriptor implies, the kernarg preload
// decode, and the descriptors the layout refuses.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise_failure.h"
#include "hotswap/raiser/user-sgpr-layout.h"

#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"

#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <mutex>

// initMCState calls COMGR::ensureLLVMInitialized, whose production definition
// lives in libamd_comgr. Provide the registration here so the test binary stays
// minimal instead of linking the full Comgr.
namespace COMGR {
void ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
  });
}
} // namespace COMGR

using namespace COMGR::hotswap;
using namespace llvm;

namespace {

// Reason of the RaiseFailure carried by an Error, or None when the Error is not
// a RaiseFailure. Consumes the Error.
RaiseFailureReason reasonOf(Error E) {
  RaiseFailureReason Reason = RaiseFailureReason::None;
  handleAllErrors(std::move(E),
                  [&](const RaiseFailure &F) { Reason = F.reason(); });
  return Reason;
}

// Whether an Error is success, consuming it either way (llvm::Error is not
// directly convertible to the bool gtest's ASSERT_* macros expect).
bool succeeded(Error E) {
  if (E) {
    consumeError(std::move(E));
    return false;
  }
  return true;
}

// Owns an MCState so the ISAProfile's referenced subtarget outlives it.
class Profile {
public:
  explicit Profile(StringRef Isa) {
    if (Expected<MCState> S = initMCState(Isa)) {
      State = std::move(*S);
      Ok = true;
    } else {
      consumeError(S.takeError());
    }
  }
  bool ok() const { return Ok && State.SubtargetInfo != nullptr; }
  ISAProfile get() const {
    return ISAProfile::fromSubtarget(*State.SubtargetInfo);
  }

private:
  bool Ok = false;
  MCState State;
};

TEST(UserSgprLayout, DerivesCanonicalOrderAndAccessors) {
  Profile P("gfx942");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR |
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  Meta.ComputePgmRsrc2 =
      (4u << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT) |
      COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X;

  UserSgprLayout Layout;
  ASSERT_TRUE(succeeded(
      UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx942", Layout)));

  // dispatch_ptr precedes kernarg_segment_ptr in the canonical order; the
  // workgroup id follows the user-SGPR region and is not counted in it.
  EXPECT_EQ(Layout.UserSgprCount, 4u);
  ASSERT_EQ(Layout.Entries.size(), 5u);
  ASSERT_TRUE(Layout.dispatchPtrSgpr().has_value());
  EXPECT_EQ(*Layout.dispatchPtrSgpr(), 0u);
  ASSERT_TRUE(Layout.kernargSegmentPtrSgpr().has_value());
  EXPECT_EQ(*Layout.kernargSegmentPtrSgpr(), 2u);
  ASSERT_TRUE(Layout.workgroupIdXSgpr().has_value());
  EXPECT_EQ(*Layout.workgroupIdXSgpr(), 4u);
  EXPECT_EQ(Layout.Entries[1].SrcKind, UserSgprLayout::Source::DispatchPtr);
  EXPECT_EQ(Layout.Entries[1].SubDword, 1u);
  EXPECT_EQ(Layout.Entries[4].SrcKind, UserSgprLayout::Source::WorkgroupIdX);
}

TEST(UserSgprLayout, DisabledSourcesHaveNoSgpr) {
  Profile P("gfx942");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  Meta.ComputePgmRsrc2 = 2u
                         << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout Layout;
  ASSERT_TRUE(succeeded(
      UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx942", Layout)));
  ASSERT_TRUE(Layout.kernargSegmentPtrSgpr().has_value());
  EXPECT_EQ(*Layout.kernargSegmentPtrSgpr(), 0u);
  EXPECT_FALSE(Layout.dispatchPtrSgpr().has_value());
  EXPECT_FALSE(Layout.queuePtrSgpr().has_value());
  EXPECT_FALSE(Layout.firstPreloadedKernargSgpr().has_value());
}

TEST(UserSgprLayout, EntryRunLengthMatchesDwordCount) {
  Profile P("gfx942");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  using Source = UserSgprLayout::Source;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER |
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  // 4 (private_segment_buffer) + 2 (kernarg_segment_ptr) = 6 user SGPRs.
  Meta.ComputePgmRsrc2 = 6u
                         << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout Layout;
  ASSERT_TRUE(succeeded(
      UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx942", Layout)));

  // Every source occupies exactly dwordCount(source) consecutive entries.
  for (size_t I = 0; I < Layout.Entries.size();) {
    UserSgprLayout::Source Src = Layout.Entries[I].SrcKind;
    unsigned Run = 0;
    while (I + Run < Layout.Entries.size() &&
           Layout.Entries[I + Run].SrcKind == Src &&
           Layout.Entries[I + Run].SubDword == Run) {
      ++Run;
    }
    EXPECT_EQ(Run, UserSgprLayout::dwordCount(Src));
    I += Run;
  }
  EXPECT_EQ(UserSgprLayout::dwordCount(Source::PrivateSegmentBuffer), 4u);
  EXPECT_EQ(UserSgprLayout::dwordCount(Source::KernargSegmentPtr), 2u);
  EXPECT_EQ(UserSgprLayout::dwordCount(Source::WorkgroupIdX), 1u);
}

TEST(UserSgprLayout, DecodesKernargPreload) {
  Profile P("gfx1250");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  const unsigned PreloadLen = 3;
  const unsigned PreloadOffsetDwords = 2;
  Meta.KernargSegmentSize = 20;
  Meta.KernargPreload = static_cast<uint16_t>(
      (PreloadLen << KERNARG_PRELOAD_SPEC_LENGTH_SHIFT) |
      (PreloadOffsetDwords << KERNARG_PRELOAD_SPEC_OFFSET_SHIFT));
  // 2 (kernarg_segment_ptr) + 3 (preload) = 5 user SGPRs.
  Meta.ComputePgmRsrc2 = 5u << COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout Layout;
  ASSERT_TRUE(succeeded(
      UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx1250", Layout)));
  EXPECT_EQ(Layout.preloadedKernargLength(), PreloadLen);
  EXPECT_EQ(Layout.preloadedKernargByteOffset(), PreloadOffsetDwords * 4);
  ASSERT_TRUE(Layout.firstPreloadedKernargSgpr().has_value());
  EXPECT_EQ(*Layout.firstPreloadedKernargSgpr(), 2u);
  ASSERT_EQ(Layout.Entries.size(), 5u);
  EXPECT_EQ(Layout.Entries[2].SrcKind,
            UserSgprLayout::Source::PreloadedKernarg);
  EXPECT_EQ(Layout.Entries[2].KernargByteOffset, PreloadOffsetDwords * 4);
  EXPECT_EQ(Layout.Entries[4].KernargByteOffset, (PreloadOffsetDwords + 2) * 4);
}

TEST(UserSgprLayout, ReservedUserSgprsRemainUnset) {
  Profile P("gfx942");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  Meta.ComputePgmRsrc2 =
      (5u << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT) |
      COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X;

  UserSgprLayout Layout;
  ASSERT_TRUE(succeeded(
      UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx942", Layout)));
  EXPECT_EQ(Layout.UserSgprCount, 5u);
  ASSERT_EQ(Layout.Entries.size(), 6u);
  EXPECT_EQ(Layout.Entries[2].SrcKind, UserSgprLayout::Source::Unset);
  EXPECT_EQ(Layout.Entries[4].SrcKind, UserSgprLayout::Source::Unset);
  ASSERT_TRUE(Layout.workgroupIdXSgpr().has_value());
  EXPECT_EQ(*Layout.workgroupIdXSgpr(), 5u);
}

TEST(UserSgprLayout, TooFewUserSgprsAreRefused) {
  Profile P("gfx942");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  Meta.ComputePgmRsrc2 = 1u
                         << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout Layout;
  Error E = UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx942", Layout);
  EXPECT_EQ(reasonOf(std::move(E)), RaiseFailureReason::UserSgprLayoutMismatch);
}

TEST(UserSgprLayout, KernargPreloadOutsideSegmentIsRefused) {
  Profile P("gfx1250");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernargSegmentSize = 4;
  Meta.KernargPreload =
      static_cast<uint16_t>((1u << KERNARG_PRELOAD_SPEC_LENGTH_SHIFT) |
                            (1u << KERNARG_PRELOAD_SPEC_OFFSET_SHIFT));
  Meta.ComputePgmRsrc2 = 1u << COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout Layout;
  Error E = UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx1250", Layout);
  EXPECT_EQ(reasonOf(std::move(E)), RaiseFailureReason::UserSgprLayoutMismatch);
}

TEST(UserSgprLayout, ArchitectedWorkgroupIdsAreNotSequentialSgprs) {
  Profile P("gfx1250");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.ComputePgmRsrc2 = COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X |
                         COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y |
                         COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z |
                         COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO;

  UserSgprLayout Layout;
  ASSERT_TRUE(succeeded(
      UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx1250", Layout)));
  EXPECT_FALSE(Layout.workgroupIdXSgpr().has_value());
  EXPECT_FALSE(Layout.workgroupIdYSgpr().has_value());
  EXPECT_FALSE(Layout.workgroupIdZSgpr().has_value());
  ASSERT_TRUE(Layout.workgroupInfoSgpr().has_value());
  EXPECT_EQ(*Layout.workgroupInfoSgpr(), 0u);
  ASSERT_EQ(Layout.Entries.size(), 1u);
  EXPECT_EQ(Layout.Entries[0].SrcKind, UserSgprLayout::Source::WorkgroupInfo);
}

TEST(UserSgprLayout, UserSgprCountFieldWidthIsIsaVersioned) {
  Profile Gfx942("gfx942");
  Profile Gfx1250("gfx1250");
  ASSERT_TRUE(Gfx942.ok());
  ASSERT_TRUE(Gfx1250.ok());

  using namespace llvm::amdhsa;
  // Two pointer SGPRs and 30 preloaded dwords exercise count 32, which the
  // gfx942 5-bit field decodes as zero.
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  Meta.KernargPreload =
      static_cast<uint16_t>(30u << KERNARG_PRELOAD_SPEC_LENGTH_SHIFT);
  Meta.KernargSegmentSize = 30 * 4;
  Meta.ComputePgmRsrc2 = 32u << COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout OnGfx1250;
  EXPECT_TRUE(succeeded(UserSgprLayout::tryFromKernelMeta(
      Meta, Gfx1250.get(), "gfx1250", OnGfx1250)));
  EXPECT_EQ(OnGfx1250.UserSgprCount, 32u);

  UserSgprLayout OnGfx942;
  Error E =
      UserSgprLayout::tryFromKernelMeta(Meta, Gfx942.get(), "gfx942", OnGfx942);
  EXPECT_EQ(reasonOf(std::move(E)), RaiseFailureReason::UserSgprLayoutMismatch);
}

TEST(UserSgprLayout, ExcessiveUserSgprCountIsRefused) {
  Profile Gfx942("gfx942");
  Profile Gfx1250("gfx1250");
  ASSERT_TRUE(Gfx942.ok());
  ASSERT_TRUE(Gfx1250.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  UserSgprLayout Layout;

  Meta.ComputePgmRsrc2 = 17u
                         << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT;
  Error Gfx942Error =
      UserSgprLayout::tryFromKernelMeta(Meta, Gfx942.get(), "gfx942", Layout);
  EXPECT_EQ(reasonOf(std::move(Gfx942Error)),
            RaiseFailureReason::UserSgprLayoutMismatch);

  Meta.ComputePgmRsrc2 = 33u << COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT_SHIFT;
  Error Gfx1250Error =
      UserSgprLayout::tryFromKernelMeta(Meta, Gfx1250.get(), "gfx1250", Layout);
  EXPECT_EQ(reasonOf(std::move(Gfx1250Error)),
            RaiseFailureReason::UserSgprLayoutMismatch);
}

TEST(UserSgprLayout, KernargPreloadOnUnsupportedIsaIsRefused) {
  Profile P("gfx900");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernargSegmentSize = 4;
  Meta.KernargPreload =
      static_cast<uint16_t>(1u << KERNARG_PRELOAD_SPEC_LENGTH_SHIFT);
  Meta.ComputePgmRsrc2 = 1u
                         << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout Layout;
  Error E = UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx900", Layout);
  EXPECT_EQ(reasonOf(std::move(E)), RaiseFailureReason::UserSgprLayoutMismatch);
}

TEST(UserSgprLayout, PrintSummarizesEntries) {
  Profile P("gfx942");
  ASSERT_TRUE(P.ok());

  using namespace llvm::amdhsa;
  KernelMeta Meta;
  Meta.Name = "k";
  Meta.KernelCodeProperties =
      KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  Meta.ComputePgmRsrc2 = 2u
                         << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT;

  UserSgprLayout Layout;
  ASSERT_TRUE(succeeded(
      UserSgprLayout::tryFromKernelMeta(Meta, P.get(), "gfx942", Layout)));
  std::string S;
  raw_string_ostream OS(S);
  Layout.print(OS);
  EXPECT_NE(S.find("user_sgpr_count=2"), std::string::npos);
  EXPECT_NE(S.find("s[0]=KernargSegmentPtr"), std::string::npos);
}

} // namespace
