//===- KernargAbiLayoutTest.cpp - Hotswap kernarg ABI layout tests --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Covers the source kernel-argument ABI reconstruction the raiser seeds entry
// registers from: the user-SGPR layout derived from the kernel descriptor, the
// hidden-argument byte classification, and the IR synthesis of source hidden
// argument values.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/kernarg-layout.h"
#include "hotswap/raiser/raise_failure.h"
#include "hotswap/raiser/source-hidden-args.h"
#include "hotswap/raiser/user-sgpr-layout.h"

#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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

KernelArgMeta arg(StringRef ValueKind, uint32_t Offset, uint32_t Size) {
  KernelArgMeta A;
  A.ValueKind = ValueKind.str();
  A.Offset = Offset;
  A.Size = Size;
  return A;
}

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

// A module with one entry-block-positioned builder, for the IR-emitting
// hidden-argument helpers.
class HiddenArgHarness {
public:
  HiddenArgHarness() : M("kernarg-abi-test", C), B(C) {
    Fn = Function::Create(FunctionType::get(Type::getVoidTy(C), false),
                          GlobalValue::ExternalLinkage, "k", M);
    B.SetInsertPoint(BasicBlock::Create(C, "entry", Fn));
  }

  SourceHiddenArgContext context(ArrayRef<KernelArgMeta> Args,
                                 unsigned ScaledReplicationFactor = 1,
                                 bool AssumeHipGlobalOffsetZero = false) {
    return SourceHiddenArgContext{C,
                                  M,
                                  B,
                                  *Fn,
                                  B.getInt8Ty(),
                                  B.getInt32Ty(),
                                  B.getInt64Ty(),
                                  Args,
                                  AssumeHipGlobalOffsetZero,
                                  /*TargetCodeObjectVersion=*/6,
                                  ScaledReplicationFactor};
  }

  std::string dump() const {
    std::string S;
    raw_string_ostream OS(S);
    Fn->print(OS);
    return S;
  }

private:
  LLVMContext C;
  Module M;

public:
  IRBuilder<> B;
  Function *Fn = nullptr;
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

TEST(ClassifySourceHiddenArgByte, MatchesHiddenArgByteIndex) {
  std::vector<KernelArgMeta> Args = {arg("hidden_block_count_x", 8, 4)};
  std::optional<SourceHiddenArgByte> B = classifySourceHiddenArgByte(Args, 11);
  ASSERT_TRUE(B.has_value());
  EXPECT_EQ(B->Kind, SourceHiddenArgKind::HiddenBlockCountX);
  EXPECT_EQ(B->ArgOffset, 8u);
  EXPECT_EQ(B->ByteOffset, 11u);
  EXPECT_EQ(B->byteIndexInArg(), 3u);
}

TEST(ClassifySourceHiddenArgByte, RegularArgIsNotAMatch) {
  std::vector<KernelArgMeta> Args = {arg("by_value", 0, 8)};
  EXPECT_FALSE(classifySourceHiddenArgByte(Args, 0).has_value());
}

TEST(ClassifySourceHiddenArgByte, UnknownHiddenKindIsUnsupported) {
  std::vector<KernelArgMeta> Args = {arg("hidden_something_new", 0, 4)};
  std::optional<SourceHiddenArgByte> B = classifySourceHiddenArgByte(Args, 0);
  ASSERT_TRUE(B.has_value());
  EXPECT_EQ(B->Kind, SourceHiddenArgKind::UnsupportedHidden);
}

TEST(ClassifySourceHiddenArgByte, OffsetPastAllArgsIsNotAMatch) {
  std::vector<KernelArgMeta> Args = {arg("hidden_block_count_x", 0, 4)};
  EXPECT_FALSE(classifySourceHiddenArgByte(Args, 64).has_value());
}

TEST(EmitSourceHidden, BlockCountSynthesizesDispatchDivision) {
  HiddenArgHarness H;
  std::vector<KernelArgMeta> Args = {arg("hidden_block_count_x", 0, 4)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V = emitSourceHiddenDword(Ctx, 0);
  ASSERT_TRUE(static_cast<bool>(V));
  EXPECT_NE(*V, nullptr);
  // block_count = grid_size / group_size, backed by dispatch-packet reads.
  std::string IR = H.dump();
  EXPECT_NE(IR.find("udiv"), std::string::npos);
  EXPECT_NE(IR.find("dispatch.ptr"), std::string::npos);
}

TEST(EmitSourceHidden, RegularArgReturnsNullWithoutError) {
  HiddenArgHarness H;
  std::vector<KernelArgMeta> Args = {arg("global_buffer", 0, 8)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V = emitSourceHiddenDword(Ctx, 0);
  ASSERT_TRUE(static_cast<bool>(V));
  EXPECT_EQ(*V, nullptr);
}

TEST(EmitSourceHidden, UnsupportedHiddenKindIsAnError) {
  HiddenArgHarness H;
  std::vector<KernelArgMeta> Args = {arg("hidden_private_base", 0, 4)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V = emitSourceHiddenDword(Ctx, 0);
  ASSERT_FALSE(static_cast<bool>(V));
  EXPECT_EQ(reasonOf(V.takeError()),
            RaiseFailureReason::UnsupportedSourceHiddenArg);
}

TEST(EmitSourceHidden, InvalidHiddenArgSizeIsAnError) {
  HiddenArgHarness H;
  std::vector<KernelArgMeta> Args = {arg("hidden_block_count_x", 0, 8)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V = emitSourceHiddenDword(Ctx, 0);
  ASSERT_FALSE(static_cast<bool>(V));
  EXPECT_EQ(reasonOf(V.takeError()),
            RaiseFailureReason::UnsupportedSourceHiddenArg);
}

TEST(EmitSourceHidden, ZeroSizedHiddenArgIsAnError) {
  HiddenArgHarness H;
  std::vector<KernelArgMeta> Args = {arg("hidden_block_count_x", 0, 0)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V = emitSourceHiddenDword(Ctx, 0);
  ASSERT_FALSE(static_cast<bool>(V));
  EXPECT_EQ(reasonOf(V.takeError()),
            RaiseFailureReason::UnsupportedSourceHiddenArg);
}

TEST(EmitSourceHidden, GlobalOffsetNeedsHipZeroAssumption) {
  std::vector<KernelArgMeta> Args = {arg("hidden_global_offset_x", 0, 8)};

  HiddenArgHarness Reject;
  SourceHiddenArgContext RejectCtx = Reject.context(Args);
  Expected<Value *> Rejected = emitSourceHiddenDword(RejectCtx, 0);
  ASSERT_FALSE(static_cast<bool>(Rejected));
  EXPECT_EQ(reasonOf(Rejected.takeError()),
            RaiseFailureReason::UnsupportedSourceHiddenArg);

  HiddenArgHarness Accept;
  SourceHiddenArgContext AcceptCtx =
      Accept.context(Args, /*ScaledReplicationFactor=*/1,
                     /*AssumeHipGlobalOffsetZero=*/true);
  Expected<Value *> Accepted = emitSourceHiddenDword(AcceptCtx, 0);
  ASSERT_TRUE(static_cast<bool>(Accepted));
  EXPECT_NE(*Accepted, nullptr);
}

TEST(EmitSourceHidden, ScaledReplicationDescalesOnlyXGroupSize) {
  std::vector<KernelArgMeta> ArgsX = {arg("hidden_group_size_x", 0, 2)};
  std::vector<KernelArgMeta> ArgsY = {arg("hidden_group_size_y", 0, 2)};

  HiddenArgHarness AlongX;
  SourceHiddenArgContext CtxX =
      AlongX.context(ArgsX, /*ScaledReplicationFactor=*/2);
  ASSERT_TRUE(static_cast<bool>(
      emitSourceHiddenInteger(CtxX, 0, /*ByteWidth=*/2, /*IsSigned=*/false)));
  EXPECT_NE(AlongX.dump().find("_descaled"), std::string::npos);

  HiddenArgHarness AlongY;
  SourceHiddenArgContext CtxY =
      AlongY.context(ArgsY, /*ScaledReplicationFactor=*/2);
  ASSERT_TRUE(static_cast<bool>(
      emitSourceHiddenInteger(CtxY, 0, /*ByteWidth=*/2, /*IsSigned=*/false)));
  EXPECT_EQ(AlongY.dump().find("_descaled"), std::string::npos);
}

TEST(EmitSourceHidden, UnsupportedIntegerWidthIsAnError) {
  HiddenArgHarness H;
  std::vector<KernelArgMeta> Args = {arg("hidden_block_count_x", 0, 4)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V =
      emitSourceHiddenInteger(Ctx, 0, /*ByteWidth=*/3, /*IsSigned=*/false);
  ASSERT_FALSE(static_cast<bool>(V));
  EXPECT_EQ(reasonOf(V.takeError()),
            RaiseFailureReason::UnsupportedSourceHiddenArg);
}

TEST(EmitSourceHidden, DwordSpanningNonHiddenByteIsAnError) {
  HiddenArgHarness H;
  // A 2-byte hidden field followed by a regular arg: a 4-byte read at offset 0
  // starts hidden but runs into non-hidden memory.
  std::vector<KernelArgMeta> Args = {arg("hidden_grid_dims", 0, 2),
                                     arg("by_value", 2, 4)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V = emitSourceHiddenDword(Ctx, 0);
  ASSERT_FALSE(static_cast<bool>(V));
  EXPECT_EQ(reasonOf(V.takeError()),
            RaiseFailureReason::UnsupportedSourceHiddenArg);
}

TEST(EmitSourceHidden, DwordStartingBeforeHiddenArgIsAnError) {
  HiddenArgHarness H;
  std::vector<KernelArgMeta> Args = {arg("by_value", 0, 2),
                                     arg("hidden_grid_dims", 2, 2)};
  SourceHiddenArgContext Ctx = H.context(Args);

  Expected<Value *> V = emitSourceHiddenDword(Ctx, 0);
  ASSERT_FALSE(static_cast<bool>(V));
  EXPECT_EQ(reasonOf(V.takeError()),
            RaiseFailureReason::UnsupportedSourceHiddenArg);
}

} // namespace
