//===- source_hidden_args_test.cpp - source_hidden_args unit tests --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/source-hidden-args.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace llvm;
using COMGR::hotswap::KernelArgMeta;
using COMGR::hotswap::SourceHiddenArgContext;
using COMGR::hotswap::SourceHiddenArgValue;
using COMGR::hotswap::emitSourceHiddenInteger;

namespace {

KernelArgMeta makeArg(const char *Name, int Offset, int Size,
                      const char *ValueKind) {
  KernelArgMeta Arg;
  Arg.Name = Name;
  Arg.Offset = Offset;
  Arg.Size = Size;
  Arg.ValueKind = ValueKind;
  return Arg;
}

struct HiddenArgModule {
  LLVMContext C;
  Module M{"source-hidden-args-test", C};
  IRBuilder<> B{C};
  Function *F = nullptr;

  HiddenArgModule() {
    auto *FTy = FunctionType::get(Type::getVoidTy(C), {}, false);
    F = Function::Create(FTy, GlobalValue::ExternalLinkage, "kernel", M);
    F->setCallingConv(CallingConv::AMDGPU_KERNEL);
    BasicBlock *BB = BasicBlock::Create(C, "entry", F);
    B.SetInsertPoint(BB);
  }

  std::string str() {
    B.CreateRetVoid();
    std::string Out;
    raw_string_ostream OS(Out);
    M.print(OS, nullptr);
    return OS.str();
  }

  SourceHiddenArgContext context(ArrayRef<KernelArgMeta> Args,
                                 bool AssumeHipGlobalOffsetZero = false,
                                 unsigned TargetCodeObjectVersion = 6) {
    return SourceHiddenArgContext{C,
                                  M,
                                  B,
                                  Type::getInt8Ty(C),
                                  Type::getInt32Ty(C),
                                  Type::getInt64Ty(C),
                                  Args,
                                  AssumeHipGlobalOffsetZero,
                                  TargetCodeObjectVersion};
  }
};

} // namespace

TEST(SourceHiddenArgs, GroupSizeXUsesAqlDispatchPacketOffset) {
  std::vector<KernelArgMeta> Args = {
      makeArg("group_x", 44, 2, "hidden_group_size_x"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/44, /*ByteWidth=*/2,
                              /*IsSigned=*/false);

  ASSERT_TRUE(Value.Matched);
  ASSERT_NE(Value.Value, nullptr);
  EXPECT_TRUE(Value.FailureDetail.empty());

  std::string IR = HM.str();
  EXPECT_NE(IR.find("@llvm.amdgcn.dispatch.ptr"), std::string::npos);
  EXPECT_NE(IR.find("getelementptr inbounds i8, ptr addrspace(4) %dispatch_ptr, i32 4"),
            std::string::npos);
  EXPECT_EQ(IR.find("i32 24"), std::string::npos)
      << "SI::KernelInputOffsets::LOCAL_SIZE_X is not the AQL packet offset";
}

TEST(SourceHiddenArgs, BlockCountXUsesGridDividedByWorkgroupSize) {
  std::vector<KernelArgMeta> Args = {
      makeArg("blocks_x", 32, 4, "hidden_block_count_x"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/32, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  ASSERT_TRUE(Value.Matched);
  ASSERT_NE(Value.Value, nullptr);

  std::string IR = HM.str();
  EXPECT_NE(IR.find("i32 4"), std::string::npos);
  EXPECT_NE(IR.find("i32 12"), std::string::npos);
  EXPECT_NE(IR.find("udiv i32"), std::string::npos);
}

TEST(SourceHiddenArgs, GridDimsUsesAqlDispatchPacketSetupField) {
  std::vector<KernelArgMeta> Args = {
      makeArg("grid_dims", 96, 2, "hidden_grid_dims"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/96, /*ByteWidth=*/2,
                              /*IsSigned=*/false);

  ASSERT_TRUE(Value.Matched);
  ASSERT_NE(Value.Value, nullptr);

  std::string IR = HM.str();
  EXPECT_NE(
      IR.find(
          "getelementptr inbounds i8, ptr addrspace(4) %dispatch_ptr, i32 2"),
      std::string::npos);
  EXPECT_NE(IR.find("and i32"), std::string::npos);
  EXPECT_NE(IR.find("3"), std::string::npos);
  EXPECT_EQ(IR.find("i32 16"), std::string::npos)
      << "grid_dims must not be derived from grid_size_y extent";
  EXPECT_EQ(IR.find("i32 20"), std::string::npos)
      << "grid_dims must not be derived from grid_size_z extent";
}

TEST(SourceHiddenArgs, GlobalOffsetXIsConstantZero) {
  std::vector<KernelArgMeta> Args = {
      makeArg("global_offset_x", 72, 8, "hidden_global_offset_x"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx =
      HM.context(Args, /*AssumeHipGlobalOffsetZero=*/true);

  SourceHiddenArgValue Low =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/72, /*ByteWidth=*/4,
                              /*IsSigned=*/false);
  SourceHiddenArgValue High =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/76, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  ASSERT_TRUE(Low.Matched);
  ConstantInt *LowCI = dyn_cast<ConstantInt>(Low.Value);
  ASSERT_NE(LowCI, nullptr);
  EXPECT_TRUE(LowCI->isZero());

  ASSERT_TRUE(High.Matched);
  ConstantInt *HighCI = dyn_cast<ConstantInt>(High.Value);
  ASSERT_NE(HighCI, nullptr);
  EXPECT_TRUE(HighCI->isZero());

  std::string IR = HM.str();
  EXPECT_EQ(IR.find("@llvm.amdgcn.dispatch.ptr"), std::string::npos);
}

TEST(SourceHiddenArgs, GlobalOffsetXRefusesWithoutHipLaunchAssumption) {
  std::vector<KernelArgMeta> Args = {
      makeArg("global_offset_x", 72, 8, "hidden_global_offset_x"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/72, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  EXPECT_TRUE(Value.Matched);
  EXPECT_EQ(Value.Value, nullptr);
  EXPECT_NE(Value.FailureDetail.find("unsupported source hidden argument kind"),
            std::string::npos);
}

TEST(SourceHiddenArgs, UnsupportedHiddenArgFailsLoudly) {
  std::vector<KernelArgMeta> Args = {
      makeArg("printf", 64, 8, "hidden_printf_buffer"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/64, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  EXPECT_TRUE(Value.Matched);
  EXPECT_EQ(Value.Value, nullptr);
  EXPECT_NE(Value.FailureDetail.find("unsupported source hidden argument kind"),
            std::string::npos);
}

TEST(SourceHiddenArgs, PrivateAndSharedBaseRefuseWithoutApertureProof) {
  std::vector<KernelArgMeta> Args = {
      makeArg("private_base", 64, 4, "hidden_private_base"),
      makeArg("shared_base", 68, 4, "hidden_shared_base"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Private =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/64, /*ByteWidth=*/4,
                              /*IsSigned=*/false);
  SourceHiddenArgValue Shared =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/68, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  EXPECT_TRUE(Private.Matched);
  EXPECT_EQ(Private.Value, nullptr);
  EXPECT_NE(Private.FailureDetail.find("hidden_private_base"),
            std::string::npos);

  EXPECT_TRUE(Shared.Matched);
  EXPECT_EQ(Shared.Value, nullptr);
  EXPECT_NE(Shared.FailureDetail.find("hidden_shared_base"),
            std::string::npos);
}

TEST(SourceHiddenArgs, HostcallUsesTargetImplicitArgOffset) {
  std::vector<KernelArgMeta> Args = {
      makeArg("hostcall", 56, 8, "hidden_hostcall_buffer"),
  };
  HiddenArgModule HM;
  HM.F->addFnAttr("amdgpu-no-implicitarg-ptr");
  HM.F->addFnAttr("amdgpu-no-hostcall-ptr");
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/56, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  ASSERT_TRUE(Value.Matched);
  ASSERT_NE(Value.Value, nullptr);
  EXPECT_TRUE(Value.FailureDetail.empty());

  std::string IR = HM.str();
  EXPECT_NE(IR.find("@llvm.amdgcn.implicitarg.ptr"), std::string::npos);
  EXPECT_NE(IR.find("i32 80"), std::string::npos);
  EXPECT_EQ(IR.find("i32 56"), std::string::npos);
  EXPECT_EQ(IR.find("amdgpu-no-implicitarg-ptr"), std::string::npos);
  EXPECT_EQ(IR.find("amdgpu-no-hostcall-ptr"), std::string::npos);
}

// Pointer hidden args move between source and target ABI offsets.
TEST(SourceHiddenArgs, PointerIdentityHiddenArgsUseTargetImplicitArgOffsets) {
  struct Case {
    const char *Name;
    const char *ValueKind;
    const char *NoAttr;
    int ExpectedOffset;
  };
  const Case Cases[] = {
      {"default_queue", "hidden_default_queue", "amdgpu-no-default-queue", 104},
      {"completion_action", "hidden_completion_action",
       "amdgpu-no-completion-action", 112},
      {"multigrid", "hidden_multigrid_sync_arg",
       "amdgpu-no-multigrid-sync-arg", 88},
      {"heap", "hidden_heap_v1", "amdgpu-no-heap-ptr", 96},
  };

  for (const Case &C : Cases) {
    std::vector<KernelArgMeta> Args = {
        makeArg(C.Name, 56, 8, C.ValueKind),
    };
    HiddenArgModule HM;
    HM.F->addFnAttr("amdgpu-no-implicitarg-ptr");
    HM.F->addFnAttr(C.NoAttr);
    SourceHiddenArgContext Ctx = HM.context(Args);

    SourceHiddenArgValue Value =
        emitSourceHiddenInteger(Ctx, /*ByteOffset=*/56, /*ByteWidth=*/4,
                                /*IsSigned=*/false);

    ASSERT_TRUE(Value.Matched) << C.ValueKind;
    ASSERT_NE(Value.Value, nullptr) << C.ValueKind;
    EXPECT_TRUE(Value.FailureDetail.empty()) << C.ValueKind;

    std::string IR = HM.str();
    EXPECT_NE(IR.find("@llvm.amdgcn.implicitarg.ptr"), std::string::npos)
        << C.ValueKind;
    EXPECT_NE(IR.find("i32 " + std::to_string(C.ExpectedOffset)),
              std::string::npos)
        << C.ValueKind;
    EXPECT_EQ(IR.find("i32 56"), std::string::npos) << C.ValueKind;
    EXPECT_EQ(IR.find("amdgpu-no-implicitarg-ptr"), std::string::npos)
        << C.ValueKind;
    EXPECT_EQ(IR.find(C.NoAttr), std::string::npos) << C.ValueKind;
  }
}

// COV4 has no heap pointer field, so heap_v1 must refuse.
TEST(SourceHiddenArgs, HeapV1RefusesBeforeCodeObjectV5) {
  std::vector<KernelArgMeta> Args = {
      makeArg("heap", 56, 8, "hidden_heap_v1"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx =
      HM.context(Args, /*AssumeHipGlobalOffsetZero=*/false,
                 /*TargetCodeObjectVersion=*/4);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/56, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  EXPECT_TRUE(Value.Matched);
  EXPECT_EQ(Value.Value, nullptr);
  EXPECT_NE(Value.FailureDetail.find("hidden_heap_v1"), std::string::npos);
}

// Bad helper widths should report through SourceHiddenArgValue, not abort.
TEST(SourceHiddenArgs, UnsupportedIntegerWidthFailsWithoutFatalError) {
  std::vector<KernelArgMeta> Args = {
      makeArg("hostcall", 56, 8, "hidden_hostcall_buffer"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/56, /*ByteWidth=*/8,
                              /*IsSigned=*/false);

  EXPECT_TRUE(Value.Matched);
  EXPECT_EQ(Value.Value, nullptr);
  EXPECT_NE(Value.FailureDetail.find("unsupported source hidden integer byte width"),
            std::string::npos);
}

TEST(SourceHiddenArgs, NonHiddenOffsetDoesNotMatch) {
  std::vector<KernelArgMeta> Args = {
      makeArg("n", 24, 4, "by_value"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/24, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  EXPECT_FALSE(Value.Matched);
  EXPECT_EQ(Value.Value, nullptr);
  EXPECT_TRUE(Value.FailureDetail.empty());
}
