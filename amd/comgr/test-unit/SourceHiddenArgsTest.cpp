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
                                 bool AssumeHipGlobalOffsetZero = false) {
    return SourceHiddenArgContext{C,
                                  M,
                                  B,
                                  Type::getInt8Ty(C),
                                  Type::getInt32Ty(C),
                                  Type::getInt64Ty(C),
                                  Args,
                                  AssumeHipGlobalOffsetZero};
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
      makeArg("hostcall", 64, 8, "hidden_hostcall_buffer"),
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
