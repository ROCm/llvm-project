//===- RaiserScaffoldingTest.cpp - Hotswap transpiler scaffolding test ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins the scaffolding contract `raiseToIR` advertises: an empty input
// produces a well-formed `llvm::Module` containing one `AMDGPU_KERNEL`
// function whose body is exactly `ret void`, with the AMDGPU triple set.
// Empty inputs succeed; missing kernel descriptor / malformed ISA inputs
// are rejected with a structured failure.
//
//===----------------------------------------------------------------------===//

#include "hotswap/canonical-op.h"
#include "hotswap/decode.h"
#include "hotswap/decoded-inst.h"
#include "hotswap/pipeline.h"
#include "hotswap/raiser.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {

COMGR::hotswap::KernelMeta makeKernelMeta(llvm::StringRef Name) {
  COMGR::hotswap::KernelMeta Meta;
  Meta.Name = Name.str();
  Meta.HasKernelDescriptor = true;
  return Meta;
}

} // namespace

TEST(RaiserScaffolding, EmptyInputProducesValidModule) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<COMGR::hotswap::RaiseResult> Result =
      COMGR::hotswap::raiseToIR({}, "kernel", Meta,
                                COMGR::hotswap::RaiseOptions{/*SourceIsa=*/"gfx942"});

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  ASSERT_NE(Result->Ctx, nullptr);
  ASSERT_NE(Result->Module, nullptr);

  std::string Err;
  llvm::raw_string_ostream ErrStream(Err);
  EXPECT_FALSE(llvm::verifyModule(*Result->Module, &ErrStream)) << Err;
}

TEST(RaiserScaffolding, ModuleAdvertisesAMDGPUTriple) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<COMGR::hotswap::RaiseResult> Result =
      COMGR::hotswap::raiseToIR({}, "kernel", Meta,
                                COMGR::hotswap::RaiseOptions{/*SourceIsa=*/"gfx942"});

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  ASSERT_NE(Result->Module, nullptr);
  EXPECT_EQ(Result->Module->getTargetTriple().str(), "amdgcn-amd-amdhsa");
}

TEST(RaiserScaffolding, KernelFunctionIsAMDGPUKernelWithRetVoid) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<COMGR::hotswap::RaiseResult> Result =
      COMGR::hotswap::raiseToIR({}, "kernel", Meta,
                                COMGR::hotswap::RaiseOptions{/*SourceIsa=*/"gfx942"});

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  llvm::Function *Fn = Result->Module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  EXPECT_EQ(Fn->getCallingConv(), llvm::CallingConv::AMDGPU_KERNEL);
  ASSERT_EQ(Fn->size(), 1u);
  llvm::BasicBlock &Entry = Fn->getEntryBlock();
  ASSERT_FALSE(Entry.empty());
  EXPECT_TRUE(llvm::isa<llvm::ReturnInst>(Entry.getTerminator()));
}

TEST(RaiserScaffolding, MissingKernelDescriptorIsRejected) {
  COMGR::hotswap::KernelMeta Meta;
  Meta.Name = "kernel";
  Meta.HasKernelDescriptor = false;
  llvm::Expected<COMGR::hotswap::RaiseResult> Result =
      COMGR::hotswap::raiseToIR({}, "kernel", Meta,
                                COMGR::hotswap::RaiseOptions{/*SourceIsa=*/"gfx942"});

  ASSERT_FALSE(static_cast<bool>(Result));
}

TEST(RaiserScaffolding, EmptyTargetIsaIsRejected) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<COMGR::hotswap::RaiseResult> Result =
      COMGR::hotswap::raiseToIR({}, "kernel", Meta,
                                COMGR::hotswap::RaiseOptions{});

  ASSERT_FALSE(static_cast<bool>(Result));
}

TEST(RaiserScaffolding, MalformedTargetIsaIsRejected) {
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<COMGR::hotswap::RaiseResult> Result =
      COMGR::hotswap::raiseToIR({}, "kernel", Meta,
                                COMGR::hotswap::RaiseOptions{/*SourceIsa=*/"not-a-real-isa"});

  ASSERT_FALSE(static_cast<bool>(Result));
}

TEST(RaiserScaffolding, PreloadedHiddenArgIsAnOrdinaryKernargLoad) {
  // A hardware-preloaded kernarg dword that happens to be a hidden argument
  // needs no synthesis: the target runtime populates hidden arguments at the
  // source byte offsets, so the seed is the same ordinary kernarg load used
  // for an explicit argument. This used to refuse unless the caller asserted
  // HIP launch semantics for hidden_global_offset_*.
  COMGR::hotswap::KernelMeta Meta = makeKernelMeta("kernel");
  Meta.Args.push_back({"global_offset_x", 72, 8, "hidden_global_offset_x", 0});
  Meta.KernargSegmentSize = 328;
  Meta.KernelCodeProperties =
      1u << llvm::amdhsa::
          KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT;
  Meta.KernargPreload =
      (1u << llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH_SHIFT) |
      (18u << llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET_SHIFT);
  Meta.ComputePgmRsrc2 =
      3u << llvm::amdhsa::COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT_SHIFT;

  COMGR::hotswap::ScopedStrictMode StrictMode(/*enabled=*/true);
  COMGR::hotswap::RaiseOptions Options;
  Options.SourceIsa = "gfx1250";
  Options.CompilationTargetIsa = "gfx942";
  llvm::Expected<COMGR::hotswap::RaiseResult> Result =
      COMGR::hotswap::raiseToIR({}, "kernel", Meta, Options);

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  ASSERT_NE(Result->Module, nullptr);

  std::string IR;
  llvm::raw_string_ostream IRStream(IR);
  Result->Module->print(IRStream, nullptr);

  EXPECT_NE(IR.find("call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr"),
            std::string::npos);
  EXPECT_EQ(IR.find("call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr"),
            std::string::npos);
}

// s_add_pc_i64's successor is Offset + Size + displacement (byte units), not
// the SOPP dword-scaled target.
TEST(DecodeBlockSuccessors, AddPcI64UsesByteOffsetAndIgnoresLowBits) {
  COMGR::hotswap::DecodedInst Di;
  Di.CanonOp = COMGR::hotswap::CanonicalOp::S_ADD_PC_I64;
  Di.IsBranch = true; // S_ADD_PC_I64 sets the AMDGPU isBranch bit
  Di.Offset = 8;
  Di.Size = 4;

  EXPECT_TRUE(COMGR::hotswap::decodedInstEndsBlock(Di));

  auto SuccessorForImm = [&](int64_t Imm) {
    Di.Inst = llvm::MCInst();
    Di.Inst.addOperand(llvm::MCOperand::createImm(Imm));
    return llvm::cantFail(COMGR::hotswap::computeDecodedBlockSuccessors(
        Di, /*NextBlockOffset=*/12));
  };

  // 8 + 4 + 8 = 20 (byte); SOPP scaling would give 8 + 4 + 8*4 = 44.
  llvm::SmallVector<uint64_t> Succ = SuccessorForImm(8);

  ASSERT_EQ(Succ.size(), 1u);
  EXPECT_EQ(Succ[0], 20u);

  // The ISA ignores the low two literal bits: +11 behaves as +8 and -1 as -4.
  Succ = SuccessorForImm(11);
  ASSERT_EQ(Succ.size(), 1u);
  EXPECT_EQ(Succ[0], 20u);

  Succ = SuccessorForImm(-1);
  ASSERT_EQ(Succ.size(), 1u);
  EXPECT_EQ(Succ[0], 8u);
}
