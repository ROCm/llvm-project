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
// function whose body is exactly `ret void`, with the AMDGPU triple set and
// the source kernarg segment declared. Empty inputs succeed; malformed ISA
// inputs are rejected with a BadInput RaiseFailure carried in the returned
// `llvm::Error`. Descriptor presence is enforced upstream by the code-object
// loader, so it is no longer a raiser precondition.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raiser.h"

#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <mutex>

// hotswap::raiser also carries the wave-projection objects, which link the
// hotswap::decoder MC stack; its initMCState calls
// COMGR::ensureLLVMInitialized, whose production definition lives in
// libamd_comgr. Provide the registration here so the test binary stays minimal
// instead of linking the full Comgr.
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

using COMGR::hotswap::KernelMeta;
using COMGR::hotswap::RaiseFailure;
using COMGR::hotswap::RaiseFailureReason;
using COMGR::hotswap::RaiseResult;
using COMGR::hotswap::raiseToIR;

namespace {

KernelMeta makeKernelMeta(llvm::StringRef Name,
                          uint32_t KernargSegmentSize = 0) {
  KernelMeta Meta;
  Meta.Name = Name.str();
  Meta.KernargSegmentSize = KernargSegmentSize;
  return Meta;
}

// The RaiseFailureReason a refused raise reports, or None if the error was not
// a RaiseFailure.
RaiseFailureReason refusalReason(llvm::Error E) {
  RaiseFailureReason Reason = RaiseFailureReason::None;
  llvm::handleAllErrors(std::move(E),
                        [&](const RaiseFailure &F) { Reason = F.reason(); });
  return Reason;
}

} // namespace

TEST(RaiserScaffolding, EmptyInputProducesValidModule) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  ASSERT_NE(Result->Ctx, nullptr);
  ASSERT_NE(Result->Module, nullptr);

  std::string Err;
  llvm::raw_string_ostream ErrStream(Err);
  EXPECT_FALSE(llvm::verifyModule(*Result->Module, &ErrStream)) << Err;
}

TEST(RaiserScaffolding, ModuleAdvertisesAMDGPUTriple) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  ASSERT_NE(Result->Module, nullptr);
  EXPECT_EQ(Result->Module->getTargetTriple().str(), "amdgcn-amd-amdhsa");
}

TEST(RaiserScaffolding, KernelFunctionIsAMDGPUKernelWithRetVoid) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  llvm::Function *Fn = Result->Module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  EXPECT_EQ(Fn->getCallingConv(), llvm::CallingConv::AMDGPU_KERNEL);
  ASSERT_EQ(Fn->size(), 1u);
  llvm::BasicBlock &Entry = Fn->getEntryBlock();
  ASSERT_FALSE(Entry.empty());
  EXPECT_TRUE(llvm::isa<llvm::ReturnInst>(Entry.getTerminator()));
}

TEST(RaiserScaffolding, KernelDeclaresSourceKernargSegment) {
  KernelMeta Meta = makeKernelMeta("kernel", /*KernargSegmentSize=*/40);
  llvm::Expected<RaiseResult> Result = raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  llvm::Function *Fn = Result->Module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  ASSERT_EQ(Fn->arg_size(), 1u);
  EXPECT_EQ(Fn->getArg(0)->getType()->getPointerAddressSpace(), 4u);
  auto *SegmentTy =
      llvm::dyn_cast_if_present<llvm::ArrayType>(Fn->getParamByRefType(0));
  ASSERT_NE(SegmentTy, nullptr);
  EXPECT_TRUE(SegmentTy->getElementType()->isIntegerTy(8));
  EXPECT_EQ(SegmentTy->getNumElements(), 40u);
  EXPECT_EQ(Fn->getParamAlign(0).valueOrOne().value(), 16u);
}

TEST(RaiserScaffolding, EmptyKernargSegmentTakesNoParameter) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  llvm::Function *Fn = Result->Module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  EXPECT_EQ(Fn->arg_size(), 0u);
}

TEST(RaiserScaffolding, KernelSuppressesTargetHiddenArguments) {
  KernelMeta Meta = makeKernelMeta("kernel", /*KernargSegmentSize=*/40);
  llvm::Expected<RaiseResult> Result = raiseToIR("gfx942", "kernel", Meta);

  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  llvm::Function *Fn = Result->Module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  EXPECT_TRUE(Fn->hasFnAttribute("amdgpu-no-implicitarg-ptr"));
}

TEST(RaiserScaffolding, EmptyTargetIsaIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseToIR("", "kernel", Meta);

  ASSERT_FALSE(static_cast<bool>(Result));
  EXPECT_EQ(refusalReason(Result.takeError()), RaiseFailureReason::BadInput);
}

TEST(RaiserScaffolding, MalformedTargetIsaIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result =
      raiseToIR("not-a-real-isa", "kernel", Meta);

  ASSERT_FALSE(static_cast<bool>(Result));
  EXPECT_EQ(refusalReason(Result.takeError()), RaiseFailureReason::BadInput);
}
