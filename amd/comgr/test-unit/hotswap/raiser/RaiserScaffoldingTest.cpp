//===- RaiserScaffoldingTest.cpp - Hotswap transpiler scaffolding test ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins the scaffolding contract `raiseToIR` advertises: empty text with a valid
// source ISA produces a well-formed `llvm::Module` containing one
// `AMDGPU_KERNEL` function whose body is exactly `ret void`, with the AMDGPU
// triple set. Empty text succeeds; an empty or non-AMDGPU source ISA is
// rejected with a structured `Error` carried by the returned `Expected`.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raiser.h"

#include "hotswap/raiser/raise_failure.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace COMGR::hotswap;

namespace {

KernelMeta makeKernelMeta(llvm::StringRef Name) {
  KernelMeta Meta;
  Meta.Name = Name.str();
  return Meta;
}

// Raise empty text for a gfx942 in-place kernel named "kernel".
llvm::Expected<RaiseResult> raiseEmptyKernel() {
  KernelMeta Meta = makeKernelMeta("kernel");
  return raiseToIR(/*TextBytes=*/llvm::ArrayRef<uint8_t>(), "gfx942", "kernel",
                   Meta);
}

} // namespace

TEST(RaiserScaffolding, EmptyInputProducesValidModule) {
  llvm::Expected<RaiseResult> Result = raiseEmptyKernel();
  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  ASSERT_NE(Result->Ctx, nullptr);
  ASSERT_NE(Result->Module, nullptr);

  std::string Err;
  llvm::raw_string_ostream ErrStream(Err);
  EXPECT_FALSE(llvm::verifyModule(*Result->Module, &ErrStream)) << Err;
}

TEST(RaiserScaffolding, ModuleAdvertisesAMDGPUTriple) {
  llvm::Expected<RaiseResult> Result = raiseEmptyKernel();
  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  ASSERT_NE(Result->Module, nullptr);
  EXPECT_EQ(Result->Module->getTargetTriple().str(), "amdgcn-amd-amdhsa");
}

TEST(RaiserScaffolding, KernelFunctionIsAMDGPUKernelWithRetVoid) {
  llvm::Expected<RaiseResult> Result = raiseEmptyKernel();
  ASSERT_TRUE(static_cast<bool>(Result)) << llvm::toString(Result.takeError());
  llvm::Function *Fn = Result->Module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  EXPECT_EQ(Fn->getCallingConv(), llvm::CallingConv::AMDGPU_KERNEL);
  ASSERT_EQ(Fn->size(), 1u);
  llvm::BasicBlock &Entry = Fn->getEntryBlock();
  ASSERT_FALSE(Entry.empty());
  EXPECT_TRUE(llvm::isa<llvm::ReturnInst>(Entry.getTerminator()));
}

TEST(RaiserScaffolding, EmptySourceIsaIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result =
      raiseToIR(llvm::ArrayRef<uint8_t>(), "", "kernel", Meta);

  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Msg = llvm::toString(Result.takeError());
  EXPECT_NE(Msg.find("does not name an AMDGPU GPU"), std::string::npos) << Msg;
}

TEST(RaiserScaffolding, MalformedSourceIsaIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result =
      raiseToIR(llvm::ArrayRef<uint8_t>(), "not-a-real-isa", "kernel", Meta);

  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Msg = llvm::toString(Result.takeError());
  EXPECT_NE(Msg.find("does not name an AMDGPU GPU"), std::string::npos) << Msg;
}
