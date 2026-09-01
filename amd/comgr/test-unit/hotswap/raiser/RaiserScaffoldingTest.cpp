//===- RaiserScaffoldingTest.cpp - Hotswap transpiler scaffolding test ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins the refusals `raiseToIR` reaches before it has a kernel to raise, or on
// input no code object can express. The shape of a raised module -- its triple,
// data layout, and the signature and attributes of the lifted function -- is
// pinned by test-lit/hotswap/raiser/kernel_scaffolding.s, which reads it off
// the emitted IR. What is left here is what only the entry point can be handed:
// an ISA string the driver would have taken from the code object, and a kernel
// extent holding no code at all.
//
// These assert the `RaiseFailureReason` rather than the rendered message, which
// is the distinction the enumerators exist for: a caller buckets a refusal
// without parsing diagnostic text.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raiser.h"

#include "hotswap/raiser/raise_failure.h"

// RaiseResult owns the context and module by pointer, so destroying one needs
// both definitions even though no test here looks inside them.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

using COMGR::hotswap::KernelMeta;
using COMGR::hotswap::RaiseFailure;
using COMGR::hotswap::RaiseFailureReason;
using COMGR::hotswap::RaiseResult;
using COMGR::hotswap::raiseToIR;
using COMGR::hotswap::TextSection;

namespace {

KernelMeta makeKernelMeta(llvm::StringRef Name) {
  KernelMeta Meta;
  Meta.Name = Name.str();
  return Meta;
}

// Raise a kernel whose extent holds no code, onto an ISA other than the one it
// was compiled for.
llvm::Expected<RaiseResult> raiseEmptyText(llvm::StringRef SourceIsa,
                                           llvm::StringRef TargetIsa,
                                           const KernelMeta &Meta) {
  COMGR::hotswap::KernelRequest Kernel{"kernel", Meta, /*StartOffset=*/0,
                                       /*EndOffset=*/0};
  return raiseToIR(TextSection{}, SourceIsa, TargetIsa, Kernel);
}

// The same raise back onto the ISA the kernel was compiled for, for the cases
// the target ISA has no say in.
llvm::Expected<RaiseResult> raiseEmptyText(llvm::StringRef Isa,
                                           const KernelMeta &Meta) {
  return raiseEmptyText(Isa, Isa, Meta);
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

TEST(RaiserScaffolding, EmptySourceIsaIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseEmptyText("", Meta);

  ASSERT_FALSE(static_cast<bool>(Result));
  EXPECT_EQ(refusalReason(Result.takeError()), RaiseFailureReason::BadInput);
}

TEST(RaiserScaffolding, MalformedSourceIsaIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseEmptyText("not-a-real-isa", Meta);

  ASSERT_FALSE(static_cast<bool>(Result));
  EXPECT_EQ(refusalReason(Result.takeError()), RaiseFailureReason::BadInput);
}

TEST(RaiserScaffolding, MalformedTargetIsaIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result =
      raiseEmptyText("gfx942", "not-a-real-isa", Meta);

  ASSERT_FALSE(static_cast<bool>(Result));
  EXPECT_EQ(refusalReason(Result.takeError()), RaiseFailureReason::BadInput);
}

TEST(RaiserScaffolding, EmptyKernelExtentIsRejected) {
  KernelMeta Meta = makeKernelMeta("kernel");
  llvm::Expected<RaiseResult> Result = raiseEmptyText("gfx942", Meta);

  // An extent with nothing in it never reaches an instruction that ends the
  // program, so it is refused for the same reason a truncated one is.
  ASSERT_FALSE(static_cast<bool>(Result));
  EXPECT_EQ(refusalReason(Result.takeError()),
            RaiseFailureReason::UnterminatedKernelExtent);
}
