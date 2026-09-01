//===- RaiseFailureTest.cpp - structured raise-failure unit tests ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the RaiseFailure error payload: the three shape factories
// return an llvm::Error carrying a RaiseFailure, and these tests inspect the
// payload's reason, the optional format/offset accessors, and the stable
// reasonString / log diagnostic text.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raise_failure.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using COMGR::hotswap::RaiseFailure;
using COMGR::hotswap::RaiseFailureReason;
using COMGR::hotswap::reasonString;

namespace {

std::string toLog(const RaiseFailure &F) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  F.log(OS);
  return S;
}

// Run Check on the single RaiseFailure the factory returns, asserting the error
// is exactly one RaiseFailure.
template <typename FnT> void inspect(llvm::Error E, FnT Check) {
  ASSERT_TRUE(static_cast<bool>(E));
  unsigned Handled = 0;
  llvm::handleAllErrors(std::move(E), [&](const RaiseFailure &F) {
    ++Handled;
    Check(F);
  });
  EXPECT_EQ(Handled, 1u);
}

TEST(RaiseFailure, GeneralCarriesOnlyReasonAndDetail) {
  inspect(RaiseFailure::general(RaiseFailureReason::BadInput,
                                "source ISA string is empty"),
          [](const RaiseFailure &F) {
            EXPECT_EQ(F.reason(), RaiseFailureReason::BadInput);
            EXPECT_EQ(F.detail().str(), "source ISA string is empty");
            EXPECT_TRUE(F.mnemonic().empty());
            EXPECT_FALSE(F.offset().has_value());
            EXPECT_FALSE(F.format().has_value());
            EXPECT_EQ(toLog(F), "BadInput :: source ISA string is empty");
          });
}

TEST(RaiseFailure, AtInstructionLocatesTheInstruction) {
  inspect(RaiseFailure::atInstruction(RaiseFailureReason::UnsupportedOpcode,
                                      "v_foo_e32", 0x2c, "VALU",
                                      "operand shape not modelled"),
          [](const RaiseFailure &F) {
            EXPECT_EQ(F.reason(), RaiseFailureReason::UnsupportedOpcode);
            EXPECT_EQ(F.mnemonic().str(), "v_foo_e32");
            ASSERT_TRUE(F.offset().has_value());
            EXPECT_EQ(*F.offset(), 0x2cu);
            ASSERT_TRUE(F.format().has_value());
            EXPECT_EQ(F.format()->str(), "VALU");
            EXPECT_EQ(toLog(F),
                      "UnsupportedOpcode: v_foo_e32 [VALU] @offset=0x2c "
                      ":: operand shape not modelled");
          });
}

TEST(RaiseFailure, AtInstructionAllowsAnEmptyDetail) {
  inspect(RaiseFailure::atInstruction(RaiseFailureReason::UnsupportedOpcode,
                                      "s_bar", 0, "SOP1"),
          [](const RaiseFailure &F) {
            EXPECT_TRUE(F.detail().empty());
            EXPECT_EQ(toLog(F), "UnsupportedOpcode: s_bar [SOP1] @offset=0x0");
          });
}

TEST(RaiseFailure, InKernelFormatsTheKernelScopedMessage) {
  inspect(RaiseFailure::inKernel(RaiseFailureReason::MissingKernelDescriptor,
                                 "my_kernel", ".kd symbol not parsed"),
          [](const RaiseFailure &F) {
            EXPECT_EQ(F.reason(), RaiseFailureReason::MissingKernelDescriptor);
            EXPECT_TRUE(F.mnemonic().empty());
            EXPECT_FALSE(F.offset().has_value());
            EXPECT_FALSE(F.format().has_value());
            EXPECT_EQ(F.detail().str(),
                      "kernel 'my_kernel': .kd symbol not parsed");
          });
}

TEST(RaiseFailure, ReasonStringTokensAreStable) {
  EXPECT_EQ(reasonString(RaiseFailureReason::None), "None");
  EXPECT_EQ(reasonString(RaiseFailureReason::BadInput), "BadInput");
  EXPECT_EQ(reasonString(RaiseFailureReason::UnsupportedInstructionForm),
            "unsupported-instruction-form");
  EXPECT_EQ(reasonString(RaiseFailureReason::UnsupportedFloatingPointMode),
            "unsupported-floating-point-mode");
  EXPECT_EQ(reasonString(RaiseFailureReason::CrossWaveLaneIdLeak),
            "cross-wave-lane-id-leak");
}

} // namespace
