//===- llvm/unittest/rocm-dgb-symbols/AsmParserTest.cpp - AsmParser tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Local.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class DIExpressionAsmParserTest : public testing::Test {
protected:
  LLVMContext Context;
  Type *Int64Ty = Type::getInt64Ty(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int16Ty = Type::getInt16Ty(Context);
  Type *Int8Ty = Type::getInt8Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  std::unique_ptr<Module> M;
  const DIExpression *Expr;

  void parseNamedDIExpression(const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, Context);
    if (!M)
      GTEST_SKIP();
    bool BrokenDebugInfo = false;
    bool HardError = verifyModule(*M, &errs(), &BrokenDebugInfo);
    if (HardError || BrokenDebugInfo)
      GTEST_SKIP();
    const NamedMDNode *N = M->getNamedMetadata("named");
    if (!N || N->getNumOperands() != 1u ||
        !isa<const DIExpression>(N->getOperand(0)))
      GTEST_SKIP();
    Expr = cast<const DIExpression>(N->getOperand(0));
  }
};

TEST_F(DIExpressionAsmParserTest, Referrer) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpReferrer(i32))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Referrer(Int32Ty)}));
}

TEST_F(DIExpressionAsmParserTest, Arg) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpArg(3, float))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Arg(3, FloatTy)}));
}

TEST_F(DIExpressionAsmParserTest, TypeObject) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpTypeObject(i32))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::TypeObject(Int32Ty)}));
}

TEST_F(DIExpressionAsmParserTest, Constant) {
  parseNamedDIExpression(
      R"(!named = !{!DIExpression(DIOpConstant(float 2.0))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  DIExprBuilder Builder{Context, *Expr->getNewElementsRef()};
  ASSERT_EQ(SmallVector<DIOp::Variant>(Builder.range()),
            SmallVector<DIOp::Variant>(
                {DIOp::Constant(ConstantFP::get(Context, APFloat(2.0f)))}));
}

TEST_F(DIExpressionAsmParserTest, Reinterpret) {
  parseNamedDIExpression(
      R"(!named = !{!DIExpression(DIOpReinterpret(i32 addrspace(5)*))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>(
                {DIOp::Reinterpret(PointerType::get(Context, 5))}));
}

TEST_F(DIExpressionAsmParserTest, BitOffset) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpBitOffset(i32))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::BitOffset(Int32Ty)}));
}

TEST_F(DIExpressionAsmParserTest, ByteOffset) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpByteOffset(i32))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::ByteOffset(Int32Ty)}));
}

TEST_F(DIExpressionAsmParserTest, Composite) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpComposite(2, i8))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Composite(2, Int8Ty)}));
}

TEST_F(DIExpressionAsmParserTest, Extend) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpExtend(2))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Extend(2)}));
}

TEST_F(DIExpressionAsmParserTest, Select) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpSelect())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Select()}));
}

TEST_F(DIExpressionAsmParserTest, AddrOf) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpAddrOf(7))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::AddrOf(7)}));
}

TEST_F(DIExpressionAsmParserTest, Deref) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpDeref(i32))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Deref(Int32Ty)}));
}

TEST_F(DIExpressionAsmParserTest, Read) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpRead())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Read()}));
}

TEST_F(DIExpressionAsmParserTest, Add) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpAdd())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Add()}));
}

TEST_F(DIExpressionAsmParserTest, Sub) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpSub())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Sub()}));
}

TEST_F(DIExpressionAsmParserTest, Mul) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpMul())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Mul()}));
}

TEST_F(DIExpressionAsmParserTest, Div) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpDiv())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Div()}));
}

TEST_F(DIExpressionAsmParserTest, LShr) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpLShr())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::LShr()}));
}

TEST_F(DIExpressionAsmParserTest, AShr) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpAShr())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::AShr()}));
}

TEST_F(DIExpressionAsmParserTest, Shl) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpShl())})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Shl()}));
}

TEST_F(DIExpressionAsmParserTest, PushLane) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpPushLane(i32))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::PushLane(Int32Ty)}));
}

TEST_F(DIExpressionAsmParserTest, Fragment) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(DIOpFragment(0, 1))})");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>({DIOp::Fragment(0, 1)}));
}

TEST_F(DIExpressionAsmParserTest, MultipleOps) {
  parseNamedDIExpression(R"(!named = !{!DIExpression(
    DIOpArg(0, i8),
    DIOpArg(1, i8),
    DIOpAdd(),
    DIOpArg(2, i8),
    DIOpComposite(2, i16),
    DIOpReinterpret(i8 addrspace(1)*)
  )}
)");
  ASSERT_TRUE(Expr->holdsNewElements());
  ASSERT_EQ(SmallVector<DIOp::Variant>(*Expr->getNewElementsRef()),
            SmallVector<DIOp::Variant>(
                {DIOp::Arg(0, Int8Ty), DIOp::Arg(1, Int8Ty), DIOp::Add(),
                 DIOp::Arg(2, Int8Ty), DIOp::Composite(2, Int16Ty),
                 DIOp::Reinterpret(PointerType::get(Int8Ty, 1))}));
}

} // end namespace
