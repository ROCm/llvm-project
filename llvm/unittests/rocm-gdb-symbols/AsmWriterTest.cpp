//===- llvm/unittest/rocm-gdb-symbols/AsmWriter.cpp - AsmWriter tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class DIExpressionAsmWriterTest : public testing::Test {
public:
  DIExpressionAsmWriterTest() : Builder(Context), OS(S) {}

protected:
  LLVMContext Context;
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int64Ty = Type::getInt64Ty(Context);
  DIExprBuilder Builder;
  std::string S;
  raw_string_ostream OS;
};

TEST_F(DIExpressionAsmWriterTest, Empty) {
  DIExpression *Expr = Builder.intoExpression();
  EXPECT_FALSE(Expr->isValid());
}

TEST_F(DIExpressionAsmWriterTest, Referrer) {
  Builder.append<DIOp::Referrer>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpReferrer(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Arg) {
  Builder.append<DIOp::Arg>(1, Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpArg(1, i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, TypeObject) {
  Builder.append<DIOp::TypeObject>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpTypeObject(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Constant) {
  Builder
      .append<DIOp::Constant>(
          static_cast<ConstantData *>(ConstantInt::get(Int32Ty, 1)))
      .intoExpression()
      ->print(OS);
  EXPECT_EQ("!DIExpression(DIOpConstant(i32 1))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Convert) {
  Builder.append<DIOp::Convert>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpConvert(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Reinterpret) {
  Builder.append<DIOp::Reinterpret>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpReinterpret(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, BitOffset) {
  Builder.append<DIOp::BitOffset>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpBitOffset(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, ByteOffset) {
  Builder.append<DIOp::ByteOffset>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpByteOffset(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Composite) {
  Builder.append<DIOp::Composite>(2, Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpComposite(2, i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Extend) {
  Builder.append<DIOp::Extend>(2).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpExtend(2))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Select) {
  Builder.append<DIOp::Select>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpSelect())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, AddrOf) {
  Builder.append<DIOp::AddrOf>(5).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpAddrOf(5))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Deref) {
  Builder.append<DIOp::Deref>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpDeref(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Read) {
  Builder.append<DIOp::Read>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpRead())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Add) {
  Builder.append<DIOp::Add>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpAdd())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Sub) {
  Builder.append<DIOp::Sub>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpSub())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Mul) {
  Builder.append<DIOp::Mul>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpMul())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Div) {
  Builder.append<DIOp::Div>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpDiv())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, LShr) {
  Builder.append<DIOp::LShr>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpLShr())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, AShr) {
  Builder.append<DIOp::AShr>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpAShr())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, Shl) {
  Builder.append<DIOp::Shl>().intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpShl())", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, PushLane) {
  Builder.append<DIOp::PushLane>(Int64Ty).intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpPushLane(i64))", OS.str());
}

TEST_F(DIExpressionAsmWriterTest, MultipleOps) {
  Builder.insert(Builder.begin(),
                 {DIOp::Variant{std::in_place_type<DIOp::Referrer>, Int32Ty},
                  DIOp::Variant{std::in_place_type<DIOp::Referrer>, Int64Ty},
                  DIOp::Variant{std::in_place_type<DIOp::Add>}});
  Builder.intoExpression()->print(OS);
  EXPECT_EQ("!DIExpression(DIOpReferrer(i32), DIOpReferrer(i64), DIOpAdd())",
            OS.str());
}

} // namespace
