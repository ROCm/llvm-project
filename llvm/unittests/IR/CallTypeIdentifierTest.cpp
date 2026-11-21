//===- CallTypeIdentifierTest.cpp - Call type identifier tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/CallTypeIdentifier.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MD5.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class CallTypeIdentifierTest : public testing::Test {
protected:
  LLVMContext Ctx;
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *PtrTy = PointerType::get(Ctx, 0);

  std::string getId(Type *RetTy, ArrayRef<Type *> Params,
                    CallingConv::ID CC = CallingConv::C,
                    bool IsVarArg = false) {
    return getDefaultCallTypeIdentifier(
        FunctionType::get(RetTy, Params, IsVarArg), CC);
  }
};

TEST_F(CallTypeIdentifierTest, Encoding) {
  EXPECT_EQ(getId(VoidTy, {PtrTy, I32Ty}), "llvm.v1:cc0:f_isVoidp0i32f");

  Type *RetTy = StructType::get(Ctx, {I32Ty, ArrayType::get(FloatTy, 2)});
  Type *Named = StructType::create({I32Ty, FloatTy}, "Named");
  Type *Packed =
      StructType::create(Ctx, {I32Ty, FloatTy}, "Packed", /*isPacked=*/true);
  // An opaque structure type is not a valid parameter type, but it can be an
  // element of a structure type.
  Type *WithOpaque = StructType::get(StructType::create(Ctx, "Opaque"));
  Type *TargetTy = TargetExtType::get(Ctx, "test.ty", {FloatTy}, {1, 2});
  EXPECT_EQ(getId(RetTy,
                  {PointerType::get(Ctx, 3),
                   FixedVectorType::get(Type::getHalfTy(Ctx), 4),
                   ScalableVectorType::get(Type::getInt64Ty(Ctx), 2), Named,
                   Packed, WithOpaque, TargetTy},
                  CallingConv::Fast, /*IsVarArg=*/true),
            "llvm.v1:cc8:f_sl_i32a2f32sp3v4f16nxv2i64sl_i32f32ssp_i32f32s"
            "sl_sosttest.ty_f32_1_2tvarargf");
}

TEST_F(CallTypeIdentifierTest, Hash) {
  // The call type ID is the MD5 hash of the identifier, like a GUID.
  FunctionType *FTy = FunctionType::get(VoidTy, {PtrTy}, /*isVarArg=*/false);
  std::string Id = getDefaultCallTypeIdentifier(FTy, CallingConv::C);
  EXPECT_EQ(Id, "llvm.v1:cc0:f_isVoidp0f");
  EXPECT_EQ(getDefaultCallTypeId(FTy, CallingConv::C), MD5Hash(Id));
  EXPECT_EQ(getDefaultCallTypeId(FTy, CallingConv::C),
            UINT64_C(657395674655738670));
}

TEST_F(CallTypeIdentifierTest, StructureTypes) {
  // Identified structure types are compared by their elements.
  Type *Literal = StructType::get(Ctx, {I32Ty, FloatTy});
  Type *S1 = StructType::create({I32Ty, FloatTy}, "S");
  Type *S2 = StructType::create({I32Ty, FloatTy}, "S");
  EXPECT_NE(S1, S2);
  EXPECT_EQ(getId(VoidTy, {S1}), getId(VoidTy, {S2}));
  EXPECT_EQ(getId(VoidTy, {S1}), getId(VoidTy, {Literal}));

  // All opaque structure types are equal.
  EXPECT_EQ(getId(VoidTy, {StructType::get(StructType::create(Ctx, "O1"))}),
            getId(VoidTy, {StructType::get(StructType::create(Ctx, "O2"))}));

  // Packed and other structure types differ.
  Type *PackedLiteral =
      StructType::get(Ctx, {I32Ty, FloatTy}, /*isPacked=*/true);
  EXPECT_NE(getId(VoidTy, {PackedLiteral}), getId(VoidTy, {Literal}));

  // Nested structure types are distinguishable.
  Type *Outer1 = StructType::get(StructType::get(I32Ty), I32Ty);
  Type *Outer2 = StructType::get(StructType::get(I32Ty, I32Ty));
  EXPECT_NE(getId(VoidTy, {Outer1}), getId(VoidTy, {Outer2}));
}

TEST_F(CallTypeIdentifierTest, Differences) {
  std::string Id = getId(VoidTy, {I32Ty});
  EXPECT_EQ(getId(VoidTy, {I32Ty}), Id);
  EXPECT_NE(getId(VoidTy, {FloatTy}), Id);
  EXPECT_NE(getId(I32Ty, {I32Ty}), Id);
  EXPECT_NE(getId(VoidTy, {I32Ty, I32Ty}), Id);
  EXPECT_NE(getId(VoidTy, {I32Ty}, CallingConv::Fast), Id);
  EXPECT_NE(getId(VoidTy, {I32Ty}, CallingConv::C, /*IsVarArg=*/true), Id);
  EXPECT_NE(getId(VoidTy, {PtrTy}), getId(VoidTy, {PointerType::get(Ctx, 3)}));
}

} // namespace
