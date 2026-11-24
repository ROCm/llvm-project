//===- CallTypeIdentifier.cpp - Call type identifiers ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/CallTypeIdentifier.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

using namespace llvm;

/// Appends the encoding of \p Ty to \p Enc. The encoding follows the type
/// mangling of overloaded intrinsics, except for structure types: they are
/// encoded by their elements, and packed structures differ from other
/// structures.
static void appendTypeEncoding(std::string &Enc, Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::HalfTyID:
    Enc += "f16";
    return;
  case Type::BFloatTyID:
    Enc += "bf16";
    return;
  case Type::FloatTyID:
    Enc += "f32";
    return;
  case Type::DoubleTyID:
    Enc += "f64";
    return;
  case Type::X86_FP80TyID:
    Enc += "f80";
    return;
  case Type::FP128TyID:
    Enc += "f128";
    return;
  case Type::PPC_FP128TyID:
    Enc += "ppcf128";
    return;
  case Type::VoidTyID:
    Enc += "isVoid";
    return;
  case Type::LabelTyID:
    Enc += "label";
    return;
  case Type::MetadataTyID:
    Enc += "Metadata";
    return;
  case Type::X86_AMXTyID:
    Enc += "x86amx";
    return;
  case Type::TokenTyID:
    Enc += "token";
    return;
  case Type::IntegerTyID:
    Enc += "i" + utostr(cast<IntegerType>(Ty)->getBitWidth());
    return;
  case Type::ByteTyID:
    Enc += "b" + utostr(cast<ByteType>(Ty)->getBitWidth());
    return;
  case Type::FunctionTyID: {
    auto *FTy = cast<FunctionType>(Ty);
    Enc += "f_";
    appendTypeEncoding(Enc, FTy->getReturnType());
    for (Type *ParamTy : FTy->params())
      appendTypeEncoding(Enc, ParamTy);
    if (FTy->isVarArg())
      Enc += "vararg";
    // Ensure nested function types are distinguishable.
    Enc += 'f';
    return;
  }
  case Type::PointerTyID:
    Enc += "p" + utostr(cast<PointerType>(Ty)->getAddressSpace());
    return;
  case Type::TypedPointerTyID:
    // Typed pointers exist only inside some backends. As for other pointers,
    // only the address space is part of the encoding.
    Enc += "p" + utostr(cast<TypedPointerType>(Ty)->getAddressSpace());
    return;
  case Type::StructTyID: {
    auto *STy = cast<StructType>(Ty);
    // The name of an identified structure type is local to a module, so encode
    // the elements. The elements of an opaque structure type are not known.
    if (STy->isOpaque()) {
      Enc += "so";
      return;
    }
    Enc += STy->isPacked() ? "sp_" : "sl_";
    for (Type *ElemTy : STy->elements())
      appendTypeEncoding(Enc, ElemTy);
    // Ensure nested structure types are distinguishable.
    Enc += 's';
    return;
  }
  case Type::ArrayTyID:
    Enc += "a" + utostr(cast<ArrayType>(Ty)->getNumElements());
    appendTypeEncoding(Enc, cast<ArrayType>(Ty)->getElementType());
    return;
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID: {
    auto *VTy = cast<VectorType>(Ty);
    ElementCount EC = VTy->getElementCount();
    if (EC.isScalable())
      Enc += "nx";
    Enc += "v" + utostr(EC.getKnownMinValue());
    appendTypeEncoding(Enc, VTy->getElementType());
    return;
  }
  case Type::TargetExtTyID: {
    auto *TETy = cast<TargetExtType>(Ty);
    Enc += 't';
    Enc += TETy->getName();
    for (Type *ParamTy : TETy->type_params()) {
      Enc += '_';
      appendTypeEncoding(Enc, ParamTy);
    }
    for (unsigned IntParam : TETy->int_params())
      Enc += "_" + utostr(IntParam);
    // Ensure nested target extension types are distinguishable.
    Enc += 't';
    return;
  }
  }
  llvm_unreachable("unhandled type");
}

std::string llvm::getDefaultCallTypeIdentifier(FunctionType *FTy,
                                               CallingConv::ID CC) {
  // The prefix names the scheme and its version, so that identifiers of
  // different schemes never compare equal.
  std::string Id = "llvm.v1:cc" + utostr(CC) + ":";
  appendTypeEncoding(Id, FTy);
  return Id;
}

uint64_t llvm::getDefaultCallTypeId(FunctionType *FTy, CallingConv::ID CC) {
  return MD5Hash(getDefaultCallTypeIdentifier(FTy, CC));
}
