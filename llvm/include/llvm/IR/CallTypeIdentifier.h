//===- CallTypeIdentifier.h - Call type identifiers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A call type identifier describes the signature of a function or of an
// indirect call. It is a string, and the call type ID is its 64-bit MD5 hash,
// computed like a GlobalValue GUID. Interprocedural analyses assume that an
// indirect call can reach a function only if both have the same call type ID.
// A target whose ABI lets different IR signatures call each other can use a
// less strict ID through TargetTransformInfo::getCallTypeId.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CALLTYPEIDENTIFIER_H
#define LLVM_IR_CALLTYPEIDENTIFIER_H

#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <string>

namespace llvm {

class FunctionType;

/// \returns the target-independent call type identifier for a function or an
/// indirect call with the function type \p FTy and the calling convention
/// \p CC.
///
/// Two signatures get the same identifier if they have the same calling
/// convention and structurally equal function types. Identified structure
/// types are compared by their elements, not by their names, because names
/// are local to a module. All opaque structure types are equal. Parameter and
/// return attributes are not part of the identifier.
LLVM_ABI std::string getDefaultCallTypeIdentifier(FunctionType *FTy,
                                                  CallingConv::ID CC);

/// \returns the target-independent call type ID: the MD5 hash of
/// getDefaultCallTypeIdentifier(\p FTy, \p CC).
LLVM_ABI uint64_t getDefaultCallTypeId(FunctionType *FTy, CallingConv::ID CC);

} // namespace llvm

#endif // LLVM_IR_CALLTYPEIDENTIFIER_H
