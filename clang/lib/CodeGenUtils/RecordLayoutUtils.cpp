//===--- RecordLayoutUtils.cpp - Shared record layout queries -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/RecordLayoutUtils.h"
#include "clang/Basic/TargetInfo.h"

namespace clang::CodeGenUtils {

bool isDiscreteBitFieldABI(const ASTContext &Ctx, const RecordDecl *RD) {
  return Ctx.getTargetInfo().getCXXABI().isMicrosoft() || RD->isMsStruct(Ctx);
}

// ROCgdb reconstructs the de facto AMDGPU aggregate return/argument
// convention from the DWARF type, allocating one register for a member whose
// type has no non-static data members.  Dropping such members from the IR
// record turns them into explicit padding, which the backend spreads over one
// register per byte, so the convention no longer matches what the debugger
// (or previously compiled device code) expects.  Keep the pre-existing rule
// for AMDGPU until the register assignment stops depending on record layout
// padding.
static bool useLegacyEmptyFieldLayout(const ASTContext &Ctx) {
  return Ctx.getTargetInfo().getTriple().isAMDGCN();
}

bool isEmptyFieldForLayout(const ASTContext &Ctx, const FieldDecl *FD) {
  if (useLegacyEmptyFieldLayout(Ctx))
    return FD->isZeroSize(Ctx);

  if (FD->isZeroLengthBitField())
    return true;

  if (FD->isUnnamedBitField())
    return false;

  return isEmptyRecordForLayout(Ctx, FD->getType());
}

bool isEmptyRecordForLayout(const ASTContext &Ctx, QualType T) {
  if (useLegacyEmptyFieldLayout(Ctx)) {
    const auto *CXXRD = T->getAsCXXRecordDecl();
    return CXXRD && CXXRD->isEmpty();
  }

  const auto *RD = T->getAsRecordDecl();
  if (!RD)
    return false;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (CXXRD->isDynamicClass())
      return false;

    for (const auto &I : CXXRD->bases())
      if (!isEmptyRecordForLayout(Ctx, I.getType()))
        return false;
  }

  for (const auto *I : RD->fields())
    if (!isEmptyFieldForLayout(Ctx, I))
      return false;

  return true;
}

bool isOverlappingVBaseABI(const ASTContext &Ctx) {
  return !Ctx.getTargetInfo().getCXXABI().isMicrosoft();
}

} // namespace clang::CodeGenUtils
