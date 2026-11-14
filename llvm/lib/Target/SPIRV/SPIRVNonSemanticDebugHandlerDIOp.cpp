//===-- SPIRVNonSemanticDebugHandlerDIOp.cpp - DIOp expressions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVGlobalRegistry.h"
#include "SPIRVNonSemanticDebugHandler.h"
#include "SPIRVSubtarget.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

constexpr uint32_t DIOpOpcodeBase = 10001;

// SPIR-V id of the type registered under \p Handle. The registry is per
// function, so each function is tried.
static MCRegister findTypeInFunction(SPIRV::IRHandle Handle,
                                     SPIRV::ModuleAnalysisInfo &MAI) {
  for (const auto &[MF, Aliases] : MAI.RegisterAliasTable) {
    SPIRVGlobalRegistry *GR =
        MF->getSubtarget<SPIRVSubtarget>().getSPIRVGlobalRegistry();
    Register VReg = GR->find(Handle, MF);
    if (!VReg.isValid())
      continue;
    auto It = Aliases.find(VReg);
    if (It != Aliases.end())
      return It->second;
  }
  return MCRegister();
}

// SPIR-V id of the type the module declares for \p Ty. No type is emitted
// here: HIP DIOps only name the types of local variables, and the module
// already declares those for the variables themselves.
static MCRegister findExistingType(const Type *Ty,
                                   SPIRV::ModuleAnalysisInfo &MAI) {
  // Untyped pointers are registered by address space, not by LLVM type.
  if (Ty->isPointerTy())
    return findTypeInFunction(
        SPIRV::irhandle_untyped_pointer(Ty->getPointerAddressSpace()), MAI);
  return findTypeInFunction(SPIRV::handle(Ty), MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDIOp(
    const DIOp::Variant &Op, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, SPIRV::ModuleAnalysisInfo &MAI) {
  SmallVector<MCRegister, 3> Operands;
  Operands.push_back(emitOpConstantI32(DIOpOpcodeBase + DIOp::getBitcodeID(Op),
                                       I32TypeReg, MAI));

  auto Add = [&](auto Value) -> bool {
    using T = std::decay_t<decltype(Value)>;
    MCRegister Reg;
    if constexpr (std::is_same_v<T, Type *>) {
      Reg = findExistingType(Value, MAI);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      Reg = emitOpConstantI32(Value, I32TypeReg, MAI);
    } else {
      // Only an i32 literal is emitted. Any other constant is dropped.
      auto *CI = dyn_cast<ConstantInt>(Value);
      if (CI && CI->getType()->isIntegerTy(32))
        Reg = emitOpConstantI32(CI->getZExtValue(), I32TypeReg, MAI);
    }
    if (!Reg.isValid())
      return false;
    Operands.push_back(Reg);
    return true;
  };

  bool Ok = true;
  switch (DIOp::getBitcodeID(Op)) {
#define HANDLE_OP1(Name, OpType, OpName)                                       \
  case DIOp::Name::getBitcodeID():                                             \
    Ok = Add(std::get<DIOp::Name>(Op).get##OpName());                          \
    break;
#define HANDLE_OP2(Name, OpType1, OpName1, OpType2, OpName2)                   \
  case DIOp::Name::getBitcodeID():                                             \
    Ok = Add(std::get<DIOp::Name>(Op).get##OpName1()) &&                       \
         Add(std::get<DIOp::Name>(Op).get##OpName2());                         \
    break;
#include "llvm/IR/DIExprOps.def"
  default:
    break;
  }
  if (!Ok)
    return std::nullopt;

  auto [It, Inserted] = DIOpOperationCache.try_emplace(Operands);
  if (!Inserted)
    return It->second;
  MCRegister Reg = emitExtInst(SPIRV::NonSemanticExtInst::DebugOperation,
                               VoidTypeReg, ExtInstSetReg, Operands, MAI);
  It->second = Reg;
  return Reg;
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDIOpExpression(
    const DIExpression *Expr, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, SPIRV::ModuleAnalysisInfo &MAI) {
  SmallVector<MCRegister> OperationRegs;
  for (const DIOp::Variant &Op : *Expr->getNewElementsRef()) {
    std::optional<MCRegister> OpReg =
        emitDIOp(Op, VoidTypeReg, I32TypeReg, ExtInstSetReg, MAI);
    if (!OpReg)
      return std::nullopt;
    OperationRegs.push_back(*OpReg);
  }

  auto [It, Inserted] =
      DebugExpressionCache.try_emplace(std::move(OperationRegs));
  if (!Inserted)
    return It->second;
  MCRegister Reg = emitExtInst(SPIRV::NonSemanticExtInst::DebugExpression,
                               VoidTypeReg, ExtInstSetReg, It->first, MAI);
  It->second = Reg;
  return Reg;
}

void SPIRVNonSemanticDebugHandler::addDIOpGlobalExpression(
    SmallVectorImpl<MCRegister> &Ops, const DIExpression *Expr,
    MCRegister VoidTypeReg, MCRegister I32TypeReg, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (!Expr || !Expr->holdsNewElements())
    return;
  std::optional<MCRegister> ExprReg =
      emitDIOpExpression(Expr, VoidTypeReg, I32TypeReg, ExtInstSetReg, MAI);
  if (!ExprReg)
    return;
  // Operand 9 is the optional Static Member Declaration.
  Ops.resize(10, CachedDebugInfoNoneReg);
  Ops.push_back(*ExprReg);
}
