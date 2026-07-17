//===- SISpillUtils.cpp - SI spill helper functions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SISpillUtils.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

/// Add Reg as a debug operand to MI, converting to a DBG_VALUE_LIST as needed.
/// Returns the operand that is used for Reg.
///
/// If Op is a frame index, it's replaced in-place with Reg.
/// If Op is not a frame index and MI is a DBG_VALUE, converts MI to
/// DBG_VALUE_LIST and adds Reg as an additional operand.
/// If Op is not a frame index and MI is already a DBG_VALUE_LIST, adds Reg
/// as an additional operand.
///
static MachineOperand *
addOrReplaceFrameIndexOp(MachineInstr &MI, MachineOperand *Op, Register Reg) {
  if (Op->isFI()) {
    Op->ChangeToRegister(Reg, /*isDef=*/false);
    return Op;
  }

  // Convert DBG_VALUE to DBG_VALUE_LIST when adding additional operands.
  // This happens when a spill spans multiple registers (e.g., multi-lane SGPR
  // spills to VGPR) and we've already replaced the original FI operand.
  if (MI.isNonListDebugValue()) {
    // DBG_VALUE format: <location>, <indirect offset>, <variable>, <expression>
    // DBG_VALUE_LIST format: <variable>, <expression>, <location1>, ...
    // For DIOp-based expressions, we can ignore the indirect offset.
    MachineOperand LocationOp = MI.getOperand(0);
    MI.removeOperand(1);
    MI.removeOperand(0);
    MI.addOperand(LocationOp);
    MachineFunction *MF = MI.getParent()->getParent();
    const auto &TII = *MF->getSubtarget().getInstrInfo();
    MI.setDesc(TII.get(TargetOpcode::DBG_VALUE_LIST));
  }

  MI.addOperand(MachineOperand::CreateReg(
      Reg, /*isDef=*/false, /*isImp=*/false, /*isKill=*/false, /*isDead=*/false,
      /*isUndef=*/false, /*isEarlyClobber=*/false, /*SubReg=*/0,
      /*isDebug=*/true));
  return &MI.getOperand(MI.getNumOperands() - 1);
}

/// Return a type that indicates that the register should not be focused (the
/// current default).
static Type *getWholeRegType(LLVMContext &Ctx, const GCNSubtarget &ST) {
  StringRef Name =
      ST.isWave32() ? "amdgpu.debug.whole.reg32" : "amdgpu.debug.whole.reg64";
  return TargetExtType::get(Ctx, Name);
}

/// Update DBG_VALUE and DBG_VALUE_LIST instructions so that they correctly
/// reflect performed stack to VGPR spills.
/// Examples:
///  DBG_VALUE  %stack.8, 0, !"next", !DIExpression(DIOpArg(0, ptr addrspace(5)),
///                                                 DIOpDeref(i32))
///    --->
///  DBG_VALUE  %249 : vgpr_32, 0, !"next", !DIExpression(DIOpArg(0, amdgpu.debug.whole.reg32),
///                                                       DIOpConstant(i8 40),
///                                                       DIOpByteOffset(i32))
///
///
///  DBG_VALUE_LIST !"next", !DIExpression(DIOpArg(0, ptr addrspace(5)),
///                                        DIOpDeref(i32),
///                                        DIOpArg(1, ptr addrspace(5)),
///                                        DIOpDeref(i32),
///                                        DIOpAdd()),
///                 %stack.9, %stack.5
///    --->
///  DBG_VALUE_LIST !"next", !DIExpression(DIOpArg(0, amdgpu.debug.whole.reg32),
///                                        DIOpConstant(i8 40),
///                                        DIOpByteOffset(i32),
///                                        DIOpArg(1, ptr addrspace(5)),
///                                        DIOpDeref(i32),
///                                        DIOpAdd()),
///                 %14 : vgpr_32, %stack.5
///
void llvm::updateDbgValueForSISpill(MachineFunction &MF, MachineInstr &MI,
                                    const BitVector &SpillFIs,
                                    SISpillKind Kind) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const MachineFrameInfo &FrInfo = MF.getFrameInfo();
  const DIExpression *Expr = MI.getDebugExpression();

  auto WasOpndSpilled = [&](const MachineOperand &Opnd) {
    return (Opnd.isFI() && !FrInfo.isFixedObjectIndex(Opnd.getIndex()) &&
            SpillFIs[Opnd.getIndex()]);
  };
  auto ClearSpilledOpnds = [&] {
    for (MachineOperand &Op : MI.debug_operands())
      if (WasOpndSpilled(Op))
        Op.ChangeToRegister(Register(), /*isDef=*/false);
  };

  if (llvm::none_of(MI.debug_operands(), WasOpndSpilled))
    return;

  // For old-style DIExpressions, just drop all spilled FIs. FIXME: We should
  // instead, update it with the correct register value. It should be worked out
  // later.
  if (Expr->holdsOldElements()) {
    ClearSpilledOpnds();
    return;
  }

  LLVMContext &Ctx = Expr->getContext();
  auto *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  constexpr unsigned VGPRLaneSize = 32;
  IntegerType *TypeIntLane = IntegerType::get(Ctx, VGPRLaneSize);
  Type *TypeWholeReg = getWholeRegType(Ctx, ST);
  IntegerType *TypeInt32 = IntegerType::get(Ctx, 32);

  ArrayRef<DIOp::Variant> ExprOps = *Expr->getNewElementsRef();
  DIExprBuilder Builder(Ctx);
  for (auto Iter = ExprOps.begin(), End = ExprOps.end(); Iter != End; ++Iter) {
    auto *Arg = std::get_if<DIOp::Arg>(&*Iter);
    if (!Arg) {
      Builder.append(*Iter);
      continue;
    }
    MachineOperand *MO = &MI.getDebugOperand(Arg->getIndex());
    if (!WasOpndSpilled(*MO)) {
      Builder.append(*Arg);
      continue;
    }
    auto Next = std::next(Iter);
    if (Next == End || !std::holds_alternative<DIOp::Deref>(*Next)) {
      ClearSpilledOpnds();
      return;
    }
    // Skip the Deref next iteration.
    Iter = Next;

    Type *TypeDeref = std::get<DIOp::Deref>(*Next).getResultType();
    switch (Kind) {
    case SISpillKind::VGPRToAGPR: {
      const SIMachineFunctionInfo::VGPRSpillToAGPR &Spill =
          FuncInfo->getVGPRToAGPRSpill(MO->getIndex());
      for (MCPhysReg Reg : Spill.Lanes) {
        MO = addOrReplaceFrameIndexOp(MI, MO, Reg);
        unsigned ArgNo = MI.getDebugOperandIndex(MO);
        Type *ArgTy = Spill.Lanes.size() == 1 ? TypeDeref : TypeIntLane;
        Builder.append(DIOp::Arg(ArgNo, ArgTy));
      }
      if (Spill.Lanes.size() > 1)
        Builder.append(DIOp::Composite(Spill.Lanes.size(), TypeDeref));
      break;
    }

    case SISpillKind::SGPRToVGPR: {
      ArrayRef<SIRegisterInfo::SpilledReg> VGPRSpills =
          FuncInfo->getSGPRSpillToVirtualVGPRLanes(MO->getIndex());
      for (const SIRegisterInfo::SpilledReg &Spill : VGPRSpills) {
        MO = addOrReplaceFrameIndexOp(MI, MO, Spill.VGPR);
        unsigned ArgNo = MI.getDebugOperandIndex(MO);
        ConstantData *C =
            ConstantInt::get(TypeInt32, Spill.Lane * VGPRLaneSize / 8);
        Type *OffsetTy = VGPRSpills.size() == 1 ? TypeDeref : TypeIntLane;
        Builder.append(DIOp::Arg(ArgNo, TypeWholeReg));
        Builder.append(DIOp::Constant(C));
        Builder.append(DIOp::ByteOffset(OffsetTy));
      }
      if (VGPRSpills.size() > 1)
        Builder.append(DIOp::Composite(VGPRSpills.size(), TypeDeref));
      break;
    }
    }
  }

  MI.getDebugExpressionOp().setMetadata(Builder.intoExpression());
}
