//===- SISpillUtils.cpp - SI spill helper functions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SISpillUtils.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

// Replace frame index in a DBG_VALUE or DBG_VALUE_LIST instruction with VGPR
// lane.
static void updateDbgValueInstForSpillFIs(MachineInstr &MI,
                                          const BitVector &SpillFIs) {
  assert(MI.isDebugValue());
  const MachineFunction *MF = MI.getParent()->getParent();
  auto *FuncInfo = MF->getInfo<SIMachineFunctionInfo>();
  const auto &FrInfo = MF->getFrameInfo();

  auto WasOpndSpilled = [&](const MachineOperand &Opnd) {
    return (Opnd.isFI() && !FrInfo.isFixedObjectIndex(Opnd.getIndex()) &&
            SpillFIs[Opnd.getIndex()]);
  };

  if (MI.getDebugExpression()->holdsOldElements()) {
    // For old-style DIExpressions, just do nothing and we will drop all
    // spilled FIs below.
    // FIXME: We should instead, update it with the
    // correct register value. It should be worked out later.
  } else {
    DIExprBuilder Builder(*MI.getDebugExpression());
    IntegerType *TypeInt16 = IntegerType::get(Builder.getContext(), 16);
    IntegerType *TypeInt32 = IntegerType::get(Builder.getContext(), 32);
    for (auto &&I = Builder.begin(); I != Builder.end();) {
      if (auto *Arg = std::get_if<DIOp::Arg>(&*I++)) {
        MachineOperand &MO = MI.getDebugOperand(Arg->getIndex());
        if (!WasOpndSpilled(MO))
          continue;
        ArrayRef<SIRegisterInfo::SpilledReg> VGPRSpills =
            FuncInfo->getSGPRSpillToVirtualVGPRLanes(MO.getIndex());
        // FIXME: This is a very narrow pattern to match, we could handle much
        // more, both intervening ops and multi-lane spills
        if (I != Builder.end() && std::get_if<DIOp::Deref>(&*I) &&
            VGPRSpills.size() == 1) {
          const SIRegisterInfo::SpilledReg &VGPRSpill = VGPRSpills.front();
          // Change the type of DIOpArg and replace the following DIOpDeref
          // with DIOpConstant + DIOpByteOfset.
          Arg->setResultType(TypeInt32);
          ConstantData *C =
              ConstantInt::get(TypeInt16, VGPRSpill.Lane * 8, true);
          const std::initializer_list<DIOp::Variant> Ops = {
              DIOp::Constant(C), DIOp::ByteOffset(TypeInt32)};
          I = Builder.insert(Builder.erase(I), Ops) + Ops.size();
          // Replace stack (frame index) argument of MI with VGPR
          MO.ChangeToRegister(VGPRSpill.VGPR, false);
        } else {
          MO.ChangeToRegister(Register(), /*isDef=*/false);
        }
      }
    }
    MI.getDebugExpressionOp().setMetadata(Builder.intoExpression());
  }
  // Any spilled FIs we haven't handled by this point should just be dropped.
  for (MachineOperand &Op : MI.debug_operands()) {
    if (WasOpndSpilled(Op))
      Op.ChangeToRegister(Register(), /*isDef=*/false);
  }
}

void llvm::updateDbgValueInstsForSpillFIs(MIVector &Insts,
                                          const BitVector &SpillFIs) {
  for (MachineInstr *MI : Insts) {
    if (MI->isDebugValue() &&
        std::any_of(MI->operands_begin(), MI->operands_end(),
                    [](auto &Opnd) { return Opnd.isFI(); })) {
      updateDbgValueInstForSpillFIs(*MI, SpillFIs);
    }
  }
  Insts.clear();
}
