//===- llvm/CodeGen/DwarfExpression.cpp - Dwarf Debug Framework -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfExpression.h"
#include "DwarfCompileUnit.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <stack>

using namespace llvm;

#define DEBUG_TYPE "dwarfdebug"

void DwarfExpression::emitConstu(uint64_t Value) {
  if (Value < 32)
    emitOp(dwarf::DW_OP_lit0 + Value);
  else if (Value == std::numeric_limits<uint64_t>::max()) {
    // Only do this for 64-bit values as the DWARF expression stack uses
    // target-address-size values.
    emitOp(dwarf::DW_OP_lit0);
    emitOp(dwarf::DW_OP_not);
  } else {
    emitOp(dwarf::DW_OP_constu);
    emitUnsigned(Value);
  }
}

void DwarfExpression::addReg(int64_t DwarfReg, const char *Comment) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  assert(ASTRoot || (isUnknownLocation() || isRegisterLocation()) &&
                        "location description already locked down");
  if (!ASTRoot)
    LocationKind = Register;
  if (DwarfReg < 32) {
    emitOp(dwarf::DW_OP_reg0 + DwarfReg, Comment);
  } else {
    emitOp(dwarf::DW_OP_regx, Comment);
    emitUnsigned(DwarfReg);
  }
}

void DwarfExpression::addBReg(int64_t DwarfReg, int64_t Offset) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  assert(!isRegisterLocation() && "location description already locked down");
  if (DwarfReg < 32) {
    emitOp(dwarf::DW_OP_breg0 + DwarfReg);
  } else {
    emitOp(dwarf::DW_OP_bregx);
    emitUnsigned(DwarfReg);
  }
  emitSigned(Offset);
}

void DwarfExpression::addFBReg(int64_t Offset) {
  emitOp(dwarf::DW_OP_fbreg);
  emitSigned(Offset);
}

void DwarfExpression::addOpPiece(unsigned SizeInBits, unsigned OffsetInBits) {
  if (!SizeInBits)
    return;

  const unsigned SizeOfByte = 8;
  if (OffsetInBits > 0 || SizeInBits % SizeOfByte) {
    emitOp(dwarf::DW_OP_bit_piece);
    emitUnsigned(SizeInBits);
    emitUnsigned(OffsetInBits);
  } else {
    emitOp(dwarf::DW_OP_piece);
    unsigned ByteSize = SizeInBits / SizeOfByte;
    emitUnsigned(ByteSize);
  }
  this->OffsetInBits += SizeInBits;
}

void DwarfExpression::addShr(unsigned ShiftBy) {
  emitConstu(ShiftBy);
  emitOp(dwarf::DW_OP_shr);
}

void DwarfExpression::addAnd(unsigned Mask) {
  emitConstu(Mask);
  emitOp(dwarf::DW_OP_and);
}

bool DwarfExpression::addMachineReg(const TargetRegisterInfo &TRI,
                                    llvm::Register MachineReg,
                                    unsigned MaxSize) {
  if (!MachineReg.isPhysical()) {
    if (isFrameRegister(TRI, MachineReg)) {
      DwarfRegs.push_back(Register::createRegister(-1, nullptr));
      return true;
    }
    // Try getting dwarf register for virtual register anyway, eg. for NVPTX.
    int64_t Reg = TRI.getDwarfRegNumForVirtReg(MachineReg, false);
    if (Reg > 0) {
      DwarfRegs.push_back(Register::createRegister(Reg, nullptr));
      return true;
    }
    return false;
  }

  int64_t Reg = TRI.getDwarfRegNum(MachineReg, false);

  // If this is a valid register number, emit it.
  if (Reg >= 0) {
    DwarfRegs.push_back(Register::createRegister(Reg, nullptr));
    return true;
  }

  // Walk up the super-register chain until we find a valid number.
  // For example, EAX on x86_64 is a 32-bit fragment of RAX with offset 0.
  for (MCPhysReg SR : TRI.superregs(MachineReg)) {
    Reg = TRI.getDwarfRegNum(SR, false);
    if (Reg >= 0) {
      unsigned Idx = TRI.getSubRegIndex(SR, MachineReg);
      unsigned Size = TRI.getSubRegIdxSize(Idx);
      unsigned RegOffset = TRI.getSubRegIdxOffset(Idx);
      DwarfRegs.push_back(Register::createRegister(Reg, "super-register"));
      // Use a DW_OP_bit_piece to describe the sub-register.
      setSubRegisterPiece(Size, RegOffset);
      return true;
    }
  }

  // Otherwise, attempt to find a covering set of sub-register numbers.
  // For example, Q0 on ARM is a composition of D0+D1.
  unsigned CurPos = 0;
  // The size of the register in bits.
  const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(MachineReg);
  unsigned RegSize = TRI.getRegSizeInBits(*RC);
  // Keep track of the bits in the register we already emitted, so we
  // can avoid emitting redundant aliasing subregs. Because this is
  // just doing a greedy scan of all subregisters, it is possible that
  // this doesn't find a combination of subregisters that fully cover
  // the register (even though one may exist).
  SmallBitVector Coverage(RegSize, false);
  for (MCPhysReg SR : TRI.subregs(MachineReg)) {
    unsigned Idx = TRI.getSubRegIndex(MachineReg, SR);
    unsigned Size = TRI.getSubRegIdxSize(Idx);
    unsigned Offset = TRI.getSubRegIdxOffset(Idx);
    Reg = TRI.getDwarfRegNum(SR, false);
    if (Reg < 0)
      continue;

    // Used to build the intersection between the bits we already
    // emitted and the bits covered by this subregister.
    SmallBitVector CurSubReg(RegSize, false);
    CurSubReg.set(Offset, Offset + Size);

    // If this sub-register has a DWARF number and we haven't covered
    // its range, and its range covers the value, emit a DWARF piece for it.
    if (Offset < MaxSize && CurSubReg.test(Coverage)) {
      // Emit a piece for any gap in the coverage.
      if (Offset > CurPos)
        DwarfRegs.push_back(Register::createSubRegister(
            -1, Offset - CurPos, "no DWARF register encoding"));
      if (Offset == 0 && Size >= MaxSize)
        DwarfRegs.push_back(Register::createRegister(Reg, "sub-register"));
      else
        DwarfRegs.push_back(Register::createSubRegister(
            Reg, std::min<unsigned>(Size, MaxSize - Offset), "sub-register"));
    }
    // Mark it as emitted.
    Coverage.set(Offset, Offset + Size);
    CurPos = Offset + Size;
  }
  // Failed to find any DWARF encoding.
  if (CurPos == 0)
    return false;
  // Found a partial or complete DWARF encoding.
  if (CurPos < RegSize)
    DwarfRegs.push_back(Register::createSubRegister(
        -1, RegSize - CurPos, "no DWARF register encoding"));
  return true;
}

void DwarfExpression::addStackValue() {
  if (DwarfVersion >= 4)
    emitOp(dwarf::DW_OP_stack_value);
}

void DwarfExpression::addSignedConstant(int64_t Value) {
  if (IsPoisonedExpr || !IsImplemented)
    return;
  assert(isImplicitLocation() || isUnknownLocation());
  LocationKind = Implicit;
  emitOp(dwarf::DW_OP_consts);
  emitSigned(Value);
}

void DwarfExpression::addUnsignedConstant(uint64_t Value) {
  if (IsPoisonedExpr || !IsImplemented)
    return;
  assert(isImplicitLocation() || isUnknownLocation());
  LocationKind = Implicit;
  emitConstu(Value);
}

void DwarfExpression::addUnsignedConstant(const APInt &Value) {
  if (IsPoisonedExpr || !IsImplemented)
    return;
  assert(isImplicitLocation() || isUnknownLocation());
  LocationKind = Implicit;

  unsigned Size = Value.getBitWidth();
  const uint64_t *Data = Value.getRawData();

  // Chop it up into 64-bit pieces, because that's the maximum that
  // addUnsignedConstant takes.
  unsigned Offset = 0;
  while (Offset < Size) {
    addUnsignedConstant(*Data++);
    if (Offset == 0 && Size <= 64)
      break;
    addStackValue();
    addOpPiece(std::min(Size - Offset, 64u), Offset);
    Offset += 64;
  }
}

void DwarfExpression::addConstantFP(const APFloat &APF, const AsmPrinter &AP) {
  if (IsPoisonedExpr || !IsImplemented)
    return;
  assert(isImplicitLocation() || isUnknownLocation());
  APInt API = APF.bitcastToAPInt();
  int NumBytes = API.getBitWidth() / 8;
  if (NumBytes == 4 /*float*/ || NumBytes == 8 /*double*/) {
    // FIXME: Add support for `long double`.
    emitOp(dwarf::DW_OP_implicit_value);
    emitUnsigned(NumBytes /*Size of the block in bytes*/);

    // The loop below is emitting the value starting at least significant byte,
    // so we need to perform a byte-swap to get the byte order correct in case
    // of a big-endian target.
    if (AP.getDataLayout().isBigEndian())
      API = API.byteSwap();

    for (int i = 0; i < NumBytes; ++i) {
      emitData1(API.getZExtValue() & 0xFF);
      API = API.lshr(8);
    }

    return;
  }
  LLVM_DEBUG(
      dbgs() << "Skipped DW_OP_implicit_value creation for ConstantFP of size: "
             << API.getBitWidth() << " bits\n");
}

bool DwarfExpression::addMachineRegExpression(const TargetRegisterInfo &TRI,
                                              DIExpressionCursor &ExprCursor,
                                              llvm::Register MachineReg,
                                              unsigned FragmentOffsetInBits) {
  if (IsPoisonedExpr || !IsImplemented)
    return true;
  auto Fragment = ExprCursor.getFragmentInfo();
  if (!addMachineReg(TRI, MachineReg, Fragment ? Fragment->SizeInBits : ~1U)) {
    LocationKind = Unknown;
    return false;
  }

  bool HasComplexExpression = false;
  auto Op = ExprCursor.peek();
  if (Op && Op->getOp() != dwarf::DW_OP_LLVM_fragment)
    HasComplexExpression = true;

  // If the register can only be described by a complex expression (i.e.,
  // multiple subregisters) it doesn't safely compose with another complex
  // expression. For example, it is not possible to apply a DW_OP_deref
  // operation to multiple DW_OP_pieces, since composite location descriptions
  // do not push anything on the DWARF stack.
  //
  // DW_OP_entry_value operations can only hold a DWARF expression or a
  // register location description, so we can't emit a single entry value
  // covering a composite location description. In the future we may want to
  // emit entry value operations for each register location in the composite
  // location, but until that is supported do not emit anything.
  if ((HasComplexExpression || IsEmittingEntryValue) && DwarfRegs.size() > 1) {
    if (IsEmittingEntryValue)
      cancelEntryValue();
    DwarfRegs.clear();
    LocationKind = Unknown;
    return false;
  }

  // Handle simple register locations. If we are supposed to emit
  // a call site parameter expression and if that expression is just a register
  // location, emit it with addBReg and offset 0, because we should emit a DWARF
  // expression representing a value, rather than a location.
  if ((!isParameterValue() && !isMemoryLocation() && !HasComplexExpression) ||
      isEntryValue()) {
    auto FragmentInfo = ExprCursor.getFragmentInfo();
    unsigned RegSize = 0;
    for (auto &Reg : DwarfRegs) {
      RegSize += Reg.SubRegSize;
      if (Reg.DwarfRegNo >= 0)
        addReg(Reg.DwarfRegNo, Reg.Comment);
      if (FragmentInfo)
        if (RegSize > FragmentInfo->SizeInBits)
          // If the register is larger than the current fragment stop
          // once the fragment is covered.
          break;
      addOpPiece(Reg.SubRegSize);
    }

    if (isEntryValue()) {
      finalizeEntryValue();

      if (!isIndirect() && !isParameterValue() && !HasComplexExpression &&
          DwarfVersion >= 4)
        emitOp(dwarf::DW_OP_stack_value);
    }

    DwarfRegs.clear();
    // If we need to mask out a subregister, do it now, unless the next
    // operation would emit an OpPiece anyway.
    auto NextOp = ExprCursor.peek();
    if (SubRegisterSizeInBits && NextOp &&
        (NextOp->getOp() != dwarf::DW_OP_LLVM_fragment))
      maskSubRegister();
    return true;
  }

  // Don't emit locations that cannot be expressed without DW_OP_stack_value.
  if (DwarfVersion < 4)
    if (any_of(ExprCursor, [](DIExpression::ExprOperand Op) -> bool {
          return Op.getOp() == dwarf::DW_OP_stack_value;
        })) {
      DwarfRegs.clear();
      LocationKind = Unknown;
      return false;
    }

  // TODO: We should not give up here but the following code needs to be changed
  //       to deal with multiple (sub)registers first.
  if (DwarfRegs.size() > 1) {
    LLVM_DEBUG(dbgs() << "TODO: giving up on debug information due to "
                         "multi-register usage.\n");
    DwarfRegs.clear();
    LocationKind = Unknown;
    return false;
  }

  auto Reg = DwarfRegs[0];
  bool FBReg = isFrameRegister(TRI, MachineReg);
  int SignedOffset = 0;

  // Pattern-match combinations for which more efficient representations exist.
  // [Reg, DW_OP_plus_uconst, Offset] --> [DW_OP_breg, Offset].
  if (Op && (Op->getOp() == dwarf::DW_OP_plus_uconst)) {
    uint64_t Offset = Op->getArg(0);
    uint64_t IntMax = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (Offset <= IntMax) {
      SignedOffset = Offset;
      ExprCursor.take();
    }
  }

  // [Reg, DW_OP_constu, Offset, DW_OP_plus]  --> [DW_OP_breg, Offset]
  // [Reg, DW_OP_constu, Offset, DW_OP_minus] --> [DW_OP_breg,-Offset]
  // If Reg is a subregister we need to mask it out before subtracting.
  if (Op && Op->getOp() == dwarf::DW_OP_constu) {
    uint64_t Offset = Op->getArg(0);
    uint64_t IntMax = static_cast<uint64_t>(std::numeric_limits<int>::max());
    auto N = ExprCursor.peekNext();
    if (N && N->getOp() == dwarf::DW_OP_plus && Offset <= IntMax) {
      SignedOffset = Offset;
      ExprCursor.consume(2);
    } else if (N && N->getOp() == dwarf::DW_OP_minus &&
               !SubRegisterSizeInBits && Offset <= IntMax + 1) {
      SignedOffset = -static_cast<int64_t>(Offset);
      ExprCursor.consume(2);
    }
  }

  if (FBReg)
    addFBReg(SignedOffset);
  else {
    addBReg(Reg.DwarfRegNo, SignedOffset);
    // Compose the remaining subregs.
    unsigned ShAmt = Reg.SubRegSize;
    for (unsigned i = 1, e = DwarfRegs.size(); i < e; ++i) {
      Reg = DwarfRegs[i];
      addBReg(Reg.DwarfRegNo, 0);
      emitOp(dwarf::DW_OP_constu);
      emitUnsigned(ShAmt);
      emitOp(dwarf::DW_OP_shl);
      emitOp(dwarf::DW_OP_plus);
      ShAmt += Reg.SubRegSize;
    }
  }
  DwarfRegs.clear();

  // If we need to mask out a subregister, do it now, unless the next
  // operation would emit an OpPiece anyway.
  auto NextOp = ExprCursor.peek();
  if (SubRegisterSizeInBits && NextOp &&
      (NextOp->getOp() != dwarf::DW_OP_LLVM_fragment))
    maskSubRegister();

  return true;
}

void DwarfExpression::setEntryValueFlags(const MachineLocation &Loc) {
  LocationFlags |= EntryValue;
  if (Loc.isIndirect())
    LocationFlags |= Indirect;
}

void DwarfExpression::setLocation(const MachineLocation &Loc,
                                  const DIExpression *DIExpr) {
  if (Loc.isIndirect())
    setMemoryLocationKind();

  if (DIExpr->isEntryValue())
    setEntryValueFlags(Loc);
}

void DwarfExpression::beginEntryValueExpression(
    DIExpressionCursor &ExprCursor) {
  auto Op = ExprCursor.take();
  (void)Op;
  assert(Op && Op->getOp() == dwarf::DW_OP_LLVM_entry_value);
  assert(!IsEmittingEntryValue && "Already emitting entry value?");
  assert(Op->getArg(0) == 1 &&
         "Can currently only emit entry values covering a single operation");

  SavedLocationKind = LocationKind;
  LocationKind = Register;
  LocationFlags |= EntryValue;
  IsEmittingEntryValue = true;
  enableTemporaryBuffer();
}

void DwarfExpression::finalizeEntryValue() {
  assert(IsEmittingEntryValue && "Entry value not open?");
  disableTemporaryBuffer();

  emitOp(CU.getDwarf5OrGNULocationAtom(dwarf::DW_OP_entry_value));

  // Emit the entry value's size operand.
  unsigned Size = getTemporaryBufferSize();
  emitUnsigned(Size);

  // Emit the entry value's DWARF block operand.
  commitTemporaryBuffer();

  LocationFlags &= ~EntryValue;
  LocationKind = SavedLocationKind;
  IsEmittingEntryValue = false;
}

void DwarfExpression::cancelEntryValue() {
  assert(IsEmittingEntryValue && "Entry value not open?");
  disableTemporaryBuffer();

  // The temporary buffer can't be emptied, so for now just assert that nothing
  // has been emitted to it.
  assert(getTemporaryBufferSize() == 0 &&
         "Began emitting entry value block before cancelling entry value");

  LocationKind = SavedLocationKind;
  IsEmittingEntryValue = false;
}

unsigned DwarfExpression::getOrCreateBaseType(unsigned BitSize,
                                              dwarf::TypeKind Encoding) {
  // Reuse the base_type if we already have one in this CU otherwise we
  // create a new one.
  unsigned I = 0, E = CU.ExprRefedBaseTypes.size();
  for (; I != E; ++I)
    if (CU.ExprRefedBaseTypes[I].BitSize == BitSize &&
        CU.ExprRefedBaseTypes[I].Encoding == Encoding)
      break;

  if (I == E)
    CU.ExprRefedBaseTypes.emplace_back(BitSize, Encoding);
  return I;
}

/// Assuming a well-formed expression, match "DW_OP_deref*
/// DW_OP_LLVM_fragment?".
static bool isMemoryLocation(DIExpressionCursor ExprCursor) {
  while (ExprCursor) {
    auto Op = ExprCursor.take();
    switch (Op->getOp()) {
    case dwarf::DW_OP_deref:
    case dwarf::DW_OP_LLVM_fragment:
      break;
    default:
      return false;
    }
  }
  return true;
}

void DwarfExpression::addExpression(DIExpressionCursor &&ExprCursor) {
  addExpression(std::move(ExprCursor),
                [](unsigned Idx, DIExpressionCursor &Cursor) -> bool {
                  llvm_unreachable("unhandled opcode found in expression");
                });
}

bool DwarfExpression::addExpression(
    DIExpressionCursor &&ExprCursor,
    llvm::function_ref<bool(unsigned, DIExpressionCursor &)> InsertArg) {
  // Entry values can currently only cover the initial register location,
  // and not any other parts of the following DWARF expression.
  assert(!IsEmittingEntryValue && "Can't emit entry value around expression");

  if (!IsImplemented)
    return false;
  IsPoisonedExpr = false;

  std::optional<DIExpression::ExprOperand> PrevConvertOp;

  while (ExprCursor) {
    auto Op = ExprCursor.take();
    uint64_t OpNum = Op->getOp();

    if (OpNum >= dwarf::DW_OP_reg0 && OpNum <= dwarf::DW_OP_reg31) {
      emitOp(OpNum);
      continue;
    } else if (OpNum >= dwarf::DW_OP_breg0 && OpNum <= dwarf::DW_OP_breg31) {
      addBReg(OpNum - dwarf::DW_OP_breg0, Op->getArg(0));
      continue;
    }

    switch (OpNum) {
    case dwarf::DW_OP_LLVM_poisoned:
      emitUserOp(dwarf::DW_OP_LLVM_undefined);
      LocationKind = Unknown;
      break;
    case dwarf::DW_OP_LLVM_arg:
      if (!InsertArg(Op->getArg(0), ExprCursor)) {
        LocationKind = Unknown;
        return false;
      }
      break;
    case dwarf::DW_OP_LLVM_fragment: {
      unsigned SizeInBits = Op->getArg(1);
      unsigned FragmentOffset = Op->getArg(0);
      // The fragment offset must have already been adjusted by emitting an
      // empty DW_OP_piece / DW_OP_bit_piece before we emitted the base
      // location.
      assert(OffsetInBits >= FragmentOffset && "fragment offset not added?");
      assert(SizeInBits >= OffsetInBits - FragmentOffset && "size underflow");

      // If addMachineReg already emitted DW_OP_piece operations to represent
      // a super-register by splicing together sub-registers, subtract the size
      // of the pieces that was already emitted.
      SizeInBits -= OffsetInBits - FragmentOffset;

      // If addMachineReg requested a DW_OP_bit_piece to stencil out a
      // sub-register that is smaller than the current fragment's size, use it.
      if (SubRegisterSizeInBits)
        SizeInBits = std::min<unsigned>(SizeInBits, SubRegisterSizeInBits);

      // Emit a DW_OP_stack_value for implicit location descriptions.
      if (isImplicitLocation())
        addStackValue();

      // Emit the DW_OP_piece.
      addOpPiece(SizeInBits, SubRegisterOffsetInBits);
      setSubRegisterPiece(0, 0);
      // Reset the location description kind.
      LocationKind = Unknown;
      return true;
    }
    case dwarf::DW_OP_LLVM_extract_bits_sext:
    case dwarf::DW_OP_LLVM_extract_bits_zext: {
      unsigned SizeInBits = Op->getArg(1);
      unsigned BitOffset = Op->getArg(0);

      // If we have a memory location then dereference to get the value, though
      // we have to make sure we don't dereference any bytes past the end of the
      // object.
      if (isMemoryLocation()) {
        emitOp(dwarf::DW_OP_deref_size);
        emitUnsigned(alignTo(BitOffset + SizeInBits, 8) / 8);
      }

      // Extract the bits by a shift left (to shift out the bits after what we
      // want to extract) followed by shift right (to shift the bits to position
      // 0 and also sign/zero extend). These operations are done in the DWARF
      // "generic type" whose size is the size of a pointer.
      unsigned PtrSizeInBytes = CU.getAsmPrinter()->MAI->getCodePointerSize();
      unsigned LeftShift = PtrSizeInBytes * 8 - (SizeInBits + BitOffset);
      unsigned RightShift = LeftShift + BitOffset;
      if (LeftShift) {
        emitOp(dwarf::DW_OP_constu);
        emitUnsigned(LeftShift);
        emitOp(dwarf::DW_OP_shl);
      }
      emitOp(dwarf::DW_OP_constu);
      emitUnsigned(RightShift);
      emitOp(OpNum == dwarf::DW_OP_LLVM_extract_bits_sext ? dwarf::DW_OP_shra
                                                          : dwarf::DW_OP_shr);

      // The value is now at the top of the stack, so set the location to
      // implicit so that we get a stack_value at the end.
      LocationKind = Implicit;
      break;
    }
    case dwarf::DW_OP_plus_uconst:
      assert(!isRegisterLocation());
      emitOp(dwarf::DW_OP_plus_uconst);
      emitUnsigned(Op->getArg(0));
      break;
    case dwarf::DW_OP_plus:
    case dwarf::DW_OP_minus:
    case dwarf::DW_OP_mul:
    case dwarf::DW_OP_div:
    case dwarf::DW_OP_mod:
    case dwarf::DW_OP_or:
    case dwarf::DW_OP_and:
    case dwarf::DW_OP_xor:
    case dwarf::DW_OP_shl:
    case dwarf::DW_OP_shr:
    case dwarf::DW_OP_shra:
    case dwarf::DW_OP_lit0:
    case dwarf::DW_OP_not:
    case dwarf::DW_OP_dup:
    case dwarf::DW_OP_push_object_address:
    case dwarf::DW_OP_over:
    case dwarf::DW_OP_eq:
    case dwarf::DW_OP_ne:
    case dwarf::DW_OP_gt:
    case dwarf::DW_OP_ge:
    case dwarf::DW_OP_lt:
    case dwarf::DW_OP_le:
      emitOp(OpNum);
      break;
    case dwarf::DW_OP_deref:
      assert(!isRegisterLocation());
      if (!isMemoryLocation() && ::isMemoryLocation(ExprCursor))
        // Turning this into a memory location description makes the deref
        // implicit.
        LocationKind = Memory;
      else
        emitOp(dwarf::DW_OP_deref);
      break;
    case dwarf::DW_OP_constu:
      assert(!isRegisterLocation());
      emitConstu(Op->getArg(0));
      break;
    case dwarf::DW_OP_consts:
      assert(!isRegisterLocation());
      emitOp(dwarf::DW_OP_consts);
      emitSigned(Op->getArg(0));
      break;
    case dwarf::DW_OP_LLVM_convert: {
      unsigned BitSize = Op->getArg(0);
      dwarf::TypeKind Encoding = static_cast<dwarf::TypeKind>(Op->getArg(1));
      if (DwarfVersion >= 5 && CU.getDwarfDebug().useOpConvert()) {
        emitOp(dwarf::DW_OP_convert);
        // If targeting a location-list; simply emit the index into the raw
        // byte stream as ULEB128, DwarfDebug::emitDebugLocEntry has been
        // fitted with means to extract it later.
        // If targeting a inlined DW_AT_location; insert a DIEBaseTypeRef
        // (containing the index and a resolve mechanism during emit) into the
        // DIE value list.
        emitBaseTypeRef(getOrCreateBaseType(BitSize, Encoding));
      } else {
        if (PrevConvertOp && PrevConvertOp->getArg(0) < BitSize) {
          if (Encoding == dwarf::DW_ATE_signed)
            emitLegacySExt(PrevConvertOp->getArg(0));
          else if (Encoding == dwarf::DW_ATE_unsigned)
            emitLegacyZExt(PrevConvertOp->getArg(0));
          PrevConvertOp = std::nullopt;
        } else {
          PrevConvertOp = Op;
        }
      }
      break;
    }
    case dwarf::DW_OP_stack_value:
      LocationKind = Implicit;
      break;
    case dwarf::DW_OP_swap:
      assert(!isRegisterLocation());
      emitOp(dwarf::DW_OP_swap);
      break;
    case dwarf::DW_OP_xderef:
      assert(!isRegisterLocation());
      emitOp(dwarf::DW_OP_xderef);
      break;
    case dwarf::DW_OP_deref_size:
      emitOp(dwarf::DW_OP_deref_size);
      emitData1(Op->getArg(0));
      break;
    case dwarf::DW_OP_LLVM_tag_offset:
      TagOffset = Op->getArg(0);
      break;
    case dwarf::DW_OP_regx:
      emitOp(dwarf::DW_OP_regx);
      emitUnsigned(Op->getArg(0));
      break;
    case dwarf::DW_OP_bregx:
      emitOp(dwarf::DW_OP_bregx);
      emitUnsigned(Op->getArg(0));
      emitSigned(Op->getArg(1));
      break;
    default:
      llvm_unreachable("unhandled opcode found in expression");
    }
  }

  if (isImplicitLocation() && !isParameterValue())
    // Turn this into an implicit location description.
    addStackValue();

  return true;
}

void DwarfExpression::addExpression(DIExpression::NewElementsRef Expr,
                                    ArrayRef<DbgValueLocEntry> ArgLocEntries,
                                    const TargetRegisterInfo *TRI) {
  if (!IsImplemented)
    return;
  assert(!IsPoisonedExpr && "poisoned exprs should have old elements");
  this->ArgLocEntries = ArgLocEntries;
  this->TRI = TRI;
  std::optional<DIOp::Fragment> FragOp;
  for (DIOp::Variant Op : Expr) {
    if (auto *Frag = std::get_if<DIOp::Fragment>(&Op)) {
      FragOp = *Frag;
      IsFragment = true;
      break;
    }
  }
  buildAST(Expr);
  traverse(ASTRoot.get(), ValueKind::LocationDesc,
           /*PermitDivergentAddrSpace=*/
           PermitDivergentAddrSpaceResult && !IsFragment);
  if (FragOp)
    addOpPiece(FragOp->getBitSize());
  if (!IsImplemented)
    emitUserOp(dwarf::DW_OP_LLVM_undefined);
  IsFragment = false;
  ASTRoot.reset();
  this->TRI = nullptr;
  this->ArgLocEntries = std::nullopt;
}

/// add masking operations to stencil out a subregister.
void DwarfExpression::maskSubRegister() {
  assert(SubRegisterSizeInBits && "no subregister was registered");
  if (SubRegisterOffsetInBits > 0)
    addShr(SubRegisterOffsetInBits);
  uint64_t Mask = (1ULL << (uint64_t)SubRegisterSizeInBits) - 1ULL;
  addAnd(Mask);
}

void DwarfExpression::emitUserOp(uint8_t UserOp, const char *Comment) {
  emitOp(dwarf::DW_OP_LLVM_user);
  emitOp(UserOp);
}

void DwarfExpression::finalize() {
  assert(DwarfRegs.size() == 0 && "dwarf registers not emitted");
  // Emit any outstanding DW_OP_piece operations to mask out subregisters.
  if (SubRegisterSizeInBits == 0)
    return;
  // Don't emit a DW_OP_piece for a subregister at offset 0.
  if (SubRegisterOffsetInBits == 0)
    return;
  addOpPiece(SubRegisterSizeInBits, SubRegisterOffsetInBits);
}

void DwarfExpression::addFragmentOffset(const DIExpression *Expr) {
  if (!Expr || !IsImplemented)
    return;

  if (Expr->holdsOldElements() && Expr->isPoisoned())
    IsPoisonedExpr = true;

  if (!Expr->isFragment())
    return;

  uint64_t FragmentOffset = Expr->getFragmentInfo()->OffsetInBits;
  assert(FragmentOffset >= OffsetInBits &&
         "overlapping or duplicate fragments");
  if (FragmentOffset > OffsetInBits)
    addOpPiece(FragmentOffset - OffsetInBits);
  OffsetInBits = FragmentOffset;
}

void DwarfExpression::emitLegacySExt(unsigned FromBits) {
  // (((X >> (FromBits - 1)) * (~0)) << FromBits) | X
  emitOp(dwarf::DW_OP_dup);
  emitOp(dwarf::DW_OP_constu);
  emitUnsigned(FromBits - 1);
  emitOp(dwarf::DW_OP_shr);
  emitOp(dwarf::DW_OP_lit0);
  emitOp(dwarf::DW_OP_not);
  emitOp(dwarf::DW_OP_mul);
  emitOp(dwarf::DW_OP_constu);
  emitUnsigned(FromBits);
  emitOp(dwarf::DW_OP_shl);
  emitOp(dwarf::DW_OP_or);
}

void DwarfExpression::emitLegacyZExt(unsigned FromBits) {
  // Heuristic to decide the most efficient encoding.
  // A ULEB can encode 7 1-bits per byte.
  if (FromBits / 7 < 1+1+1+1+1) {
    // (X & (1 << FromBits - 1))
    emitOp(dwarf::DW_OP_constu);
    emitUnsigned((1ULL << FromBits) - 1);
  } else {
    // Note that the DWARF 4 stack consists of pointer-sized elements,
    // so technically it doesn't make sense to shift left more than 64
    // bits. We leave that for the consumer to decide though. LLDB for
    // example uses APInt for the stack elements and can still deal
    // with this.
    emitOp(dwarf::DW_OP_lit1);
    emitOp(dwarf::DW_OP_constu);
    emitUnsigned(FromBits);
    emitOp(dwarf::DW_OP_shl);
    emitOp(dwarf::DW_OP_lit1);
    emitOp(dwarf::DW_OP_minus);
  }
  emitOp(dwarf::DW_OP_and);
}

void DwarfExpression::addWasmLocation(unsigned Index, uint64_t Offset) {
  if (IsPoisonedExpr || !IsImplemented)
    return;
  emitOp(dwarf::DW_OP_WASM_location);
  emitUnsigned(Index == 4/*TI_LOCAL_INDIRECT*/ ? 0/*TI_LOCAL*/ : Index);
  emitUnsigned(Offset);
  if (Index == 4 /*TI_LOCAL_INDIRECT*/) {
    assert(LocationKind == Unknown);
    LocationKind = Memory;
  } else {
    assert(LocationKind == Implicit || LocationKind == Unknown);
    LocationKind = Implicit;
  }
}

static bool isUnsigned(const ConstantInt *CI) {
  return (CI->getIntegerType()->getSignBit() & CI->getSExtValue()) == 0;
}

void DwarfExpression::buildAST(DIExpression::NewElementsRef Elements) {
  std::stack<std::unique_ptr<Node>> Operands;

  for (const auto &Op : Elements) {
    if (std::holds_alternative<DIOp::Fragment>(Op))
      continue;
    std::unique_ptr<DwarfExpression::Node> OpNode =
        std::make_unique<DwarfExpression::Node>(Op);
    size_t OpChildrenCount = DIOp::getNumInputs(OpNode->getElement());
    if (OpChildrenCount == 0) {
      Operands.push(std::move(OpNode));
    } else {
      for (size_t I = 0; I < OpChildrenCount; ++I) {
        OpNode->getChildren().insert(OpNode->getChildren().begin(),
                                     std::move(Operands.top()));
        Operands.pop();
      }
      Operands.push(std::move(OpNode));
    }
  }

  assert(Operands.size() == 1);
  ASTRoot = std::move(Operands.top());
}

using NewOpResult = DwarfExpression::OpResult;

std::optional<NewOpResult>
DwarfExpression::traverse(Node *OpNode, std::optional<ValueKind> ReqVK,
                          bool PermitDivergentAddrSpace) {
  std::optional<NewOpResult> Result =
      std::visit([&](auto &&E) { return traverse(E, OpNode->getChildren()); },
                 OpNode->getElement());
  if (!Result) {
    IsImplemented = false;
    return Result;
  }
  if (Result->DivergentAddrSpace && !PermitDivergentAddrSpace) {
    // FIXME: When DWARF supports address space conversions, generate a
    // DW_OP_convert here to convert to the required address space.
    IsImplemented = false;
    return Result;
  }
  OpNode->setIsLowered();
  OpNode->setResultType(Result->Ty);
  return ReqVK ? convertValueKind(*Result, *ReqVK) : Result;
}

NewOpResult DwarfExpression::convertValueKind(const NewOpResult &Res,
                                              ValueKind ReqVK) {
  if (Res.VK == ValueKind::Value && ReqVK == ValueKind::LocationDesc) {
    emitOp(dwarf::DW_OP_stack_value);
    return {Res.Ty, ValueKind::LocationDesc, Res.DivergentAddrSpace};
  }

  if (Res.VK == ValueKind::LocationDesc && ReqVK == ValueKind::Value) {
    readToValue(Res.Ty);
    return {Res.Ty, ValueKind::Value, Res.DivergentAddrSpace};
  }

  return Res;
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Arg Arg,
                                                     ChildrenT Children) {
  uint32_t Index = Arg.getIndex();
  assert(Index < ArgLocEntries.size());
  auto Entry = ArgLocEntries[Index];

  if (Entry.isGlobal()) {
    const GlobalVariable *GV = Entry.getGlobal();

    // FIXME: This is a workaround to avoid generating symbols for non-global
    // address spaces, e.g. LDS. Generate a 'DW_OP_constu' with a dummy
    // constant value (0) for now.
    unsigned AMDGPUGlobalAddrSpace = 1;
    if ((AP.TM.getTargetTriple().getArch() == Triple::amdgcn) &&
        (GV->getAddressSpace() != AMDGPUGlobalAddrSpace)) {
      emitConstu(0);
      emitOp(dwarf::DW_OP_stack_value);
      return NewOpResult{Arg.getResultType(), ValueKind::LocationDesc};
    }

    // TODO: We only support PIC reloc-model and non-TLS globals so far, see
    // DwarfCompileUnit::addLocationAttribute(..., DIGlobalVariable *, ...) for
    // what (more) general support might entail.
    if (GV->isThreadLocal() || AP.TM.getRelocationModel() != Reloc::PIC_ ||
        AP.TM.getTargetTriple().isWasm())
      return std::nullopt;

    CU.getDwarfDebug().addArangeLabel(SymbolCU(&CU, AP.getSymbol(GV)));
    emitOpAddress(GV);
    emitOp(dwarf::DW_OP_stack_value);
    return NewOpResult{Arg.getResultType(), ValueKind::LocationDesc};
  }

  if (Entry.isLocation()) {
    assert(DwarfRegs.empty() && "unconsumed registers?");
    if (!TRI || !addMachineReg(*TRI, Entry.getLoc().getReg())) {
      DwarfRegs.clear();
      return std::nullopt;
    }

    // addMachineReg sets DwarfRegs and SubRegister{Size,Offset}InBits. Collect
    // them here and reset the fields to avoid hitting any asserts.
    decltype(DwarfRegs) Regs;
    std::swap(Regs, DwarfRegs);
    unsigned SubRegOffset = SubRegisterOffsetInBits;
    unsigned SubRegSize = SubRegisterSizeInBits;
    SubRegisterOffsetInBits = SubRegisterSizeInBits = 0;
    if (SubRegOffset % 8 || SubRegSize % 8)
      return std::nullopt;
    SubRegOffset /= 8;
    SubRegSize /= 8;

    auto focusThreadIfRequired = [this](int64_t DwarfRegNo) {
      // FIXME: This should be represented in the DIExpression.
      if (auto LaneSize = TRI->getDwarfRegLaneSize(DwarfRegNo, false)) {
        emitUserOp(dwarf::DW_OP_LLVM_push_lane);
        emitConstu(*LaneSize);
        emitOp(dwarf::DW_OP_mul);
        emitUserOp(dwarf::DW_OP_LLVM_offset);
      }
    };

    if (Regs.size() == 1) {
      addReg(Regs[0].DwarfRegNo, Regs[0].Comment);
      focusThreadIfRequired(Regs[0].DwarfRegNo);

      if (SubRegOffset) {
        emitUserOp(dwarf::DW_OP_LLVM_offset_uconst);
        emitUnsigned(SubRegOffset);
      }

      // Ignore SubRegSize, no correct consumer can read or write past the end
      // of the subregister location.

      return NewOpResult{Arg.getResultType(), ValueKind::LocationDesc};
    }

    assert(SubRegOffset == 0 && SubRegSize == 0 &&
           "register piece cannot apply to multiple registers");

    // When emitting fragments, the top element on the stack might be an
    // incomplete composite. Push/drop a lit0 so that we don't add the registers
    // to the larger composite.
    if (IsFragment)
      emitOp(dwarf::DW_OP_lit0);

    for (auto &Reg : Regs) {
      if (Reg.SubRegSize % 8)
        return std::nullopt;
      if (Reg.DwarfRegNo >= 0) {
        addReg(Reg.DwarfRegNo, Reg.Comment);
        focusThreadIfRequired(Regs[0].DwarfRegNo);
      }
      emitOp(dwarf::DW_OP_piece);
      emitUnsigned(Reg.SubRegSize / 8);
    }
    emitUserOp(dwarf::DW_OP_LLVM_piece_end);

    if (IsFragment) {
      emitOp(dwarf::DW_OP_swap);
      emitOp(dwarf::DW_OP_drop);
    }

    return NewOpResult{Arg.getResultType(), ValueKind::LocationDesc};
  }

  if (Entry.isInt()) {
    emitConstu(Entry.getInt());
  } else if (Entry.isConstantFP()) {
    // DwarfExpression does not support arguments wider than 64 bits
    // (see PR52584).
    // TODO: Consider chunking expressions containing overly wide
    // arguments into separate pointer-sized fragment expressions.
    APInt RawBytes = Entry.getConstantFP()->getValueAPF().bitcastToAPInt();
    if (RawBytes.getBitWidth() > 64)
      return std::nullopt;
    emitConstu(RawBytes.getZExtValue());
  } else if (Entry.isConstantInt()) {
    APInt RawBytes = Entry.getConstantInt()->getValue();
    if (RawBytes.getBitWidth() > 64)
      return std::nullopt;
    emitConstu(RawBytes.getZExtValue());
  } else if (Entry.isTargetIndexLocation()) {
    return std::nullopt;
  } else {
    llvm_unreachable("Unsupported Entry type.");
  }

  return NewOpResult{Arg.getResultType(), ValueKind::Value};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Constant Constant,
                                                     ChildrenT Children) {
  ConstantData *LiteralValue = Constant.getLiteralValue();

  // FIXME: Support ConstantFP?
  ConstantInt *IntLiteralValue = dyn_cast<ConstantInt>(LiteralValue);
  if (!IntLiteralValue)
    return std::nullopt;

  if (isUnsigned(IntLiteralValue)) {
    emitConstu(IntLiteralValue->getZExtValue());
  } else {
    emitOp(dwarf::DW_OP_consts);
    emitSigned(IntLiteralValue->getSExtValue());
  }

  return NewOpResult{IntLiteralValue->getType(), ValueKind::Value};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::PushLane PushLane,
                                                     ChildrenT Children) {
  return std::nullopt;
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Referrer ReferrerOp,
                                                     ChildrenT Children) {
  return std::nullopt;
}

std::optional<NewOpResult>
DwarfExpression::traverse(DIOp::TypeObject TypeObject, ChildrenT Children) {
  return std::nullopt;
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::AddrOf AddrOf,
                                                     ChildrenT Children) {
  return std::nullopt;
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Convert Convert,
                                                     ChildrenT Children) {
  auto Child = traverse(Children[0].get(), /*RequiredVK=*/std::nullopt,
                        /*PermitDivergentAddrSpace=*/true);
  if (!Child)
    return std::nullopt;

  Type *DestTy = Convert.getResultType();
  if (Child->Ty->isPointerTy() && DestTy->isPointerTy() &&
      Child->Ty->getPointerAddressSpace() != DestTy->getPointerAddressSpace()) {
    unsigned DivAddrSpace = Child->DivergentAddrSpace
                                ? *Child->DivergentAddrSpace
                                : Child->Ty->getPointerAddressSpace();
    return NewOpResult{DestTy, Child->VK, DivAddrSpace};
  }

  if (!Child->Ty->isIntegerTy() || !DestTy->isIntegerTy())
    return std::nullopt;

  // If we're not dealing with the divergent address space case, Convert
  // requires a value operand.
  if (Child->VK == ValueKind::LocationDesc)
    readToValue(Child->Ty);

  uint64_t ToBits = DestTy->getPrimitiveSizeInBits().getFixedValue();
  uint64_t FromBits = Child->Ty->getPrimitiveSizeInBits().getFixedValue();

  if (ToBits < FromBits) {
    // This function is called "ZExt", but it's actually doing a truncation on
    // generic types (operation is "Child & ((1u << ToBits) - 1)").
    emitLegacyZExt(ToBits);
  }
  return NewOpResult{DestTy, ValueKind::Value};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::ZExt ZExt,
                                                     ChildrenT Children) {
  auto Child = traverse(Children[0].get(), ValueKind::Value);
  if (!Child || !Child->Ty->isIntegerTy())
    return std::nullopt;

  uint64_t FromBits = Child->Ty->getPrimitiveSizeInBits().getFixedValue();
  emitLegacyZExt(FromBits);
  return NewOpResult{ZExt.getResultType(), ValueKind::Value};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::SExt SExt,
                                                     ChildrenT Children) {
  auto Child = traverse(Children[0].get(), ValueKind::Value);
  if (!Child || !Child->Ty->isIntegerTy())
    return std::nullopt;

  uint64_t FromBits = Child->Ty->getPrimitiveSizeInBits().getFixedValue();
  emitLegacySExt(FromBits);
  return NewOpResult{SExt.getResultType(), ValueKind::Value};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Deref Deref,
                                                     ChildrenT Children) {
  auto Child = traverse(Children[0].get(), ValueKind::LocationDesc,
                        /*PermitDivergentAddrSpace=*/true);
  if (!Child)
    return std::nullopt;

  // FIXME(KZHURAVL): Support non pointer types?
  if (!Child->Ty->isPointerTy())
    return std::nullopt;

  PointerType *PointerResultType = dyn_cast<PointerType>(Child->Ty);
  assert(PointerResultType && "Expected PointerType, but got something else");

  unsigned PointerLLVMAddrSpace = Child->DivergentAddrSpace
                                      ? *Child->DivergentAddrSpace
                                      : PointerResultType->getAddressSpace();
  uint64_t PointerSizeInBits =
      AP.getDataLayout().getPointerSizeInBits(PointerLLVMAddrSpace);
  assert(PointerSizeInBits % 8 == 0 && "Expected multiple of 8");

  uint64_t PointerSizeInBytes = PointerSizeInBits / 8;
  auto PointerDWARFAddrSpace = AP.TM.mapToDWARFAddrSpace(PointerLLVMAddrSpace);
  if (!PointerDWARFAddrSpace) {
    LLVM_DEBUG(dbgs() << "Failed to lower DIOpDeref of pointer to addrspace("
                      << PointerLLVMAddrSpace
                      << "): no corresponding DWARF addrspace.\n");
    return std::nullopt;
  }

  emitOp(dwarf::DW_OP_deref_size);
  emitData1(PointerSizeInBytes);
  emitConstu(*PointerDWARFAddrSpace);
  emitUserOp(dwarf::DW_OP_LLVM_form_aspace_address);

  // FIXME(KZHURAVL): Is the following result type correct?
  return NewOpResult{Deref.getResultType(), ValueKind::LocationDesc};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Extend Extend,
                                                     ChildrenT Children) {
  return std::nullopt;
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Read Read,
                                                     ChildrenT Children) {
  auto Child = traverse(Children[0].get(), ValueKind::LocationDesc);
  if (!Child)
    return std::nullopt;
  readToValue(Children[0].get());
  return NewOpResult{Child->Ty, ValueKind::Value};
}

std::optional<NewOpResult>
DwarfExpression::traverse(DIOp::Reinterpret Reinterpret, ChildrenT Children) {
  auto Child = traverse(Children[0].get(), ValueKind::LocationDesc,
                        /*PermitDivergentAddrSpace=*/true);
  if (!Child)
    return Child;
  return NewOpResult{Reinterpret.getResultType(), Child->VK,
                     Child->DivergentAddrSpace};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Select Select,
                                                     ChildrenT Children) {
  return std::nullopt;
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Composite Composite,
                                                     ChildrenT Children) {
  if (IsFragment)
    emitOp(dwarf::DW_OP_lit0);

  for (auto &Child : Children) {
    auto R = traverse(Child.get(), std::nullopt);
    if (!R)
      return std::nullopt;
    TypeSize Size = R->Ty->getPrimitiveSizeInBits();
    if (!Size.isFixed() || Size.getFixedValue() % 8 != 0)
      return std::nullopt;
    emitOp(dwarf::DW_OP_piece);
    emitUnsigned(Size.getFixedValue() / 8);
  }
  emitUserOp(dwarf::DW_OP_LLVM_piece_end);

  if (IsFragment) {
    emitOp(dwarf::DW_OP_swap);
    emitOp(dwarf::DW_OP_drop);
  }

  return NewOpResult{Composite.getResultType(), ValueKind::LocationDesc};
}

std::optional<NewOpResult>
DwarfExpression::traverseMathOp(uint8_t DwarfOp, ChildrenT Children) {
  auto LHS = traverse(Children[0].get(), ValueKind::Value);
  if (!LHS)
    return std::nullopt;
  auto RHS = traverse(Children[1].get(), ValueKind::Value);
  if (!RHS)
    return std::nullopt;

  emitOp(DwarfOp);
  return NewOpResult{LHS->Ty, ValueKind::Value};
}

std::optional<NewOpResult>
DwarfExpression::traverse(DIOp::ByteOffset ByteOffset, ChildrenT Children) {
  auto LHS = traverse(Children[0].get(), ValueKind::LocationDesc);
  if (!LHS)
    return std::nullopt;
  auto RHS = traverse(Children[1].get(), ValueKind::Value);
  if (!RHS)
    return std::nullopt;

  emitUserOp(dwarf::DW_OP_LLVM_offset);
  return NewOpResult{ByteOffset.getResultType(), ValueKind::LocationDesc};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::BitOffset BitOffset,
                                                     ChildrenT Children) {
  auto LHS = traverse(Children[0].get(), ValueKind::LocationDesc);
  if (!LHS)
    return std::nullopt;
  auto RHS = traverse(Children[1].get(), ValueKind::Value);
  if (!RHS)
    return std::nullopt;

  emitUserOp(dwarf::DW_OP_LLVM_bit_offset);
  return NewOpResult{BitOffset.getResultType(), ValueKind::LocationDesc};
}

std::optional<NewOpResult> DwarfExpression::traverse(DIOp::Fragment Fragment,
                                                     ChildrenT Children) {
  llvm_unreachable("should have dropped fragments by now");
  return std::nullopt;
}

void DwarfExpression::readToValue(Type *Ty) {
  uint64_t PrimitiveSizeInBits = Ty->getPrimitiveSizeInBits();
  assert(PrimitiveSizeInBits != 0 && "Expected primitive type");

  uint64_t ByteAlignedPrimitiveSizeInBits = alignTo<8>(PrimitiveSizeInBits);
  uint64_t PrimitiveSizeInBytes = ByteAlignedPrimitiveSizeInBits / 8;
  bool NeedsMask = ByteAlignedPrimitiveSizeInBits != PrimitiveSizeInBits;

  emitOp(dwarf::DW_OP_deref_size);
  emitData1(PrimitiveSizeInBytes);

  if (NeedsMask) {
    uint64_t Mask = (1ULL << PrimitiveSizeInBits) - 1ULL;
    emitConstu(Mask);
    emitOp(dwarf::DW_OP_and);
  }
}

void DwarfExpression::readToValue(DwarfExpression::Node *OpNode) {
  assert(OpNode->isLowered() && "Expected lowered node");
  assert(OpNode->getResultType() && "Expected non-null result type");
  readToValue(OpNode->getResultType());
}
