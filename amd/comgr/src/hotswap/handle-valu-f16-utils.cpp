//===- handle-valu-f16-utils.cpp - F16 VALU helpers -----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-f16-utils.h"

#include "SIDefines.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace COMGR::hotswap {

// VOP3 f16 source modifiers carry both arithmetic modifiers and half-register
// selection:
//   bit 0: source neg
//   bit 1: source abs
//   bit 2: source op_sel (0 = low 16 bits, 1 = high 16 bits)
// For VOP3_t16, src0_modifiers bit 3 is also the destination op_sel. It
// selects which 16-bit half of the destination VGPR receives the result; the
// other half must be preserved by explicitly merging with the old destination
// dword. Missing or non-immediate modifier operands are treated as TableGen
// layout drift and refused rather than defaulting to low-half semantics.
bool readRequiredVOP3F16SrcMods(const DecodedInst &Di, HandlerResult &Hr,
                                unsigned SrcIndex, StringRef OpName,
                                unsigned &Mods) {
  if (SrcIndex >= Di.NumSrcs) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3",
        (Twine(OpName) + " missing f16 source operand").str());
    return false;
  }

  unsigned ModIdx = Di.ModMap[SrcIndex];
  if (ModIdx == UINT_MAX || !Di.isImm(ModIdx)) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3",
        (Twine(OpName) + " missing immediate f16 src" + Twine(SrcIndex) +
         "_modifiers operand; operand table layout does not match the "
         "expected VOP3 f16 profile")
            .str());
    return false;
  }

  int64_t Raw = Di.getImm(ModIdx);
  const unsigned Allowed = SrcIndex == 0 ? 0xFu : 0x7u;
  if (Raw < 0 || (static_cast<unsigned>(Raw) & ~Allowed) != 0) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3",
        (Twine(OpName) + " has unsupported f16 src" + Twine(SrcIndex) +
         "_modifiers bits")
            .str());
    return false;
  }

  Mods = static_cast<unsigned>(Raw);
  return true;
}

bool readOptionalVOP3F16SrcMods(const DecodedInst &Di, HandlerResult &Hr,
                                unsigned SrcIndex, StringRef OpName,
                                unsigned &Mods) {
  Mods = 0;
  if (SrcIndex >= Di.NumSrcs) {
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3",
        (Twine(OpName) + " missing f16 source operand").str());
    return false;
  }

  unsigned ModIdx = Di.ModMap[SrcIndex];
  if (ModIdx == UINT_MAX)
    return true;
  return readRequiredVOP3F16SrcMods(Di, Hr, SrcIndex, OpName, Mods);
}

Value *readOpSelF16FromMods(RaiseContext &Ctx, OpResolver &Op,
                            unsigned SrcIndex, unsigned Mods) {
  // Modifier layout for decoded VOP3 F16 operands:
  //   NEG       - negate the selected F16 value
  //   ABS       - take fabs before negation, matching AMDGPU source-mod order
  //   OP_SEL_0  - select bits [31:16] instead of bits [15:0]
  //
  // The containing register is still read as an i32 because the register file
  // stores VGPR dwords. We select the requested half, bitcast those 16 bits to
  // `half`, then apply arithmetic source modifiers.
  Type *I16Ty = Type::getInt16Ty(Ctx.C);
  Value *Raw = Op.src(SrcIndex);
  if ((Mods & SISrcMods::OP_SEL_0) != 0)
    Raw = Ctx.B.CreateLShr(Raw, 16, "f16_src_hi");
  Value *Bits = Ctx.B.CreateTrunc(Raw, I16Ty);
  Value *V = Ctx.B.CreateBitCast(Bits, Ctx.F16Ty);
  if (Mods & SISrcMods::ABS)
    V = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, V, nullptr, "abs_f16");
  if (Mods & SISrcMods::NEG)
    V = Ctx.B.CreateFNeg(V, "neg_f16");
  return V;
}

Value *readOpSelF16(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op,
                    HandlerResult &Hr, unsigned SrcIndex, StringRef OpName) {
  unsigned Mods = 0;
  if (!readRequiredVOP3F16SrcMods(Di, Hr, SrcIndex, OpName, Mods))
    return nullptr;
  return readOpSelF16FromMods(Ctx, Op, SrcIndex, Mods);
}

Value *readOptionalOpSelF16(RaiseContext &Ctx, const DecodedInst &Di,
                            OpResolver &Op, HandlerResult &Hr,
                            unsigned SrcIndex, StringRef OpName) {
  unsigned Mods = 0;
  if (!readOptionalVOP3F16SrcMods(Di, Hr, SrcIndex, OpName, Mods))
    return nullptr;
  return readOpSelF16FromMods(Ctx, Op, SrcIndex, Mods);
}

bool readVOP3F16DstHigh(const DecodedInst &Di, HandlerResult &Hr,
                        StringRef OpName, bool &DstHigh) {
  unsigned Mods = 0;
  if (!readRequiredVOP3F16SrcMods(Di, Hr, 0, OpName, Mods))
    return false;
  DstHigh = (Mods & SISrcMods::DST_OP_SEL) != 0;
  return true;
}

void writeOpSelF16(RaiseContext &Ctx, OpResolver &Op, Value *Result,
                   bool DstHigh, StringRef MergeLoName,
                   StringRef MergeHiName) {
  // True16 instructions write one 16-bit lane of a VGPR dword. LLVM IR has no
  // partial-register write, so model it explicitly as a read/modify/write:
  // preserve the old opposite half and OR in the new result bits.
  Type *I16Ty = Type::getInt16Ty(Ctx.C);
  Value *Bits =
      Ctx.B.CreateZExt(Ctx.B.CreateBitCast(Result, I16Ty), Ctx.I32Ty);
  Value *Old = Ctx.Regs.readReg32(Ctx.B, Op.dst());
  if (!DstHigh) {
    Value *High =
        Ctx.B.CreateAnd(Old, ConstantInt::get(Ctx.I32Ty, 0xFFFF0000u));
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(High, Bits, MergeLoName));
    return;
  }

  Value *Low = Ctx.B.CreateAnd(Old, ConstantInt::get(Ctx.I32Ty, 0x0000FFFFu));
  Value *Shifted = Ctx.B.CreateShl(Bits, 16);
  Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Low, Shifted, MergeHiName));
}

} // namespace COMGR::hotswap
