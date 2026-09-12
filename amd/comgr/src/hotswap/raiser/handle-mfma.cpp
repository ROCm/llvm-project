//===- handle-mfma.cpp - Hotswap MFMA raising ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/handlers.h"

#include "hotswap/decoder/amdgpu-mc-tables.h"
#include "hotswap/decoder/canonical-op.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace COMGR::hotswap {
namespace {

Expected<ParsedReg> requireSourceReg(RaiseContext &Ctx, const DecodedInst &Di,
                                     OperandResolver &Op, unsigned Index) {
  Expected<std::optional<ParsedReg>> Source = Op.srcReg(Index);
  if (!Source)
    return Source.takeError();
  if (!*Source)
    return unsupported(Ctx, Di, "MFMA source must be a register");
  return **Source;
}

Expected<int64_t> readNamedImmediate(RaiseContext &Ctx, const DecodedInst &Di,
                                     AMDGPU::OpName Name) {
  int Index = COMGR::hotswap::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Index < 0)
    return int64_t(0);
  if (!Di.isImm(static_cast<unsigned>(Index)))
    return unsupported(Ctx, Di, "MFMA modifier must be immediate");
  return Di.getImm(static_cast<unsigned>(Index));
}

} // namespace

Error handleMFMA(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  if (!Ctx.Projection.TargetSTI.hasFeature(AMDGPU::FeatureMAIInsts))
    return unsupported(Ctx, Di, "target ISA does not support MFMA");

  Intrinsic::ID IntrinsicId;
  Type *InputTy;
  Type *AccumulatorTy;
  switch (Di.CanonOp) {
  case CanonicalOp::V_MFMA_F32_16x16x16_F16:
    IntrinsicId = Intrinsic::amdgcn_mfma_f32_16x16x16f16;
    InputTy = FixedVectorType::get(Ctx.B.getHalfTy(), 4);
    AccumulatorTy = FixedVectorType::get(Ctx.B.getFloatTy(), 4);
    break;
  case CanonicalOp::V_MFMA_F32_16x16x16_BF16_1K:
    IntrinsicId = Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k;
    InputTy = FixedVectorType::get(Ctx.B.getInt16Ty(), 4);
    AccumulatorTy = FixedVectorType::get(Ctx.B.getFloatTy(), 4);
    break;
  case CanonicalOp::V_MFMA_I32_16x16x32_I8:
    IntrinsicId = Intrinsic::amdgcn_mfma_i32_16x16x32_i8;
    InputTy = Ctx.B.getInt64Ty();
    AccumulatorTy = FixedVectorType::get(Ctx.B.getInt32Ty(), 4);
    break;
  default:
    return unsupported(Ctx, Di, "MFMA opcode has no intrinsic mapping");
  }

  if (Op.nSrcs() < 3)
    return unsupported(Ctx, Di, "MFMA requires three source operands");
  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<ParsedReg> SourceA = requireSourceReg(Ctx, Di, Op, 0);
  if (!SourceA)
    return SourceA.takeError();
  Expected<ParsedReg> SourceB = requireSourceReg(Ctx, Di, Op, 1);
  if (!SourceB)
    return SourceB.takeError();
  Expected<ParsedReg> SourceC = requireSourceReg(Ctx, Di, Op, 2);
  if (!SourceC)
    return SourceC.takeError();

  Expected<int64_t> Cbsz = readNamedImmediate(Ctx, Di, AMDGPU::OpName::cbsz);
  if (!Cbsz)
    return Cbsz.takeError();
  Expected<int64_t> Abid = readNamedImmediate(Ctx, Di, AMDGPU::OpName::abid);
  if (!Abid)
    return Abid.takeError();
  Expected<int64_t> Blgp = readNamedImmediate(Ctx, Di, AMDGPU::OpName::blgp);
  if (!Blgp)
    return Blgp.takeError();

  AllocaRegFile &Registers = Ctx.registers().regFile();
  Value *A = Registers.readRegVec(Ctx.B, *SourceA, InputTy);
  Value *B = Registers.readRegVec(Ctx.B, *SourceB, InputTy);
  Value *C = Registers.readRegVec(Ctx.B, *SourceC, AccumulatorTy);
  Module *M = Ctx.B.GetInsertBlock()->getModule();
  Function *Fn = Intrinsic::getOrInsertDeclaration(M, IntrinsicId);
  Value *CbszValue = Ctx.B.getInt32(*Cbsz);
  Value *AbidValue = Ctx.B.getInt32(*Abid);
  Value *BlgpValue = Ctx.B.getInt32(*Blgp);
  Value *Result =
      Ctx.B.CreateCall(Fn, {A, B, C, CbszValue, AbidValue, BlgpValue}, "mfma");
  Ctx.registers().writeRegVec(*Destination, Result);
  return Error::success();
}

} // namespace COMGR::hotswap
