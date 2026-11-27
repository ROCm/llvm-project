//===- SQTTAutomatic.cpp - Automatic SQTT instrumentation -----------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements automatic barrier and memory-operation instrumentation.
///
//===----------------------------------------------------------------------===//

#include "SQTTPass.h"

using namespace llvm;

SQTTInstrumentPass::BarrierKind
SQTTInstrumentPass::classifyBarrier(CallInst *CI) {
  if (!CI)
    return BarrierKind::None;
  Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return BarrierKind::None;
  switch (Callee->getIntrinsicID()) {
  case Intrinsic::amdgcn_s_barrier_signal:
  case Intrinsic::amdgcn_s_barrier_signal_var:
  case Intrinsic::amdgcn_s_barrier_signal_isfirst:
    return BarrierKind::Signal;
  case Intrinsic::amdgcn_s_barrier_wait:
    return BarrierKind::Wait;
  case Intrinsic::amdgcn_s_barrier:
    return BarrierKind::Full;
  default:
    return BarrierKind::None;
  }
}

bool SQTTInstrumentPass::instrumentBarriers(Function &F, GfxGen Gen) {
  // Snapshot all insertion points before changing the CFG.
  SmallVector<std::pair<Instruction *, uint32_t>, 8> Insertions;
  auto MarkerId = [this](BarrierKind Kind) {
    return Kind == BarrierKind::None
               ? 0
               : FirstBarrierID + static_cast<uint32_t>(Kind);
  };
  for (BasicBlock &BB : F) {
    CallInst *Signal = nullptr;
    for (Instruction &I : BB) {
      auto *Call = dyn_cast<CallInst>(&I);
      BarrierKind Kind = classifyBarrier(Call);
      if (Signal &&
          (Signal->getNextNode() != &I || Kind != BarrierKind::Wait)) {
        Insertions.push_back(
            {Signal->getNextNode(), MarkerId(BarrierKind::Signal)});
        Signal = nullptr;
      }
      if (Kind == BarrierKind::Signal)
        Signal = Call;
      else if (Kind == BarrierKind::Wait && Signal) {
        Insertions.push_back({Call, MarkerId(BarrierKind::Full)});
        Signal = nullptr;
      } else if (Kind == BarrierKind::Wait || Kind == BarrierKind::Full)
        Insertions.push_back({Call, MarkerId(Kind)});
    }
    if (Signal)
      Insertions.push_back(
          {Signal->getNextNode(), MarkerId(BarrierKind::Signal)});
  }
  if (Insertions.empty())
    return false;
  for (auto [Before, MarkerId] : Insertions) {
    IRBuilder<> B(Before);
    insertTraceMarker(B, encodeMarker(MarkerId, false, false), F, Gen);
  }
  return true;
}

SQTTInstrumentPass::MemOpKind
SQTTInstrumentPass::classifyMemOp(Instruction *I) {
  if (Value *Pointer = getMemoryPointer(I)) {
    unsigned AS = cast<PointerType>(Pointer->getType())->getAddressSpace();
    // Atomics are read-modify-write, so use store markers.
    if (AS == 3 || AS == 5)
      return MemOpKind::None;
    return isa<LoadInst>(I) ? MemOpKind::Load : MemOpKind::Store;
  }
  auto *CI = dyn_cast<CallInst>(I);
  Function *Callee = CI ? CI->getCalledFunction() : nullptr;
  if (!Callee)
    return MemOpKind::None;
  BufferOpKind Kind = classifyBufferOp(Callee->getName());
  return Kind == BufferOpKind::Load   ? MemOpKind::Load
         : Kind == BufferOpKind::None ? MemOpKind::None
                                      : MemOpKind::Store;
}

bool SQTTInstrumentPass::instrumentMemoryOps(Function &F, GfxGen Gen) {
  // Snapshot first: inserting a trace changes the instruction stream being
  // chunked.
  SmallVector<std::pair<Instruction *, uint32_t>, 16> Insertions;
  auto MarkerId = [this](MemOpKind Kind) {
    return Kind == MemOpKind::None ? 0
                                   : FirstVmemID + static_cast<uint32_t>(Kind);
  };
  for (BasicBlock &BB : F) {
    MemOpKind RunKind = MemOpKind::None;
    Instruction *LastOp = nullptr;
    unsigned RunSize = 0, Gap = 0;
    auto Flush = [&] {
      if (RunSize)
        Insertions.push_back({LastOp, MarkerId(RunKind)});
      RunSize = 0;
    };
    for (Instruction &I : BB) {
      MemOpKind Kind = classifyMemOp(&I);
      if (Kind == MemOpKind::None) {
        Gap += LastOp != nullptr;
        continue;
      }
      if (Kind != RunKind || Gap > Config.MemoryMaxGap)
        Flush();

      RunKind = Kind;
      LastOp = &I;
      Gap = 0;
      if (++RunSize == Config.MemoryChunkSize)
        Flush();
    }
    Flush();
  }
  if (Insertions.empty())
    return false;

  for (auto [LastOp, MarkerId] : Insertions) {
    IRBuilder<> B(LastOp->getNextNode());
    insertTraceMarker(B, encodeMarker(MarkerId, false, false), F, Gen);
  }
  return true;
}
