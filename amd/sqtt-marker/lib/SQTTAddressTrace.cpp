//===- SQTTAddressTrace.cpp - SQTT address tracing ------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements per-lane address trace instrumentation and payload framing.
///
//===----------------------------------------------------------------------===//

#include "SQTTPass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

constexpr StringLiteral MemoryTraceNames[] = {
    "addr_trace_load", "addr_trace_store", "addr_trace_atomic"};
constexpr StringLiteral LDSTraceNames[] = {
    "addr_trace_lds_load", "addr_trace_lds_store", "addr_trace_lds_atomic"};
constexpr StringLiteral BufferTraceNames[][3] = {
    {"addr_trace_buffer_load", "addr_trace_buffer_store",
     "addr_trace_buffer_atomic"},
    {"addr_trace_struct_buffer_load", "addr_trace_struct_buffer_store",
     "addr_trace_struct_buffer_atomic"}};

static constexpr char ExecTraceAsm[] = "s_mov_b32 m0, exec_lo\n"
                                       "s_nop 0\n"
                                       "s_ttracedata\n"
                                       "s_mov_b32 m0, exec_hi\n"
                                       "s_nop 0\n"
                                       "s_ttracedata";

} // namespace

static unsigned traceOperationIndex(bool IsStore, bool IsAtomic) {
  return IsAtomic ? 2 : IsStore;
}

static void emitExecMaskTraces(IRBuilder<> &B) {
  LLVMContext &Ctx = B.getContext();
  B.CreateCall(InlineAsm::get(FunctionType::get(Type::getInt32Ty(Ctx), false),
                              ExecTraceAsm, "={m0}", /*hasSideEffects=*/true));
}

SQTTInstrumentPass::AddrTraceOp
SQTTInstrumentPass::classifyAddrTraceOp(Instruction *I, bool TraceMemory,
                                        bool TraceLds) {
  const auto None = [=] {
    return AddrTraceOp{I, {}, AddrTraceKind::None, 0, false};
  };
  if (Value *Pointer = getMemoryPointer(I)) {
    // The memory address protocol is defined only for flat (0) and
    // global (1) pointers.  Other AMDGPU address spaces have different
    // representations and must not be reinterpreted as global addresses.
    unsigned AS = cast<PointerType>(Pointer->getType())->getAddressSpace();
    bool IsStore = isa<StoreInst>(I);
    unsigned Op = traceOperationIndex(IsStore, isa<AtomicRMWInst>(I) ||
                                                   isa<AtomicCmpXchgInst>(I));
    if (AS == 3 && TraceLds)
      return {I, LDSTraceNames[Op], AddrTraceKind::LDS, 0, false};
    if ((AS == 0 || AS == 1) && TraceMemory)
      return {I, MemoryTraceNames[Op], AddrTraceKind::Memory, 0, false};
    return None();
  }

  auto *CI = dyn_cast<CallInst>(I);
  Function *Callee = CI ? CI->getCalledFunction() : nullptr;
  if (!Callee)
    return None();

  StringRef Name = Callee->getName();
  BufferOpKind BufferKind = classifyBufferOp(Name);
  // buffer.load.lds has a distinct operand layout (including an LDS
  // destination pointer), so the ordinary buffer component protocol is not
  // valid for it.
  if (TraceMemory && BufferKind != BufferOpKind::None &&
      !Name.ends_with(".buffer.load.lds")) {
    bool IsStruct = isStructBuffer(Name);
    unsigned Op = static_cast<unsigned>(BufferKind) - 1;
    unsigned Rsrc = BufferKind == BufferOpKind::Load ? 0
                    : isBufferCmpSwap(Name)          ? 2
                                                     : 1;
    return {I, BufferTraceNames[IsStruct][Op], AddrTraceKind::Buffer, Rsrc,
            IsStruct};
  }

  Intrinsic::ID IID = Callee->getIntrinsicID();
  if (TraceLds && (IID == Intrinsic::amdgcn_ds_permute ||
                   IID == Intrinsic::amdgcn_ds_bpermute ||
                   IID == Intrinsic::amdgcn_ds_bpermute_fi_b32)) {
    bool IsBPermute = IID == Intrinsic::amdgcn_ds_bpermute ||
                      IID == Intrinsic::amdgcn_ds_bpermute_fi_b32;
    return {I, IsBPermute ? "addr_trace_ds_bpermute" : "addr_trace_ds_permute",
            AddrTraceKind::Permute, 0, false};
  }
  return None();
}

std::string SQTTInstrumentPass::getSourceLoc(Instruction *I) {
  const DebugLoc &DL = I->getDebugLoc();
  if (!DL)
    return "";

  // Walk the inline chain innermost -> outermost.  At each level, getScope()
  // gives the file the source line lives in; getLine() gives the line.
  // getInlinedAt() walks one step outward (the call site).  Format matches
  // rocprofiler-sdk codeobj's printer: "<inner>:<line> -> <outer>:<line>".
  std::string Out;
  DILocation *L = DL.get();
  while (L) {
    if (!Out.empty())
      Out += " -> ";
    if (auto *Scope = L->getScope())
      Out += Scope->getFilename().str();
    Out += ':';
    Out += std::to_string(L->getLine());
    L = L->getInlinedAt();
  }
  return Out;
}

std::string SQTTInstrumentPass::getFunctionSourceLoc(Function &F) {
  DISubprogram *SP = F.getSubprogram();
  if (!SP)
    return "";
  StringRef File = SP->getFilename();
  unsigned Line = SP->getLine();
  if (File.empty() && Line == 0)
    return "";
  std::string Out = File.str();
  Out += ':';
  Out += std::to_string(Line);
  return Out;
}

void SQTTInstrumentPass::emitAddressTrace(IRBuilder<> &B, const AddrTraceOp &Op,
                                          uint32_t HeaderId, GfxGen Gen) {
  Module *M = B.GetInsertBlock()->getParent()->getParent();
  LLVMContext &Ctx = M->getContext();
  Type *I32 = Type::getInt32Ty(Ctx);

  bool SchedBarrier = !Config.needsScopeCheck();
  emitTraceBoundary(B, /*after=*/false, SchedBarrier);
  emitBareTrace(B, encodeMarker(HeaderId, false, false), M, Gen);
  unsigned WaveSize = getWaveSize(Gen);

  switch (Op.Kind) {
  case AddrTraceKind::Buffer: {
    CallInst *BufOp = cast<CallInst>(Op.I);
    Value *Rsrc = BufOp->getArgOperand(Op.BufferRsrcIndex);
    Value *Vindex = Op.StructBuffer
                        ? BufOp->getArgOperand(Op.BufferRsrcIndex + 1)
                        : nullptr;
    Value *Voffset =
        BufOp->getArgOperand(Op.BufferRsrcIndex + (Op.StructBuffer ? 2 : 1));
    Value *Soffset =
        BufOp->getArgOperand(Op.BufferRsrcIndex + (Op.StructBuffer ? 3 : 2));

    // Header, EXEC, descriptor words, scalar offset, then lane data.
    emitExecMaskTraces(B);
    Value *RsrcLo, *RsrcHi;
    if (Rsrc->getType()->isVectorTy()) {
      RsrcLo = B.CreateExtractElement(Rsrc, uint64_t{0});
      RsrcHi = B.CreateExtractElement(Rsrc, uint64_t{1});
    } else {
      Value *RsrcInt = B.CreatePtrToInt(Rsrc, Type::getIntNTy(Ctx, 128));
      RsrcLo = B.CreateTrunc(RsrcInt, I32);
      RsrcHi = B.CreateTrunc(B.CreateLShr(RsrcInt, 32), I32);
    }
    emitBareTraceValue(B, RsrcLo, M);
    emitBareTraceValue(B, RsrcHi, M);
    if (Soffset->getType() != I32)
      Soffset = B.CreateZExtOrTrunc(Soffset, I32);
    emitBareTraceValue(B, Soffset, M);
    emitReadlaneTraceLoop(B, Voffset, nullptr, WaveSize);
    if (Vindex)
      emitReadlaneTraceLoop(B, Vindex, nullptr, WaveSize);
    break;
  }
  case AddrTraceKind::Permute:
    emitExecMaskTraces(B);
    emitReadlaneTraceLoop(B, cast<CallInst>(Op.I)->getArgOperand(0), nullptr,
                          WaveSize);
    break;
  case AddrTraceKind::Memory:
  case AddrTraceKind::LDS: {
    Value *Ptr = getMemoryPointer(Op.I);
    assert(Ptr && "expected Load/Store/Atomic instruction");
    Value *AddrLo, *AddrHi = nullptr;
    if (Op.Kind == AddrTraceKind::Memory) {
      Value *Addr = B.CreatePtrToInt(Ptr, Type::getInt64Ty(Ctx));
      AddrLo = B.CreateTrunc(Addr, I32);
      AddrHi = B.CreateTrunc(B.CreateLShr(Addr, 32), I32);
    } else
      AddrLo = B.CreatePtrToInt(Ptr, I32);
    emitExecMaskTraces(B);
    emitReadlaneTraceLoop(B, AddrLo, AddrHi, WaveSize);
    break;
  }
  default:
    llvm_unreachable("unsupported address trace kind");
  }

  emitTraceBoundary(B, /*after=*/true, SchedBarrier);
}

void SQTTInstrumentPass::emitReadlaneTraceLoop(IRBuilder<> &B,
                                               Value *FirstValue,
                                               Value *SecondValue,
                                               unsigned WaveSize) {
  Function &F = *B.GetInsertBlock()->getParent();
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  if (FirstValue->getType() != I32)
    FirstValue = B.CreateZExtOrTrunc(FirstValue, I32);
  if (SecondValue && SecondValue->getType() != I32)
    SecondValue = B.CreateZExtOrTrunc(SecondValue, I32);

  Function *ReadLane =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_readlane, {I32});
  BasicBlock *PreheaderBB = B.GetInsertBlock();
  BasicBlock *AfterBB;
  if (B.GetInsertPoint() == PreheaderBB->end())
    AfterBB = BasicBlock::Create(Ctx, "sqtt.lanes.after", &F,
                                 PreheaderBB->getNextNode());
  else {
    AfterBB =
        PreheaderBB->splitBasicBlock(B.GetInsertPoint(), "sqtt.lanes.after");
    PreheaderBB->getTerminator()->eraseFromParent();
  }
  BasicBlock *LoopBB = BasicBlock::Create(Ctx, "sqtt.lanes.loop", &F, AfterBB);

  IRBuilder<> PreB(PreheaderBB);
  PreB.CreateBr(LoopBB);

  IRBuilder<> LoopB(LoopBB);
  PHINode *Lane = LoopB.CreatePHI(I32, 2, "lane");
  Lane->addIncoming(ConstantInt::get(I32, 0), PreheaderBB);
  emitBareTraceValue(LoopB, LoopB.CreateCall(ReadLane, {FirstValue, Lane}), M);
  if (SecondValue)
    emitBareTraceValue(LoopB, LoopB.CreateCall(ReadLane, {SecondValue, Lane}),
                       M);

  Value *LaneNext =
      LoopB.CreateAdd(Lane, ConstantInt::get(I32, 1), "lane.next");
  Lane->addIncoming(LaneNext, LoopBB);
  Value *Done = LoopB.CreateICmpEQ(LaneNext, ConstantInt::get(I32, WaveSize));
  Instruction *LoopBr = LoopB.CreateCondBr(Done, AfterBB, LoopBB);

  MDNode *LoopID = MDNode::getDistinct(
      Ctx,
      {nullptr,
       MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.unroll.disable")})});
  LoopID->replaceOperandWith(0, LoopID);
  LoopBr->setMetadata(LLVMContext::MD_loop, LoopID);
  B.SetInsertPoint(AfterBB, AfterBB->begin());
}

bool SQTTInstrumentPass::instrumentAddressTraces(Function &F, GfxGen Gen) {
  SmallVector<AddrTraceOp, 16> Ops;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (AddrTraceOp Op = classifyAddrTraceOp(&I, Config.TraceMemoryAddrs,
                                               Config.TraceLDSAddrs);
          Op.Kind != AddrTraceKind::None)
        Ops.push_back(Op);
    }
  }
  if (Ops.empty())
    return false;

  // Wave size is recorded once per module for the .sqtt_funcmap header.
  // RDNA is wave-32, CDNA is wave-64; a single AMDGPU code object normally
  // targets one or the other. If we ever see a mix, default to wave-64;
  // the decoder treats exec_hi=0 padding as "no upper-half lanes" so the
  // wider format stays correct for both.
  unsigned WaveSize = getWaveSize(Gen);
  if (WaveSize > AddrTraceWaveSize)
    AddrTraceWaveSize = WaveSize;

  for (AddrTraceOp &Op : Ops) {
    uint32_t OpId = NextEventID++;
    unsigned ExtraPayloadCount =
        2 + (Op.Kind == AddrTraceKind::Buffer ? 3 : 0) +
        WaveSize *
            ((Op.Kind == AddrTraceKind::Memory || Op.StructBuffer) ? 2 : 1);
    Markers.push_back({OpId, MarkerKind::AddressPoint, Op.Name.str(),
                       getSourceLoc(Op.I), 0, ExtraPayloadCount});

    IRBuilder<> B(Op.I);
    emitScopedTrace(B, F, Gen, "sqtt.addr.trace", "sqtt.addr.skip",
                    /*pinSkipHead=*/false, [&](IRBuilder<> &Trace) {
                      emitAddressTrace(Trace, Op, OpId, Gen);
                    });
  }
  return true;
}
