//===- LedgerPass.cpp - Scheduling ledger, IR half ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// psched-ledger: pass plugin that applies the IR-level part of a scheduling
// ledger to AMDGPU kernels. Proof of concept.
//
// A ledger is a JSON file of compiler advice derived from profiles, named by
// the PSCHED_LEDGER environment variable (files or directories, colon
// separated). The source decides what is computed; the ledger only chooses
// among forms the compiler can show to be equivalent, so it can change speed
// but not results. This plugin handles, per function entry ("match" is a
// substring of the function name):
//
//   attrs   scheduling function attributes read by the AMDGPU scheduler
//           (amdgpu-sched-strategy, amdgpu-post-sched-strategy,
//           amdgpu-expert-scheduling-mode); any other name is refused
//   hints   existing scheduling intrinsics (sched_barrier, sched_group_barrier,
//           iglp_opt) in the blocks that call a named intrinsic
//   choose  instruction choices, each applied only at sites where the
//           equivalence is proven from the IR: mul24, wmma_bias, byte_negate,
//           bit_spread
//
// The per-region part of a ledger (instruction order and latency) is applied
// by the backend, llvm/lib/Target/AMDGPU/AMDGPUSchedLedger.cpp. Nothing here
// depends on debug info. PSCHED_VERBOSE=1 reports what was applied;
// PSCHED_DUMP_IR=<substring> writes the IR of matching functions to
// /tmp/psched-ir. Entries that match nothing are reported.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace llvm;

namespace {

/// True if any frame of the instruction's inlined-at chain is at File:Line.
/// An empty File matches any file; otherwise it is compared with the basename.
bool atLine(const Instruction &I, StringRef File, unsigned Line) {
  for (const DILocation *L = I.getDebugLoc().get(); L; L = L->getInlinedAt())
    if (L->getLine() == Line && (File.empty() || sys::path::filename(L->getFilename()) == File))
      return true;
  return false;
}

/// True if the instruction calls a function (usually an intrinsic) whose name
/// contains Name. This is how a hint names its place without debug info: "the
/// blocks that do matrix operations" is {"intrinsic": "amdgcn.wmma"}.
bool callsNamed(const Instruction &I, StringRef Name) {
  const auto *CB = dyn_cast<CallBase>(&I);
  const Function *Callee = CB ? CB->getCalledFunction() : nullptr;
  return Callee && Callee->getName().contains(Name);
}

void emit(IRBuilder<> &B, Intrinsic::ID ID, ArrayRef<uint32_t> Args) {
  SmallVector<Value *, 3> Vals;
  for (uint32_t A : Args)
    Vals.push_back(B.getInt32(A));
  B.CreateCall(Intrinsic::getOrInsertDeclaration(B.GetInsertBlock()->getModule(), ID), Vals);
}

/// Applies one hint object to F. Returns the number of basic blocks it was inserted into.

// ---------------------------------------------------------------------------
// Instruction choices ("choose"). Each one replaces an instruction by an
// equivalent form, and only where this pass proves the equivalence from the
// IR itself (operand types, intrinsic semantics). Where the proof fails the
// instruction is left alone and counted. So a ledger can ask for a choice but
// cannot make the result differ.

/// Signed range of a value, or the full range. Knows two things ValueTracking
/// does not: the result of an integer WMMA (each call adds 16 products of
/// 8-bit values to its accumulator) and lanes of vectors built from ranged
/// scalars.
ConstantRange rangeOf(Value *V, unsigned Depth = 0) {
  unsigned BW = V->getType()->getScalarSizeInBits();
  ConstantRange Full = ConstantRange::getFull(BW);
  if (Depth > 8)
    return Full;
  if (auto *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue())
      return ConstantRange(APInt::getZero(BW));
    if (auto *CI = dyn_cast_or_null<ConstantInt>(C->getSplatValue() ? C->getSplatValue() : C))
      return ConstantRange(CI->getValue());
    return Full;
  }
  if (auto *E = dyn_cast<ExtractElementInst>(V))
    return rangeOf(E->getVectorOperand(), Depth + 1);
  if (auto *SV = dyn_cast<ShuffleVectorInst>(V))
    return rangeOf(SV->getOperand(0), Depth + 1).unionWith(rangeOf(SV->getOperand(1), Depth + 1));
  if (auto *SE = dyn_cast<SExtInst>(V))
    return rangeOf(SE->getOperand(0), Depth + 1).signExtend(BW);
  if (auto *ZE = dyn_cast<ZExtInst>(V))
    return rangeOf(ZE->getOperand(0), Depth + 1).zeroExtend(BW);
  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    if (II->getIntrinsicID() == Intrinsic::amdgcn_wmma_i32_16x16x16_iu8) {
      // D = C + sum over 16 of A*B, A and B 8-bit. Each operand is signed or
      // unsigned according to its flag; the bound below holds for either
      // reading ([-128, 255] for both), so it does not depend on the flags.
      const int64_t PMin = -128 * 255, PMax = 255 * 255;
      ConstantRange Acc = rangeOf(II->getArgOperand(4), Depth + 1);
      if (Acc.isFullSet())
        return Full;
      ConstantRange Dot(APInt(BW, 16 * PMin, true), APInt(BW, 16 * PMax + 1, true));
      return Acc.add(Dot);
    }
    return Full;
  }
  if (auto *I = dyn_cast<Instruction>(V); I && V->getType()->isIntegerTy())
    return computeConstantRange(V, /*ForSigned=*/true, SimplifyQuery(I->getDataLayout(), I));
  return Full;
}

bool fitsSigned(const ConstantRange &R, unsigned Bits) {
  return !R.isFullSet() && R.getSignedMin().sge(APInt::getSignedMinValue(Bits).sext(R.getBitWidth())) &&
         R.getSignedMax().sle(APInt::getSignedMaxValue(Bits).sext(R.getBitWidth()));
}

/// mul24: a 32-bit integer multiply whose operands both fit in 24 signed bits
/// becomes llvm.amdgcn.mul.i24 (v_mul_i32_i24: 1 cycle instead of 4 on RDNA).
/// The low 32 bits of the product are the same.
std::pair<unsigned, unsigned> chooseMul24(Function &F) {
  unsigned Done = 0, Unproven = 0;
  SmallVector<BinaryOperator *, 16> Muls;
  for (Instruction &I : instructions(F))
    if (auto *BO = dyn_cast<BinaryOperator>(&I))
      if (BO->getOpcode() == Instruction::Mul && BO->getType()->isIntegerTy(32) && // scalar only
          !isa<Constant>(BO->getOperand(1)))
        Muls.push_back(BO);
  for (BinaryOperator *BO : Muls) {
    if (!fitsSigned(rangeOf(BO->getOperand(0)), 24) || !fitsSigned(rangeOf(BO->getOperand(1)), 24)) {
      ++Unproven;
      continue;
    }
    IRBuilder<> B(BO);
    Value *N = B.CreateIntrinsic(Intrinsic::amdgcn_mul_i24, {BO->getType()}, {BO->getOperand(0), BO->getOperand(1)});
    BO->replaceAllUsesWith(N);
    BO->eraseFromParent();
    ++Done;
  }
  return {Done, Unproven};
}

/// wmma_bias: an integer WMMA chain that starts from a zero accumulator and
/// whose result is only converted to float. Start the chain at the bit pattern
/// of 1.5 * 2^23 instead; then each converted lane equals
/// bitcast-to-float(result) - 1.5 * 2^23, which replaces v_cvt_f32_i32 by a
/// subtraction that can be dual-issued. Exact while |dot| < 2^22, which the
/// WMMA range proves.
std::pair<unsigned, unsigned> chooseWmmaBias(Function &F) {
  unsigned Done = 0, Unproven = 0;
  SmallVector<IntrinsicInst *, 16> Roots;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::amdgcn_wmma_i32_16x16x16_iu8 &&
          llvm::none_of(II->users(), [](User *U) {
            auto *UI = dyn_cast<IntrinsicInst>(U);
            return UI && UI->getIntrinsicID() == Intrinsic::amdgcn_wmma_i32_16x16x16_iu8;
          }))
        Roots.push_back(II);
  for (IntrinsicInst *Root : Roots) {
    // walk to the start of the chain; intermediate results may only feed the next WMMA
    IntrinsicInst *First = Root;
    bool Ok = true;
    while (auto *Prev = dyn_cast<IntrinsicInst>(First->getArgOperand(4))) {
      if (Prev->getIntrinsicID() != Intrinsic::amdgcn_wmma_i32_16x16x16_iu8 || !Prev->hasOneUse()) {
        Ok = false;
        break;
      }
      First = Prev;
    }
    Ok = Ok && isa<Constant>(First->getArgOperand(4)) && cast<Constant>(First->getArgOperand(4))->isNullValue();
    // every use: sitofp of the vector, or extractelement whose uses are all sitofp
    SmallVector<CastInst *, 16> Convs;
    for (User *U : Root->users()) {
      if (auto *C = dyn_cast<SIToFPInst>(U); C && C->getType()->getScalarType()->isFloatTy()) {
        Convs.push_back(C);
        continue;
      }
      auto *E = dyn_cast<ExtractElementInst>(U);
      if (!E) {
        Ok = false;
        break;
      }
      for (User *EU : E->users()) {
        auto *C = dyn_cast<SIToFPInst>(EU);
        if (!C || !C->getType()->isFloatTy()) {
          Ok = false;
          break;
        }
        Convs.push_back(C);
      }
    }
    ConstantRange R = rangeOf(Root);
    if (!Ok || Convs.empty() || !fitsSigned(R, 23)) { // |d| < 2^22
      if (!Convs.empty() || Ok)
        ++Unproven;
      continue;
    }
    Type *VT = First->getArgOperand(4)->getType();
    First->setArgOperand(4, ConstantInt::get(VT, 0x4B400000));
    for (CastInst *C : Convs) {
      IRBuilder<> B(C);
      Value *Bits = B.CreateBitCast(C->getOperand(0), C->getType());
      Value *N = B.CreateFSub(Bits, ConstantFP::get(C->getType(), 12582912.0));
      C->replaceAllUsesWith(N);
      C->eraseFromParent();
    }
    ++Done;
  }
  return {Done, Unproven};
}

/// True if V is a load from inside a constant global array of integers
/// whose initializer has every byte in [1, 127]. Any in-bounds read then
/// yields only such bytes, whatever its offset and width (an out-of-bounds
/// read is undefined in the original program as well).
bool loadsBytes1To127(Value *V) {
  auto *L = dyn_cast<LoadInst>(V);
  if (!L || !L->getType()->isIntegerTy())
    return false;
  auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(L->getPointerOperand()));
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer())
    return false;
  auto *Init = dyn_cast<ConstantDataArray>(GV->getInitializer());
  if (!Init || !Init->getElementType()->isIntegerTy())
    return false;
  for (unsigned char Byte : Init->getRawDataValues())
    if (Byte < 1 || Byte > 127)
      return false;
  return true;
}

/// byte_negate: the per-byte conditional negation written with __vsub4,
///   R = __vsub4(G ^ S, S),  S = N * 255,  N has only bits 0x01010101,
/// which reaches this pass as
///   R = (((G ^ S) | 0x80808080) - (S & 0x7f7f7f7f)) ^ (G & 0x80808080) ^ 0x80808080
/// becomes R = (G ^ S) + N (two instructions instead of seven).
/// Proof: each byte of S is 0x00 or 0xff according to the bit of N in that
/// byte (N*255 with N's bytes in {0,1} has no carries). The original computes
/// per byte (g ^ s) - s modulo 256 (that is what __vsub4 is: a subtraction
/// without borrow between bytes; follows from its three lines for any
/// inputs). For s = 0 this is g; for s = 0xff it is (255 - g) - 255 = -g
/// modulo 256. The new form adds n in {0,1} to (g ^ s) in one 32-bit add:
/// for n = 0 the byte is g and nothing is added; for n = 1 the byte is
/// 255 - g, and with g in [1, 127] the sum 256 - g is in [129, 255]: no carry
/// leaves the byte, and it equals -g modulo 256. So every byte is the same.
/// The range of g is read from the initializer of the constant table G is
/// loaded from (loadsBytes1To127). Not provable (left unchanged): N not
/// known to be within 0x01010101, or G not such a load.
std::pair<unsigned, unsigned> chooseByteNegate(Function &F) {
  using namespace PatternMatch;
  unsigned Done = 0, Unproven = 0;
  const DataLayout &DL = F.getParent()->getDataLayout();
  const APInt C80(32, 0x80808080u), C7F(32, 0x7f7f7f7fu), C01(32, 0x01010101u);
  SmallVector<Instruction *, 64> Xors;
  for (Instruction &I : instructions(F))
    if (I.getOpcode() == Instruction::Xor && I.getType()->isIntegerTy(32) &&
        llvm::none_of(I.users(), [](User *U) {
          auto *UI = dyn_cast<Instruction>(U);
          return UI && UI->getOpcode() == Instruction::Xor;
        }))
      Xors.push_back(&I);
  for (Instruction *Root : Xors) {
    // leaves of the xor tree under Root
    SmallVector<Value *, 4> Leaves, Work{Root};
    while (!Work.empty() && Leaves.size() <= 3) {
      Value *V = Work.pop_back_val();
      auto *X = dyn_cast<BinaryOperator>(V);
      if (X && X->getOpcode() == Instruction::Xor && (X == Root || X->hasOneUse())) {
        Work.push_back(X->getOperand(0));
        Work.push_back(X->getOperand(1));
      } else
        Leaves.push_back(V);
    }
    if (Leaves.size() != 3)
      continue;
    Value *G = nullptr, *D = nullptr, *O, *B, *A, *S, *N;
    bool HasC = false;
    for (Value *V : Leaves) {
      Value *T;
      if (match(V, m_SpecificInt(C80)))
        HasC = true;
      else if (match(V, m_And(m_Value(T), m_SpecificInt(C80))))
        G = T;
      else
        D = V;
    }
    if (!HasC || !G || !D || !match(D, m_Sub(m_Value(O), m_Value(B))) ||
        !match(O, m_Or(m_Value(A), m_SpecificInt(C80))) || !match(B, m_And(m_Value(S), m_SpecificInt(C7F))) ||
        !match(A, m_c_Xor(m_Specific(G), m_Specific(S))) || !match(S, m_Mul(m_Value(N), m_SpecificInt(255))))
      continue; // not this pattern
    KnownBits KB = computeKnownBits(N, DL);
    if (!(~KB.Zero).isSubsetOf(C01) || !loadsBytes1To127(G)) {
      ++Unproven;
      continue;
    }
    IRBuilder<> Bld(Root);
    Value *R = Bld.CreateAdd(A, N);
    Root->replaceAllUsesWith(R);
    Root->eraseFromParent();
    ++Done;
  }
  return {Done, Unproven};
}

/// Value of the i32 expression V when Leaf has the value In; false if V
/// contains anything but Leaf, constants and wrapping integer operations.
bool evalOnLeaf(Value *V, Value *Leaf, uint32_t In, uint32_t &Out, unsigned Depth = 0) {
  if (V == Leaf) {
    Out = In;
    return true;
  }
  if (auto *C = dyn_cast<ConstantInt>(V)) {
    Out = (uint32_t)C->getZExtValue();
    return true;
  }
  auto *BO = dyn_cast<BinaryOperator>(V);
  uint32_t A, B;
  if (!BO || Depth > 12 || !BO->getType()->isIntegerTy(32) || !evalOnLeaf(BO->getOperand(0), Leaf, In, A, Depth + 1) ||
      !evalOnLeaf(BO->getOperand(1), Leaf, In, B, Depth + 1))
    return false;
  switch (BO->getOpcode()) {
  case Instruction::And: Out = A & B; return true;
  case Instruction::Or: Out = A | B; return true;
  case Instruction::Xor: Out = A ^ B; return true;
  case Instruction::Add: Out = A + B; return true;
  case Instruction::Sub: Out = A - B; return true;
  case Instruction::Mul: Out = A * B; return true;
  case Instruction::Shl: Out = A << (B & 31); return B < 32;
  case Instruction::LShr: Out = A >> (B & 31); return B < 32;
  default: return false;
  }
}

/// The X of a `mul X, 0x01010101` inside the expression V, if any.
Value *findReplicatedByte(Value *V, unsigned Depth = 0) {
  auto *BO = dyn_cast<BinaryOperator>(V);
  if (!BO || Depth > 12)
    return nullptr;
  if (BO->getOpcode() == Instruction::Mul)
    if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1)); C && C->getValue() == 0x01010101u)
      return BO->getOperand(0);
  for (Value *Op : BO->operands())
    if (Value *R = findReplicatedByte(Op, Depth + 1))
      return R;
  return nullptr;
}

/// bit_spread: N = f(X) & 0x01010101 where f is built only from X, constants
/// and wrapping integer operations, and contains X * 0x01010101 (a byte
/// replicated into four bytes: a 4-cycle 32-bit multiply on RDNA, followed by
/// a mask-and-test per byte). If X is known to be below 256 and N equals
/// "bit k of nibble h of X at bit 8k" (h = 0 or 1), N is replaced by
///   (((X >> 4h) & 15) * 0x204081) & 0x01010101
/// whose multiply has operands below 2^24 (one cycle).
/// Proof: by exhaustion. X has 256 possible values (known bits); this pass
/// evaluates the original expression and the replacement for each of them
/// with 32-bit wrapping arithmetic and applies the choice only if all 256
/// results agree. Flags of the original (nuw, disjoint) are ignored in the
/// evaluation: where they would make it poison any value is a refinement.
/// Not provable (left unchanged): X not known below 256, an operation the
/// evaluator does not know, or a disagreement for either nibble.
std::pair<unsigned, unsigned> chooseBitSpread(Function &F) {
  unsigned Done = 0, Unproven = 0;
  const DataLayout &DL = F.getParent()->getDataLayout();
  SmallVector<BinaryOperator *, 64> Ands;
  for (Instruction &I : instructions(F))
    if (auto *BO = dyn_cast<BinaryOperator>(&I); BO && BO->getOpcode() == Instruction::And && BO->getType()->isIntegerTy(32))
      if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1)); C && C->getValue() == 0x01010101u)
        Ands.push_back(BO);
  for (BinaryOperator *N : Ands) {
    Value *X = findReplicatedByte(N);
    if (!X)
      continue; // not this pattern
    int Nibble = -1;
    if (computeKnownBits(X, DL).countMaxActiveBits() <= 8)
      for (int H = 0; H < 2 && Nibble < 0; ++H) {
        bool Same = true;
        for (uint32_t In = 0; In < 256 && Same; ++In) {
          uint32_t Old;
          Same = evalOnLeaf(N, X, In, Old) && Old == ((((In >> (4 * H)) & 15) * 0x204081u) & 0x01010101u);
        }
        if (Same)
          Nibble = H;
      }
    if (Nibble < 0) {
      ++Unproven;
      continue;
    }
    IRBuilder<> B(N);
    Value *T = B.CreateAnd(Nibble ? B.CreateLShr(X, 4) : X, 15);
    Value *R = B.CreateAnd(B.CreateMul(T, B.getInt32(0x204081), "", /*NUW=*/true, /*NSW=*/true), 0x01010101u);
    N->replaceAllUsesWith(R);
    N->eraseFromParent();
    ++Done;
  }
  return {Done, Unproven};
}

unsigned applyHint(Function &F, const json::Object &H) {
  // Where: {"intrinsic": substring of a called function's name}: every block with such a
  // call. Works without debug info, so a production build gets the hint too. The older
  // {"file", "line"} form needs -g and moves with the source; development use only.
  auto Line = H.getInteger("line");
  StringRef Callee = H.getString("intrinsic").value_or("");
  if (!Line && Callee.empty())
    return 0;
  StringRef File = H.getString("file").value_or("");
  auto AtHint = [&](const Instruction &I) {
    return Callee.empty() ? atLine(I, File, *Line) : callsNamed(I, Callee);
  };
  const json::Array *Groups = H.getArray("group_barriers");
  auto Barrier = H.getInteger("sched_barrier");
  auto IGLP = H.getInteger("iglp_opt");
  // Group barriers and iglp_opt collect instructions above them, so they go at the end of the
  // block. A plain barrier separates what is before the line from what follows it.
  bool AtEnd = H.getString("where").value_or(Barrier ? "before" : "block-end") == "block-end";
  uint32_t Sync = H.getInteger("sync").value_or(0);
  int64_t Repeat = H.getInteger("repeat").value_or(1);

  unsigned Blocks = 0;
  for (BasicBlock &BB : F) {
    Instruction *Hit = nullptr;
    for (Instruction &I : BB)
      if (!isa<PHINode>(I) && AtHint(I)) {
        Hit = &I;
        break;
      }
    if (!Hit)
      continue;
    IRBuilder<> B(AtEnd ? BB.getTerminator() : Hit);
    B.SetCurrentDebugLocation(Hit->getDebugLoc());
    if (Barrier)
      emit(B, Intrinsic::amdgcn_sched_barrier, {uint32_t(*Barrier)});
    if (IGLP)
      emit(B, Intrinsic::amdgcn_iglp_opt, {uint32_t(*IGLP)});
    if (Groups)
      for (int64_t R = 0; R < Repeat; ++R)
        for (const json::Value &G : *Groups)
          if (const json::Array *P = G.getAsArray(); P && P->size() == 2)
            emit(B, Intrinsic::amdgcn_sched_group_barrier,
                 {uint32_t((*P)[0].getAsInteger().value_or(0)), uint32_t((*P)[1].getAsInteger().value_or(0)), Sync});
    ++Blocks;
  }
  return Blocks;
}

struct LedgerPass : RequiredPassInfoMixin<LedgerPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    if (!Triple(M.getTargetTriple()).isAMDGPU())
      return PreservedAnalyses::all();
    const char *Spec = std::getenv("PSCHED_LEDGER");
    if (!Spec || !*Spec)
      return PreservedAnalyses::all();
    bool Verbose = std::getenv("PSCHED_VERBOSE") != nullptr;
    // PSCHED_LEDGER is a colon-separated list of files or directories of .json files.
    // Plain C++ file access is used on purpose: assertion builds of LLVM forbid its own file
    // API inside the pass pipeline (IO sandbox).
    std::vector<std::string> Files;
    SmallVector<StringRef, 4> Parts;
    StringRef(Spec).split(Parts, ':', -1, false);
    for (StringRef P : Parts) {
      std::error_code EC;
      std::filesystem::path Dir(P.str());
      if (!std::filesystem::is_directory(Dir, EC)) {
        Files.push_back(P.str());
        continue;
      }
      std::vector<std::string> InDir;
      for (const auto &Entry : std::filesystem::directory_iterator(Dir, EC))
        if (Entry.path().extension() == ".json")
          InDir.push_back(Entry.path().string());
      llvm::sort(InDir);
      Files.insert(Files.end(), InDir.begin(), InDir.end());
    }
    // PSCHED_DUMP_IR=<substring>: write the IR of every matching function, as this pass sees it
    // (before any choice is applied), to /tmp/psched-ir/<function>.ll. For writing new choices.
    if (const char *Sub = std::getenv("PSCHED_DUMP_IR")) {
      std::filesystem::create_directories("/tmp/psched-ir");
      for (Function &F : M)
        if (!F.isDeclaration() && F.getName().contains(Sub)) {
          // Print into a string and write it with plain C++ IO: raw_os_ostream is not exported by
          // the compiler binary, and LLVM's own file API is forbidden inside the pass pipeline in
          // assertion builds (IO sandbox).
          std::string Text;
          raw_string_ostream ROS(Text);
          F.print(ROS);
          std::ofstream("/tmp/psched-ir/" + F.getName().substr(0, 120).str() + ".ll") << Text;
        }
    }
    bool Changed = false;
    for (const std::string &File : Files)
      Changed |= applyFile(M, File, Verbose);
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  /// Applies the function attributes and source-line hints of one ledger file.
  bool applyFile(Module &M, StringRef Path, bool Verbose) {
    std::ifstream In(Path.str(), std::ios::binary);
    if (!In) {
      errs() << "psched: cannot read ledger '" << Path << "'\n";
      return false;
    }
    std::string Text((std::istreambuf_iterator<char>(In)), std::istreambuf_iterator<char>());
    auto Root = json::parse(Text);
    if (!Root) {
      errs() << "psched: ledger '" << Path << "' is not valid JSON: " << toString(Root.takeError()) << "\n";
      return false;
    }
    const json::Object *Obj = Root->getAsObject();
    const json::Array *Fns = Obj ? Obj->getArray("functions") : nullptr;
    if (!Fns) {
      errs() << "psched: ledger '" << Path << "' has no \"functions\" array\n";
      return false;
    }
    bool Changed = false;
    for (const json::Value &V : *Fns) {
      const json::Object *E = V.getAsObject();
      if (!E)
        continue;
      StringRef Match = E->getString("match").value_or("");
      if (Match.empty())
        continue;
      // Optional "source": only apply (and only warn) in translation units whose file name contains it.
      if (auto Src = E->getString("source"); Src && !StringRef(M.getSourceFileName()).contains(*Src))
        continue;
      const json::Object *Attrs = E->getObject("attrs");
      const json::Array *Hints = E->getArray("hints");
      const json::Array *Choose = E->getArray("choose");
      if (!Attrs && !Hints && !Choose)
        continue; // region-level entries are applied by the backend
      unsigned Hits = 0;
      SmallVector<unsigned, 8> HintBlocks(Hints ? Hints->size() : 0, 0);
      for (Function &F : M) {
        if (F.isDeclaration() || (Match != "*" && !F.getName().contains(Match)))
          continue;
        ++Hits;
        if (Attrs)
          for (const auto &KV : *Attrs)
            if (std::optional<StringRef> Val = KV.second.getAsString()) {
              // Only scheduling attributes: a ledger must not be able to
              // change what the program computes (e.g. fast-math attributes).
              StringRef Name(KV.first);
              if (Name != "amdgpu-sched-strategy" &&
                  Name != "amdgpu-post-sched-strategy" &&
                  Name != "amdgpu-expert-scheduling-mode") {
                if (Hits == 1)
                  errs() << "psched: warning: attribute '" << Name
                         << "' is not a scheduling attribute; ignored\n";
                continue;
              }
              F.addFnAttr(Name, *Val);
              Changed = true;
              if (Verbose)
                errs() << "psched: " << F.getName() << " <- " << StringRef(KV.first) << "=" << *Val << "\n";
            }
        if (Choose)
          for (const json::Value &CV : *Choose) {
            std::optional<StringRef> What = CV.getAsString();
            std::pair<unsigned, unsigned> R{0, 0};
            if (What == "mul24")
              R = chooseMul24(F);
            else if (What == "wmma_bias")
              R = chooseWmmaBias(F);
            else if (What == "byte_negate")
              R = chooseByteNegate(F);
            else if (What == "bit_spread")
              R = chooseBitSpread(F);
            else {
              if (Hits == 1)
                errs() << "psched: warning: unknown choice '" << What.value_or("?") << "'; ignored\n";
              continue;
            }
            Changed |= R.first != 0;
            if (Verbose || (!R.first && R.second))
              errs() << "psched: " << F.getName() << " <- choose " << *What << ": " << R.first << " applied, "
                     << R.second << " not provable (left unchanged)\n";
          }
        if (Hints)
          for (unsigned I = 0; I < Hints->size(); ++I)
            if (const json::Object *H = (*Hints)[I].getAsObject()) {
              unsigned N = applyHint(F, *H);
              HintBlocks[I] += N;
              Changed |= N != 0;
              if (Verbose && N)
                errs() << "psched: " << F.getName() << " <- hint #" << I << " at "
                       << (H->getInteger("line") ? "line " + std::to_string(*H->getInteger("line"))
                                                 : ("calls of *" + H->getString("intrinsic").value_or("") + "*").str())
                       << " in " << N << " block(s)\n";
            }
      }
      if (!Hits)
        errs() << "psched: warning: ledger entry '" << Match << "' matched no function (stale ledger?)\n";
      else
        for (unsigned I = 0; I < HintBlocks.size(); ++I)
          if (!HintBlocks[I])
            errs() << "psched: warning: hint #" << I << " of entry '" << Match
                   << "' matched no instruction (stale ledger, or built without -g?)\n";
    }
    return Changed;
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "psched-ledger", "0.1", [](PassBuilder &PB) {
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase) { MPM.addPass(LedgerPass()); });
          }};
}
