//===- SQTTFunctions.cpp - SQTT function instrumentation ------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements function instrumentation, pruning, and marker ID compaction.
///
//===----------------------------------------------------------------------===//

#include "SQTTPass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

void SQTTInstrumentPass::insertFunctionMarkers(Function &F, uint32_t Id,
                                               GfxGen Gen, bool UseBareTrace) {
  auto Emit = [&](Instruction *I, uint32_t Marker) {
    IRBuilder<> B(I);
    if (UseBareTrace)
      emitBareTrace(B, Marker, F.getParent(), Gen);
    else
      insertTraceMarker(B, Marker, F, Gen);
  };

  Instruction *Entry = UseBareTrace || !CurScopeCheck
                           ? &*F.getEntryBlock().getFirstInsertionPt()
                           : cast<Instruction>(CurScopeCheck)->getNextNode();
  Emit(Entry, encodeMarker(Id, /*enter=*/true, /*exit_prev=*/false));

  SmallVector<ReturnInst *, 4> Rets;
  for (BasicBlock &BB : F)
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
      Rets.push_back(RI);
  for (ReturnInst *RI : Rets)
    Emit(RI, UseBareTrace
                 ? encodeMarker(Id, /*enter=*/false, /*exit_prev=*/true)
                 : FlagExitPrev);
}

void SQTTInstrumentPass::storeEarlyMarkerMetadata(Module &M, LLVMContext &Ctx) {
  Type *I32 = Type::getInt32Ty(Ctx);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("sqtt.markers.early");
  auto AsInt = [&](uint32_t Value) {
    return ConstantAsMetadata::get(ConstantInt::get(I32, Value));
  };
  for (const MarkerRecord &Entry : Markers)
    NMD->addOperand(MDNode::get(
        Ctx,
        {AsInt(Entry.ID), AsInt(static_cast<unsigned>(Entry.Kind)),
         MDString::get(Ctx, Entry.Name), AsInt(Entry.PreOptSize),
         MDString::get(Ctx, Entry.SourceLoc), AsInt(Entry.ExtraPayloadCount)}));
}

bool SQTTInstrumentPass::recoverEarlyMarkerMetadata(Module &M,
                                                    bool &HasEarlyFunctions) {
  HasEarlyFunctions = false;
  NamedMDNode *NMD = M.getNamedMetadata("sqtt.markers.early");
  if (!NMD)
    return false;
  for (MDNode *Op : NMD->operands()) {
    if (Op->getNumOperands() < 6)
      continue;
    auto *IdC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(0));
    auto *KindC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(1));
    auto *NameS = dyn_cast<MDString>(Op->getOperand(2));
    auto *SizeC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(3));
    auto *LocS = dyn_cast<MDString>(Op->getOperand(4));
    auto *PayloadC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(5));
    if (!IdC || !KindC || !NameS || !SizeC || !LocS || !PayloadC)
      continue;

    MarkerKind Kind = static_cast<MarkerKind>(KindC->getZExtValue());
    if (Kind != MarkerKind::Function && Kind != MarkerKind::UserScope &&
        Kind != MarkerKind::Point)
      continue;
    HasEarlyFunctions |= Kind == MarkerKind::Function;
    uint32_t Id = static_cast<uint32_t>(IdC->getZExtValue());
    Markers.push_back({Id, Kind, NameS->getString().str(),
                       LocS->getString().str(),
                       static_cast<uint32_t>(SizeC->getZExtValue()),
                       static_cast<uint32_t>(PayloadC->getZExtValue())});
    MarkerRecord &Entry = Markers.back();
    if (Kind == MarkerKind::UserScope || Kind == MarkerKind::Point) {
      std::string Key = std::string(Kind == MarkerKind::Point ? "P:" : "U:") +
                        std::to_string(Entry.ExtraPayloadCount) + ":" +
                        Entry.Name;
      UserMarkerMap[Key] = Id;
    }
    // Function IDs are compacted below.  Only user IDs need to advance
    // the no-op fallback counter used when every function is filtered.
    if (Kind != MarkerKind::Function && Id >= NextEventID)
      NextEventID = Id + 1;
  }
  NMD->eraseFromParent();
  return true;
}

bool SQTTInstrumentPass::finalizeEarlyFunctionMarkers(Module &M) {
  // One state entry owns all transient state for one early marker ID.
  struct Entry {
    MarkerRecord *Record;
    uint64_t Count = 0;
    uint32_t NewID = 0;
    bool Seen = false, Disabled = false;
  };
  DenseMap<uint32_t, Entry> Entries;
  for (MarkerRecord &Record : Markers)
    if (Record.ID)
      Entries.try_emplace(Record.ID, Entry{&Record});
  auto Find = [&](uint32_t Id) -> Entry * {
    auto It = Entries.find(Id);
    return It == Entries.end() ? nullptr : &It->second;
  };

  bool Changed = false;
  if (Config.FunctionThreshold != 0) {
    // A clone group shares one early ID, so a below-threshold copy prunes
    // every copy that carries that ID.
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      MDNode *MD = F.getMetadata("sqtt.func.id");
      auto *IdC =
          MD ? mdconst::dyn_extract<ConstantInt>(MD->getOperand(0)) : nullptr;
      if (!IdC)
        continue;

      Entry *Entry = Find(IdC->getZExtValue());
      F.setMetadata("sqtt.func.id", nullptr);
      if (!Entry || Entry->Record->Kind != MarkerKind::Function)
        continue;

      bool FirstCopy = !Entry->Seen;
      Entry->Seen = true;
      if (computeFunctionSize(F, Config.Mode) <= Config.FunctionThreshold) {
        Changed |= !Entry->Disabled;
        Entry->Disabled = true;
      } else if (FirstCopy) {
        std::string Loc = getFunctionSourceLoc(F);
        if (!Loc.empty())
          Entry->Record->SourceLoc = std::move(Loc);
      }
    }

    for (auto &[Id, MapEntry] : Entries)
      if (MapEntry.Record->Kind == MarkerKind::Function && !MapEntry.Seen &&
          MapEntry.Record->PreOptSize <= Config.FunctionThreshold) {
        Changed |= !MapEntry.Disabled;
        MapEntry.Disabled = true;
      }
  }

  // Snapshot first: pruning and lowering erase calls while preserving the
  // stable state pointer that owns each old ID.
  SmallVector<std::pair<CallInst *, Entry *>, 16> Traces;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!isTraceDataCall(CI) || !CI->getMetadata(SqttMarkerHeaderMetadata))
          continue;
        auto *Arg = dyn_cast<ConstantInt>(CI->getArgOperand(0));
        if (Arg)
          if (Entry *Entry =
                  Find(static_cast<uint32_t>(Arg->getZExtValue()) >> 2))
            Traces.emplace_back(CI, Entry);
      }

  for (auto [Call, MapEntry] : Traces) {
    uint32_t Flags =
        cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue() & FlagMask;
    if (MapEntry->Disabled && (Flags == FlagEnter || Flags == FlagExitPrev)) {
      Call->eraseFromParent();
      Changed = true;
      continue;
    }
    if (!MapEntry->Disabled)
      ++MapEntry->Count;
  }

  SmallVector<Entry *, 16> Sorted;
  Sorted.reserve(Entries.size());
  for (auto &[Id, MapEntry] : Entries)
    if (!MapEntry.Disabled)
      Sorted.push_back(&MapEntry);
  llvm::sort(Sorted, [](const Entry *A, const Entry *B) {
    if (A->Count != B->Count)
      return A->Count > B->Count;
    return A->Record->ID < B->Record->ID;
  });
  uint32_t NextId = 1;
  for (Entry *Entry : Sorted)
    Entry->NewID = NextId++;

  for (auto [Call, MapEntry] : Traces) {
    // Disabled calls may have been erased above, so never dereference
    // their CallInst here.
    if (MapEntry->Disabled)
      continue;
    GfxGen Gen = getGfxGen(*Call->getFunction());
    if (Gen == GfxGen::Unknown)
      continue;
    uint32_t Value = cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue();
    IRBuilder<> B(Call);
    CallInst *Replacement =
        emitBareTrace(B,
                      (Value & FlagMask) == FlagExitPrev
                          ? FlagExitPrev
                          : (MapEntry->NewID << 2) | (Value & FlagMask),
                      &M, Gen);
    Replacement->copyMetadata(*Call);
    Call->eraseFromParent();
    Changed = true;
  }

  Markers.erase(llvm::remove_if(Markers,
                                [&](const MarkerRecord &Record) {
                                  Entry *Entry = Find(Record.ID);
                                  return Record.Kind == MarkerKind::Function &&
                                         Entry && Entry->Disabled;
                                }),
                Markers.end());
  auto Remap = [&](uint32_t &Id) {
    if (Entry *Entry = Find(Id))
      Id = Entry->NewID;
  };
  for (MarkerRecord &Entry : Markers)
    Remap(Entry.ID);
  for (StringMapEntry<uint32_t> &Entry : UserMarkerMap)
    Remap(Entry.getValue());
  NextEventID = NextId;
  return Changed;
}

bool SQTTInstrumentPass::hasMustTailCall(const Function &F) {
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      if (const auto *CB = dyn_cast<CallBase>(&I); CB && CB->isMustTailCall())
        return true;
  return false;
}

bool SQTTInstrumentPass::instrumentFunctionDirect(Function &F, GfxGen Gen) {
  if (hasMustTailCall(F) ||
      computeFunctionSize(F, Config.Mode) <= Config.FunctionThreshold)
    return false;

  uint32_t Id = NextEventID++;
  Markers.push_back(
      {Id, MarkerKind::Function, F.getName().str(), getFunctionSourceLoc(F)});
  insertFunctionMarkers(F, Id, Gen, /*useBareTrace=*/false);
  return true;
}
