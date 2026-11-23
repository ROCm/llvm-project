//===- SQTTNamedMarkers.cpp - Named SQTT markers --------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Resolves string marker sentinels and emits their trace records.
///
//===----------------------------------------------------------------------===//

#include "SQTTPass.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

enum MarkerFlag : uint8_t {
  MarkerEnter = 1,
  MarkerExit = 2,
  MarkerPoint = 4,
  MarkerPayload = 8
};

} // namespace

static uint8_t markerFlags(const CallInst *Call) {
  const Function *Callee = Call ? Call->getCalledFunction() : nullptr;
  return Callee ? StringSwitch<uint8_t>(Callee->getName())
                      .Case("sqtt_marker_enter", MarkerEnter)
                      .Case("__sqtt_named_marker_enter", MarkerEnter)
                      .Case("sqtt_marker_exit", MarkerExit)
                      .Case("__sqtt_named_marker_exit", MarkerExit)
                      .Case("sqtt_marker_point", MarkerPoint)
                      .Case("__sqtt_named_marker_point", MarkerPoint)
                      .Case("sqtt_marker_data", MarkerPoint | MarkerPayload)
                      .Case("__sqtt_named_marker_data",
                            MarkerPoint | MarkerPayload)
                      .Default(0)
                : 0;
}

uint32_t SQTTInstrumentPass::resolveMarkerString(CallInst *CI, uint8_t Flags) {
  // exit(name) pops the top of the marker stack.  The name may be null or empty
  // and is documentation only; it is neither encoded nor checked.
  if (Flags & MarkerExit)
    return FlagExitPrev;

  Value *Arg = CI->getArgOperand(0)->stripPointerCasts();
  auto *GV = dyn_cast<GlobalVariable>(Arg);
  if (!GV || !GV->hasInitializer())
    return 0;
  auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer());
  if (!CDA || !CDA->isString())
    return 0;

  std::string Name = CDA->getAsString().str();
  if (!Name.empty() && Name.back() == '\0')
    Name.pop_back();

  bool IsPoint = Flags & MarkerPoint;
  uint32_t ExtraPayloadCount = (Flags & MarkerPayload) ? 1 : 0;
  std::string Key = std::string(IsPoint ? "P:" : "U:") +
                    std::to_string(ExtraPayloadCount) + ":" + Name;
  auto [It, Inserted] = UserMarkerMap.try_emplace(Key, NextEventID);
  uint32_t Id = It->getValue();
  if (Inserted) {
    ++NextEventID;
    Markers.push_back({Id,
                       IsPoint ? MarkerKind::Point : MarkerKind::UserScope,
                       Name,
                       {},
                       0,
                       ExtraPayloadCount});
  }
  bool Enter = Flags & MarkerEnter;
  return encodeMarker(Id, Enter, false); // enter or point
}

// Early emits bare traces and leaves unresolved calls for late processing.
// Late emits scoped/bounded traces and warns for unresolved calls.
bool SQTTInstrumentPass::processMarkerCalls(Function &F, GfxGen Gen,
                                            bool UseBareTrace) {
  SmallVector<CallInst *, 8> Calls;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I); markerFlags(CI))
        Calls.push_back(CI);
  if (Calls.empty())
    return false;

  Module *M = F.getParent();
  bool Changed = false;
  auto Emit = [&](IRBuilder<> &B, uint32_t Encoded, Value *Payload = nullptr) {
    if (UseBareTrace) {
      CallInst *Header = emitBareTrace(B, Encoded, M, Gen);
      if (Payload)
        emitRawTracePayload(B, Payload, M, Header);
    } else
      insertTraceMarker(B, Encoded, F, Gen, Payload);
  };

  for (unsigned I = 0; I < Calls.size(); I++) {
    CallInst *CI = Calls[I];
    uint8_t Flags = markerFlags(CI);

    // Fuse only directly adjacent exit+enter pairs.  A marker boundary
    // must not absorb work between the calls.
    if (Flags == MarkerExit && I + 1 < Calls.size()) {
      CallInst *NextCI = Calls[I + 1];
      if (markerFlags(NextCI) == MarkerEnter && CI->getNextNode() == NextCI) {
        uint32_t EnterEncoded = resolveMarkerString(NextCI, MarkerEnter);
        if (EnterEncoded) {
          uint32_t Id = EnterEncoded >> 2;
          uint32_t Fused = encodeMarker(Id, true, true);
          IRBuilder<> B(CI);
          Emit(B, Fused);
          CI->eraseFromParent();
          NextCI->eraseFromParent();
          Changed = true;
          I++;
          continue;
        }
      }
    }

    uint32_t Encoded = resolveMarkerString(CI, Flags);
    if (!Encoded) {
      if (UseBareTrace)
        continue; // not resolvable yet, leave for late pass
      errs() << "sqtt: warning: string marker argument is not a literal, "
                "skipping\n";
      CI->eraseFromParent();
      continue;
    }

    IRBuilder<> B(CI);
    Emit(B, Encoded, Flags & MarkerPayload ? CI->getArgOperand(1) : nullptr);
    CI->eraseFromParent();
    Changed = true;
  }
  return Changed || !UseBareTrace;
}
