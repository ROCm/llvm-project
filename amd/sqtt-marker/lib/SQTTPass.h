//===- SQTTPass.h - SQTT instrumentation pass -----------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares the SQTT module pass and its instrumentation helpers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_AMD_SQTT_MARKER_LIB_SQTTPASS_H
#define LLVM_AMD_SQTT_MARKER_LIB_SQTTPASS_H

#include <cstdint>
#include <string>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#include "SQTTConfig.h"
#include "SQTTTarget.h"

/// Instruments AMDGPU modules with SQTT shaderdata markers.
///
/// The early phase resolves source markers and inserts function markers before
/// inlining. The late phase filters and compacts those markers, adds automatic
/// instrumentation and scope guards, lowers target-specific trace operations,
/// and emits the funcmap.
class SQTTInstrumentPass : public llvm::PassInfoMixin<SQTTInstrumentPass> {
public:
  enum class Mode { Early, Late };

  SQTTInstrumentPass(SQTTConfig Cfg, Mode M) : Config(Cfg), PassMode(M) {}

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

private:
  SQTTConfig Config;
  Mode PassMode;
  uint32_t NextEventID = 1;
  // One ledger backs both the early-pass handoff and the final funcmap.
  // Function and kernel definition locations are "file:line" when available.
  enum class MarkerKind : uint8_t {
    Function,
    Kernel,
    UserScope,
    Point,
    SystemPoint,
    AddressPoint
  };
  struct MarkerRecord {
    uint32_t ID = 0; // Kernels have no shaderdata ID.
    MarkerKind Kind;
    std::string Name;
    std::string SourceLoc;
    uint32_t PreOptSize = 0;        // Function rows only, before inlining.
    uint32_t ExtraPayloadCount = 0; // Header rows only.
  };
  llvm::SmallVector<MarkerRecord, 16> Markers;
  // Name-to-ID is rebuilt by compaction along with the ledger IDs.
  llvm::StringMap<uint32_t> UserMarkerMap;
  llvm::Value *CurScopeCheck =
      nullptr; // cached per-function scope check result
  uint32_t ShaderClockBitsUsed = 0;

  // Automatic marker groups are allocated in their classifier enum order.
  uint32_t FirstBarrierID = 0;
  uint32_t FirstVmemID = 0;

  unsigned AddrTraceWaveSize = 0; // set once during instrumentAddressTraces

  // Phase entry points.
  llvm::PreservedAnalyses runEarly(llvm::Module &M);
  llvm::PreservedAnalyses runLate(llvm::Module &M);

  // Marker insertion and scope filtering.
  void insertTraceMarker(llvm::IRBuilder<> &B, uint32_t MarkerId,
                         llvm::Function &F, GfxGen Gen,
                         llvm::Value *Payload = nullptr);

  llvm::Value *buildScopeCheck(llvm::IRBuilder<> &B, GfxGen Gen);
  llvm::Value *getOrCreateScopeCheck(llvm::Function &F, GfxGen Gen);
  bool finalizeExistingMarkers(llvm::Function &F, GfxGen Gen);
  // Wrap an adjacent marker run in one scope-check diamond.
  void wrapRangeWithScopeCheck(llvm::CallInst *First, llvm::CallInst *Last,
                               llvm::Function &F, GfxGen Gen, bool PinSkipHead);

  uint32_t resolveMarkerString(llvm::CallInst *CI, uint8_t Flags);

  llvm::CallInst *emitBareTrace(llvm::IRBuilder<> &B, uint32_t Encoded,
                                llvm::Module *M, GfxGen Gen);
  llvm::CallInst *emitBareTraceValue(llvm::IRBuilder<> &B, llvm::Value *Val,
                                     llvm::Module *M);
  llvm::CallInst *emitRawTracePayload(llvm::IRBuilder<> &B, llvm::Value *Val,
                                      llvm::Module *M, llvm::CallInst *Header);
  static bool isTraceDataCall(const llvm::CallInst *CI);

  bool processMarkerCalls(llvm::Function &F, GfxGen Gen, bool UseBareTrace);

  // Automatic barrier and memory instrumentation.
  enum class BarrierKind : uint32_t { Signal = 0, Wait, Full, None };
  static BarrierKind classifyBarrier(llvm::CallInst *CI);
  static bool isSyncInstruction(llvm::Instruction *I);
  bool instrumentBarriers(llvm::Function &F, GfxGen Gen);

  enum class MemOpKind : uint32_t { Load = 0, Store, None };
  static MemOpKind classifyMemOp(llvm::Instruction *I);
  bool instrumentMemoryOps(llvm::Function &F, GfxGen Gen);

  // Address tracing.
  enum class AddrTraceKind { Memory, LDS, Buffer, Permute, None };
  // Everything needed after the initial scan.  Keeping the protocol shape
  // here avoids rediscovering buffer spelling and operand layout while the
  // CFG is being rewritten.
  struct AddrTraceOp {
    llvm::Instruction *I;
    llvm::StringRef Name;
    AddrTraceKind Kind;
    unsigned BufferRsrcIndex;
    bool StructBuffer;
  };
  static AddrTraceOp classifyAddrTraceOp(llvm::Instruction *I, bool TraceMemory,
                                         bool TraceLds);
  void emitAddressTrace(llvm::IRBuilder<> &B, const AddrTraceOp &Op,
                        uint32_t HeaderId, GfxGen Gen);
  void emitReadlaneTraceLoop(llvm::IRBuilder<> &B, llvm::Value *FirstValue,
                             llvm::Value *SecondValue, unsigned WaveSize);
  void emitTraceBoundary(llvm::IRBuilder<> &B, bool After,
                         bool SchedBarrier = true);
  void emitTraceBoundaries(llvm::IRBuilder<> &B, llvm::Instruction *First,
                           llvm::Instruction *Last, bool SchedBarrier);
  // Emit a sequence directly or inside the configured scope-check diamond.
  void emitScopedTrace(llvm::IRBuilder<> &B, llvm::Function &F, GfxGen Gen,
                       const char *TraceBlockName, const char *SkipBlockName,
                       bool PinSkipHead,
                       llvm::function_ref<void(llvm::IRBuilder<> &)> Emit);
  // Return the innermost-to-outermost inline chain, or "" without debug info.
  static std::string getSourceLoc(llvm::Instruction *I);
  static std::string getFunctionSourceLoc(llvm::Function &F);

  bool instrumentAddressTraces(llvm::Function &F, GfxGen Gen);

  // Insert function entry/exit markers for either pass phase.
  void insertFunctionMarkers(llvm::Function &F, uint32_t Id, GfxGen Gen,
                             bool UseBareTrace);
  void storeEarlyMarkerMetadata(llvm::Module &M, llvm::LLVMContext &Ctx);
  bool recoverEarlyMarkerMetadata(llvm::Module &M, bool &HasEarlyFunctions);

  // Late phase: filtering, ID compaction, packing, and lowering.
  bool finalizeEarlyFunctionMarkers(llvm::Module &M);
  // Pack gfx12 shader-clock headers and lower full traces through M0/NOP.
  bool finalizeFullTraces(llvm::Function &F, GfxGen Gen);

  // -O0 fallback.
  static bool hasMustTailCall(const llvm::Function &F);
  bool instrumentFunctionDirect(llvm::Function &F, GfxGen Gen);

  void emitFuncMap(llvm::Module &M);
};

#endif // LLVM_AMD_SQTT_MARKER_LIB_SQTTPASS_H
