//===- SQTTMarkerTest.cpp - SQTT marker unit tests ------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests SQTT configuration, encoding, instrumentation, and funcmap behavior.
///
//===----------------------------------------------------------------------===//

#include "SQTTConfig.h"
#include "SQTTPass.h"
#include "SQTTTarget.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

static void setEnvironment(StringRef Name, std::optional<StringRef> Value) {
#ifdef _WIN32
  int Result = _putenv_s(Name.str().c_str(), Value ? Value->str().c_str() : "");
#else
  int Result = Value ? setenv(Name.str().c_str(), Value->str().c_str(), 1)
                     : unsetenv(Name.str().c_str());
#endif
  if (Result != 0)
    report_fatal_error("failed to update test environment");
}

namespace {

class ScopedEnv {
public:
  ScopedEnv(std::string Name, std::optional<std::string> Value)
      : Name(std::move(Name)) {
    if (const char *Old = std::getenv(Name.c_str()))
      OldValue = Old;
    setEnvironment(this->Name,
                   Value ? std::optional<StringRef>(*Value) : std::nullopt);
  }
  ~ScopedEnv() {
    setEnvironment(Name, OldValue ? std::optional<StringRef>(*OldValue)
                                  : std::nullopt);
  }

  ScopedEnv(const ScopedEnv &) = delete;
  ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
  std::string Name;
  std::optional<std::string> OldValue;
};

} // namespace

static std::vector<std::unique_ptr<ScopedEnv>> clearSqttEnvironment() {
  std::vector<std::unique_ptr<ScopedEnv>> Env;
  for (const char *Name :
       {"SQTT_INSTRUMENT_BARRIERS", "SQTT_MEM_BARRIER", "SQTT_SCOPE_WAVE",
        "SQTT_SCOPE_SIMD", "SQTT_SCOPE_CU", "SQTT_SCOPE_WG",
        "SQTT_SHADER_CLOCK_BITS", "SQTT_SHADER_CLOCK_SHIFT",
        "SQTT_INSTRUMENT_FUNCTIONS", "SQTT_INSTRUMENT_MEMORY",
        "SQTT_TRACE_ADDRESSES"})
    Env.push_back(std::make_unique<ScopedEnv>(Name, std::nullopt));
  return Env;
}

static std::unique_ptr<Module> makeModule(LLVMContext &Ctx) {
  std::unique_ptr<Module> Result =
      std::make_unique<Module>("markers-unit", Ctx);
  Result->setTargetTriple(Triple("amdgcn-amd-amdhsa"));
  return Result;
}

static llvm::Function *makeFunction(llvm::Module &Module, StringRef Name,
                                    StringRef Cpu, FunctionType *Type) {
  llvm::Function *Fn =
      llvm::Function::Create(Type, GlobalValue::ExternalLinkage, Name, Module);
  Fn->addFnAttr("target-cpu", Cpu);
  return Fn;
}

static llvm::Function *declareFunction(llvm::Module &Module, StringRef Name,
                                       Type *Result, ArrayRef<Type *> Args) {
  return llvm::Function::Create(FunctionType::get(Result, Args, false),
                                GlobalValue::ExternalLinkage, Name, &Module);
}

static llvm::Function *makeVoidFunction(llvm::Module &Module, StringRef Name,
                                        StringRef Cpu) {
  LLVMContext &Ctx = Module.getContext();
  llvm::Function *Fn = makeFunction(
      Module, Name, Cpu, FunctionType::get(Type::getVoidTy(Ctx), false));
  IRBuilder<>(BasicBlock::Create(Ctx, "entry", Fn)).CreateRetVoid();
  return Fn;
}

static llvm::Function *makeGlobalLoadFunction(llvm::Module &Module,
                                              StringRef Name, StringRef Cpu) {
  LLVMContext &Ctx = Module.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  llvm::Function *Fn =
      makeFunction(Module, Name, Cpu,
                   FunctionType::get(Type::getVoidTy(Ctx),
                                     {PointerType::get(Ctx, 1)}, false));
  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", Fn));
  Builder.CreateLoad(I32, Fn->getArg(0));
  Builder.CreateRetVoid();
  return Fn;
}

static llvm::Function *makeBufferTraceFunction(llvm::Module &Module,
                                               StringRef Cpu, bool Bpermute) {
  LLVMContext &Ctx = Module.getContext();
  Type *I32 = Type::getInt32Ty(Ctx), *I16 = Type::getInt16Ty(Ctx),
       *I64 = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  FixedVectorType *RsrcTy = FixedVectorType::get(I32, 4);
  PointerType *RsrcPtrTy = PointerType::get(Ctx, 8);
  llvm::Function *BufferFn =
      makeFunction(Module, "buffer_traces", Cpu,
                   FunctionType::get(VoidTy, {RsrcPtrTy}, false));
  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", BufferFn));
  Value *Rsrc = ConstantAggregateZero::get(RsrcTy);

  Type *OffsetTy = Bpermute ? I64 : I32;
  Builder.CreateCall(
      declareFunction(Module, "llvm.amdgcn.raw.buffer.load.unit", I32,
                      {RsrcTy, OffsetTy, I16}),
      {Rsrc, ConstantInt::get(OffsetTy, 11), ConstantInt::get(I16, 3)});
  Builder.CreateCall(declareFunction(Module,
                                     "llvm.amdgcn.struct.buffer.store.unit",
                                     VoidTy, {I32, RsrcTy, I16, I16, I64}),
                     {ConstantInt::get(I32, 17), Rsrc, ConstantInt::get(I16, 5),
                      ConstantInt::get(I16, 7), ConstantInt::get(I64, 9)});
  Builder.CreateCall(
      declareFunction(Module, "llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.unit",
                      I32, {I32, I32, RsrcPtrTy, I16, I16}),
      {ConstantInt::get(I32, 1), ConstantInt::get(I32, 2), BufferFn->getArg(0),
       ConstantInt::get(I16, 4), ConstantInt::get(I16, 6)});
  Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                         &Module, Bpermute ? Intrinsic::amdgcn_ds_bpermute
                                           : Intrinsic::amdgcn_ds_permute),
                     {ConstantInt::get(I32, 16), ConstantInt::get(I32, 33)});
  Builder.CreateRetVoid();

  llvm::Function *Addresses = makeFunction(
      Module, "address_spaces", Cpu,
      FunctionType::get(VoidTy,
                        {PointerType::get(Ctx, 0), PointerType::get(Ctx, 1),
                         PointerType::get(Ctx, 2), PointerType::get(Ctx, 4),
                         PointerType::get(Ctx, 5)},
                        false));
  IRBuilder<> AddressBuilder(BasicBlock::Create(Ctx, "entry", Addresses));
  AddressBuilder.CreateLoad(I32, Addresses->getArg(0));
  AddressBuilder.CreateStore(ConstantInt::get(I32, 1), Addresses->getArg(1));
  for (unsigned I : {2u, 3u, 4u})
    AddressBuilder.CreateLoad(I32, Addresses->getArg(I));
  AddressBuilder.CreateRetVoid();
  return BufferFn;
}

static SQTTConfig fullScopeConfig() {
  SQTTConfig Config;
  Config.WaveMask = FullWaveMask;
  Config.SimdMask = FullSimdMask;
  Config.CuMask = FullCuMask;
  Config.WgMask = FullWgMask;
  Config.MemBarrier = MemBarrierMode::None;
  return Config;
}

static void
runPass(llvm::Module &Module, const SQTTConfig &Config,
        SQTTInstrumentPass::Mode Mode = SQTTInstrumentPass::Mode::Late) {
  ModuleAnalysisManager AnalysisManager;
  SQTTInstrumentPass(Config, Mode).run(Module, AnalysisManager);
}

static CallInst *insertTraceCallBefore(Instruction *InsertPt, uint32_t Encoded,
                                       bool PassHeader = false) {
  llvm::Module *Module = InsertPt->getModule();
  IRBuilder<> Builder(InsertPt);
  CallInst *Call = Builder.CreateCall(
      Intrinsic::getOrInsertDeclaration(Module, Intrinsic::amdgcn_s_ttracedata),
      {ConstantInt::get(Type::getInt32Ty(Module->getContext()), Encoded)});
  if (PassHeader)
    Call->setMetadata("sqtt.marker_header",
                      MDNode::get(Module->getContext(), {}));
  return Call;
}

static llvm::Function *makeNamedMarkerSentinel(llvm::Module &Module,
                                               StringRef Name) {
  LLVMContext &Ctx = Module.getContext();
  return declareFunction(Module, Name, Type::getVoidTy(Ctx),
                         {PointerType::get(Ctx, 0)});
}

static GlobalVariable *makeMarkerString(llvm::Module &Module, StringRef Value) {
  LLVMContext &Ctx = Module.getContext();
  Constant *Initializer = ConstantDataArray::getString(Ctx, Value, true);
  return new GlobalVariable(Module, Initializer->getType(), true,
                            GlobalValue::PrivateLinkage, Initializer,
                            ".sqtt.marker.string");
}

static void addEarlyFunctionMapEntry(llvm::Module &Module, uint32_t Id,
                                     StringRef Name, unsigned PreOptSize,
                                     StringRef SourceLoc) {
  LLVMContext &Ctx = Module.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  NamedMDNode *EarlyMap = Module.getOrInsertNamedMetadata("sqtt.markers.early");
  EarlyMap->addOperand(MDNode::get(
      Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, Id)),
            ConstantAsMetadata::get(
                ConstantInt::get(I32, 0)), // MarkerKind::Function
            MDString::get(Ctx, Name),
            ConstantAsMetadata::get(ConstantInt::get(I32, PreOptSize)),
            MDString::get(Ctx, SourceLoc),
            ConstantAsMetadata::get(ConstantInt::get(I32, 0))}));
}

static void addEarlyFunctionMetadata(llvm::Function &Function, uint32_t Id,
                                     unsigned PreOptSize, StringRef SourceLoc) {
  llvm::Module *Module = Function.getParent();
  LLVMContext &Ctx = Module->getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Function.setMetadata(
      "sqtt.func.id",
      MDNode::get(Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, Id))}));
  addEarlyFunctionMapEntry(*Module, Id, Function.getName(), PreOptSize,
                           SourceLoc);
}

static void addPassOwnedFunctionMarkers(llvm::Function &Function, uint32_t Id) {
  BasicBlock &Entry = Function.getEntryBlock();
  insertTraceCallBefore(&*Entry.getFirstInsertionPt(),
                        encodeMarker(Id, true, false), true);
  insertTraceCallBefore(Entry.getTerminator(), encodeMarker(Id, false, true),
                        true);
}

static llvm::Function *makeLargePassOwnedFunction(llvm::Module &Module,
                                                  StringRef Name, uint32_t Id,
                                                  StringRef SourceLoc) {
  LLVMContext &Ctx = Module.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  llvm::Function *Function =
      makeFunction(Module, Name, "gfx1100",
                   FunctionType::get(Type::getVoidTy(Ctx), {I32}, false));
  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", Function));
  Value *Value = Function->getArg(0);
  for (unsigned I = 0; I < 30; ++I)
    Value = Builder.CreateAdd(Value, ConstantInt::get(I32, I + 1));
  Builder.CreateRetVoid();
  addPassOwnedFunctionMarkers(*Function, Id);
  addEarlyFunctionMetadata(*Function, Id, 40, SourceLoc);
  return Function;
}

static llvm::Function *makeMustTailFunction(llvm::Module &Module,
                                            StringRef Name) {
  LLVMContext &Ctx = Module.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *Type = FunctionType::get(I32, {I32}, false);
  llvm::Function *Callee =
      declareFunction(Module, Name.str() + ".callee", I32, {I32});
  llvm::Function *Function = makeFunction(Module, Name, "gfx1100", Type);
  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", Function));
  CallInst *Call = Builder.CreateCall(Callee, {Function->getArg(0)});
  Call->setTailCallKind(CallInst::TCK_MustTail);
  Builder.CreateRet(Call);
  return Function;
}

static std::string getFuncMap(const llvm::Module &Module) {
  for (const GlobalVariable &Global : Module.globals())
    if (Global.getSection() == ".sqtt_funcmap" && Global.hasInitializer())
      if (auto *Data = dyn_cast<ConstantDataArray>(Global.getInitializer());
          Data && Data->isString())
        return Data->getAsCString().str();
  return {};
}

static std::string runPassAndGetFuncMap(
    llvm::Module &Module, const SQTTConfig &Config,
    SQTTInstrumentPass::Mode Mode = SQTTInstrumentPass::Mode::Late) {
  runPass(Module, Config, Mode);
  return getFuncMap(Module);
}

static std::string printModule(const llvm::Module &Module) {
  std::string Text;
  raw_string_ostream Os(Text);
  Module.print(Os, nullptr);
  return Os.str();
}

template <typename Visitor>
static void forEachCall(const llvm::Function &Fn, Visitor Visit) {
  for (const BasicBlock &Block : Fn)
    for (const Instruction &Inst : Block)
      if (const auto *Call = dyn_cast<CallInst>(&Inst))
        Visit(*Call);
}

static size_t countIntrinsicCalls(const llvm::Function &Fn, Intrinsic::ID Id) {
  size_t Count = 0;
  forEachCall(Fn, [&](const CallInst &Call) {
    const llvm::Function *Callee = Call.getCalledFunction();
    Count += Callee && Callee->getIntrinsicID() == Id;
  });
  return Count;
}

static size_t countIntrinsicCalls(const llvm::Module &Module,
                                  Intrinsic::ID Id) {
  size_t Count = 0;
  for (const llvm::Function &Fn : Module)
    Count += countIntrinsicCalls(Fn, Id);
  return Count;
}

static size_t countFences(const llvm::Function &Fn) {
  size_t Count = 0;
  for (const BasicBlock &Block : Fn)
    for (const Instruction &Inst : Block)
      Count += isa<FenceInst>(Inst);
  return Count;
}

static std::vector<uint32_t> traceMarkerValues(const llvm::Function &Fn) {
  std::vector<uint32_t> Values;
  forEachCall(Fn, [&](const CallInst &Call) {
    const llvm::Function *Callee = Call.getCalledFunction();
    if (!Callee)
      return;
    Intrinsic::ID Id = Callee->getIntrinsicID();
    if (Id != Intrinsic::amdgcn_s_ttracedata &&
        Id != Intrinsic::amdgcn_s_ttracedata_imm)
      return;
    if (auto *Arg = dyn_cast<ConstantInt>(Call.getArgOperand(0)))
      Values.push_back(Arg->getZExtValue());
  });
  return Values;
}

static std::optional<uint32_t> markerAfter(Instruction *Instruction) {
  for (Instruction = Instruction->getNextNode(); Instruction;
       Instruction = Instruction->getNextNode()) {
    auto *Call = dyn_cast<CallInst>(Instruction);
    llvm::Function *Callee = Call ? Call->getCalledFunction() : nullptr;
    if (!Callee)
      return std::nullopt;
    if (Callee->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier)
      continue;
    if (Callee->getIntrinsicID() != Intrinsic::amdgcn_s_ttracedata_imm)
      return std::nullopt;
    if (auto *Value = dyn_cast<ConstantInt>(Call->getArgOperand(0)))
      return Value->getZExtValue();
    return std::nullopt;
  }
  return std::nullopt;
}

static const CallInst *findM0NopTrace(const llvm::Function &Function) {
  constexpr const char TraceAsm[] = "s_mov_b32 m0, $1\n"
                                    "s_nop 0\n"
                                    "s_ttracedata";
  const CallInst *Trace = nullptr;
  forEachCall(Function, [&](const CallInst &Call) {
    auto *AsmCall = dyn_cast<InlineAsm>(Call.getCalledOperand());
    if (!Trace && AsmCall && AsmCall->hasSideEffects() &&
        AsmCall->getAsmString() == TraceAsm)
      Trace = &Call;
  });
  return Trace;
}

static bool isTraceCall(const CallInst &Call) {
  if (const llvm::Function *Callee = Call.getCalledFunction())
    return Callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata ||
           Callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata_imm;
  if (const auto *AsmCall = dyn_cast<InlineAsm>(Call.getCalledOperand()))
    return AsmCall->getAsmString().contains("s_ttracedata");
  return false;
}

static CallInst *findTraceWithMetadata(llvm::Function &Function,
                                       StringRef Metadata) {
  for (BasicBlock &Block : Function)
    for (Instruction &Instruction : Block)
      if (auto *Call = dyn_cast<CallInst>(&Instruction);
          Call && Call->getMetadata(Metadata) && isTraceCall(*Call))
        return Call;
  return nullptr;
}

static void expectScopedSkipPin(llvm::Function &Function) {
  const BasicBlock *Skip = nullptr;
  for (const BasicBlock &Block : Function)
    if (Block.getName().starts_with("sqtt.skip"))
      Skip = &Block;
  ASSERT_NE(Skip, nullptr);
  auto It = Skip->begin();
  ASSERT_NE(It, Skip->end());
  auto *Pin = dyn_cast<CallInst>(&*It++);
  ASSERT_NE(Pin, nullptr);
  ASSERT_EQ(Pin->getCalledFunction()->getIntrinsicID(),
            Intrinsic::amdgcn_sched_barrier);
  ASSERT_NE(It, Skip->end());
  auto *Sync = dyn_cast<CallInst>(&*It);
  ASSERT_NE(Sync, nullptr);
  EXPECT_EQ(Sync->getCalledFunction()->getIntrinsicID(),
            Intrinsic::amdgcn_s_barrier);
}

static void expectScopedMarkerCase(bool Early, bool Sync) {
  LLVMContext Ctx;
  std::unique_ptr<Module> Module = makeModule(Ctx);
  llvm::Function *Function =
      makeVoidFunction(*Module, "scoped_marker", "gfx1100");
  IRBuilder<> Builder(Function->getEntryBlock().getTerminator());
  Builder.CreateCall(makeNamedMarkerSentinel(*Module, "sqtt_marker_point"),
                     {makeMarkerString(*Module, "scoped")});
  if (Sync)
    Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
        Module.get(), Intrinsic::amdgcn_s_barrier));

  SQTTConfig Config = fullScopeConfig();
  Config.CuMask = 0x1;
  if (Early)
    runPass(*Module, Config, SQTTInstrumentPass::Mode::Early);
  runPass(*Module, Config);
  EXPECT_EQ(countIntrinsicCalls(*Function, Intrinsic::amdgcn_sched_barrier),
            Sync ? 1u : 0u);
  if (Sync)
    expectScopedSkipPin(*Function);
}

static void addExistingLlvmUsed(llvm::Module &Module) {
  LLVMContext &Ctx = Module.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  GlobalVariable *Dummy =
      new GlobalVariable(Module, I32, false, GlobalValue::InternalLinkage,
                         ConstantInt::get(I32, 0), "existing_used_global");
  Constant *DummyPtr = ConstantExpr::getPointerBitCastOrAddrSpaceCast(
      Dummy, PointerType::getUnqual(Ctx));
  ArrayType *UsedTy = ArrayType::get(PointerType::getUnqual(Ctx), 1);
  GlobalVariable *Used =
      new GlobalVariable(Module, UsedTy, false, GlobalValue::AppendingLinkage,
                         ConstantArray::get(UsedTy, {DummyPtr}), "llvm.used");
  Used->setSection("llvm.metadata");
}

static void expectContains(const std::string &Text, StringRef Needle) {
  EXPECT_NE(Text.find(Needle.str()), std::string::npos)
      << "missing: " << Needle.str();
}

static void expectNotContains(const std::string &Text, StringRef Needle) {
  EXPECT_EQ(Text.find(Needle.str()), std::string::npos)
      << "unexpected: " << Needle.str();
}

template <typename Visitor>
static void forEachFuncmapLine(StringRef FuncMap, Visitor Visit) {
  SmallVector<StringRef, 32> Lines;
  FuncMap.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Line : Lines)
    Visit(Line.rtrim("\r"));
}

static std::vector<unsigned> pointEntryIds(const std::string &FuncMap,
                                           StringRef Name) {
  std::vector<unsigned> Ids;
  forEachFuncmapLine(FuncMap, [&](StringRef Line) {
    if (!Line.consume_front("P:"))
      return;
    auto [IdText, Rest] = Line.split(':');
    unsigned Id = 0;
    if (IdText.getAsInteger(10, Id))
      return;
    auto [EntryName, SourceLoc] = Rest.split('@');
    (void)SourceLoc;
    if (EntryName == Name)
      Ids.push_back(Id);
  });
  return Ids;
}

static std::optional<unsigned> pointEntryId(const std::string &FuncMap,
                                            StringRef Name) {
  std::vector<unsigned> Ids = pointEntryIds(FuncMap, Name);
  return Ids.empty() ? std::nullopt : std::optional<unsigned>(Ids.front());
}

static size_t countPointEntries(const std::string &FuncMap, StringRef Name) {
  return pointEntryIds(FuncMap, Name).size();
}

static std::optional<unsigned>
extraPayloadCountForId(const std::string &FuncMap, unsigned MarkerId) {
  std::optional<unsigned> Result;
  forEachFuncmapLine(FuncMap, [&](StringRef Line) {
    if (Result || !Line.consume_front("R:"))
      return;
    auto [IdText, Metadata] = Line.split(':');
    unsigned Id = 0;
    if (IdText.getAsInteger(10, Id) || Id != MarkerId)
      return;
    SmallVector<StringRef, 4> Fields;
    Metadata.split(Fields, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef Field : Fields) {
      if (!Field.consume_front("extra_payload_count="))
        continue;
      unsigned Count = 0;
      if (!Field.getAsInteger(10, Count))
        Result = Count;
    }
  });
  return Result;
}

static void expectPointEntryWithPayload(const std::string &FuncMap,
                                        StringRef Name,
                                        unsigned ExpectedPayloadCount) {
  std::optional<unsigned> Id = pointEntryId(FuncMap, Name);
  ASSERT_TRUE(Id.has_value()) << "missing point funcmap entry: " << Name.str();

  std::optional<unsigned> PayloadCount = extraPayloadCountForId(FuncMap, *Id);
  ASSERT_TRUE(PayloadCount.has_value())
      << "missing payload metadata for funcmap entry: " << Name.str();
  EXPECT_EQ(*PayloadCount, ExpectedPayloadCount)
      << "wrong payload metadata for funcmap entry: " << Name.str();
}

namespace {

class MarkerPass : public ::testing::Test {
protected:
  LLVMContext Ctx;
  std::unique_ptr<Module> TestModule = makeModule(Ctx);
};

TEST(MarkerConfig, ParsesEnvironmentAndRejectsConflictingModes) {
  std::vector<std::unique_ptr<ScopedEnv>> Env = clearSqttEnvironment();
  for (const auto &[Name, Value] :
       {std::pair{"SQTT_INSTRUMENT_BARRIERS", "YES"},
        std::pair{"SQTT_MEM_BARRIER", "clobber"},
        std::pair{"SQTT_INSTRUMENT_FUNCTIONS", "cost:42"},
        std::pair{"SQTT_INSTRUMENT_MEMORY", "4:7"},
        std::pair{"SQTT_TRACE_ADDRESSES", "memory, lds, bogus"},
        std::pair{"SQTT_SHADER_CLOCK_BITS", "not-a-number"},
        std::pair{"SQTT_SHADER_CLOCK_SHIFT", "8"},
        std::pair{"SQTT_SCOPE_WAVE", "not-a-mask"},
        std::pair{"SQTT_SCOPE_SIMD", "0x5"}, std::pair{"SQTT_SCOPE_CU", "-1"}})
    Env.push_back(std::make_unique<ScopedEnv>(Name, Value));

  SQTTConfig Config = SQTTConfig::fromEnvironment();

  EXPECT_TRUE(Config.InstrumentBarriers);
  EXPECT_EQ(Config.MemBarrier, MemBarrierMode::AsmClobber);
  EXPECT_EQ(Config.Mode, CostMode::WeightedCost);
  EXPECT_EQ(Config.FunctionThreshold, 42u);
  EXPECT_NE(Config.MemoryChunkSize, 0u);
  EXPECT_EQ(Config.MemoryChunkSize, 4u);
  EXPECT_EQ(Config.MemoryMaxGap, 7u);
  EXPECT_FALSE(Config.TraceMemoryAddrs);
  EXPECT_FALSE(Config.TraceLDSAddrs);
  EXPECT_EQ(Config.ShaderClockBits, 0u);
  EXPECT_EQ(Config.ShaderClockShift, 8u);
  EXPECT_EQ(Config.WaveMask, 0xFFFFFFFFu);
  EXPECT_EQ(Config.SimdMask, 0x5u);
  EXPECT_EQ(Config.CuMask, 0xFFFFFFFFu);

  for (const auto &[Name, Value] : {std::pair{"SQTT_INSTRUMENT_MEMORY", "4"},
                                    std::pair{"SQTT_TRACE_ADDRESSES", "lds"},
                                    std::pair{"SQTT_MEM_BARRIER", "bad-mode"}})
    Env.push_back(std::make_unique<ScopedEnv>(Name, Value));
  Config = SQTTConfig::fromEnvironment();
  EXPECT_EQ(Config.MemBarrier, MemBarrierMode::Fence);
  EXPECT_EQ(Config.MemoryChunkSize, 0u);
  EXPECT_TRUE(Config.TraceLDSAddrs);
  EXPECT_FALSE(Config.TraceMemoryAddrs);
}

TEST(MarkerTarget, ClassifiesArchitecturesAndInstructionCosts) {
  LLVMContext Ctx;
  std::unique_ptr<Module> LocalModule = makeModule(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);

  for (const auto &[Cpu, Expected] :
       {std::pair{"gfx90a", GfxGen::GFX9}, std::pair{"gfx1030", GfxGen::RDNA},
        std::pair{"gfx1100", GfxGen::RDNA}, std::pair{"gfx1200", GfxGen::GFX12},
        std::pair{"notgfx", GfxGen::Unknown}})
    EXPECT_EQ(getGfxGen(*makeVoidFunction(*LocalModule, Cpu, Cpu)), Expected);
  for (const auto &[Cpu, Expected] :
       {std::pair{"gfx1200", false}, std::pair{"gfx1201", false},
        std::pair{"gfx1202", true}, std::pair{"gfx1250", true},
        std::pair{"gfx1250:xnack+", true}})
    EXPECT_EQ(hasShaderCyclesU64(*makeVoidFunction(*LocalModule, Cpu, Cpu)),
              Expected);

  EXPECT_EQ(getWaveSize(GfxGen::GFX9), 64u);
  EXPECT_EQ(getWaveSize(GfxGen::RDNA), 32u);
  EXPECT_FALSE(supportsImmTrace(GfxGen::GFX9));
  EXPECT_TRUE(supportsImmTrace(GfxGen::GFX12));

  llvm::Function *Costed =
      makeFunction(*LocalModule, "costed", "gfx1100",
                   FunctionType::get(Type::getVoidTy(Ctx), {I32}, false));
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Costed);
  IRBuilder<> Builder(Entry);
  Builder.CreateAlloca(I32);
  Value *Loaded =
      Builder.CreateLoad(I32, UndefValue::get(PointerType::get(Ctx, 1)));
  Builder.CreateStore(Loaded, UndefValue::get(PointerType::get(Ctx, 3)));
  Builder.CreateCall(
      declareFunction(*LocalModule, "llvm.amdgcn.mfma.unit", I32, {}));
  Builder.CreateRetVoid();

  EXPECT_EQ(computeFunctionSize(*Costed, CostMode::InstructionCount), 4u);
  EXPECT_EQ(computeFunctionSize(*Costed, CostMode::WeightedCost), 31u);
}

TEST_F(MarkerPass, FuncmapLedgerPreservesProtocolOrderAndDebugLocations) {
  Type *I32 = Type::getInt32Ty(Ctx);
  llvm::Function *Device = makeFunction(*TestModule, "ledger_device", "gfx1100",
                                        FunctionType::get(I32, {I32}, false));
  IRBuilder<> DeviceBuilder(BasicBlock::Create(Ctx, "entry", Device));
  Value *Sum =
      DeviceBuilder.CreateAdd(Device->getArg(0), ConstantInt::get(I32, 1));
  Sum = DeviceBuilder.CreateAdd(Sum, ConstantInt::get(I32, 2));
  DeviceBuilder.CreateRet(Sum);

  llvm::Function *Kernel = makeFunction(
      *TestModule, "ledger_kernel", "gfx1100",
      FunctionType::get(Type::getVoidTy(Ctx),
                        {PointerType::get(Ctx, 1), PointerType::get(Ctx, 3)},
                        false));
  Kernel->setCallingConv(CallingConv::AMDGPU_KERNEL);

  TestModule->addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                            DEBUG_METADATA_VERSION);
  DIBuilder Debug(*TestModule);
  DIFile *File = Debug.createFile("ledger.hip", "/source");
  DICompileUnit *Unit = Debug.createCompileUnit(
      dwarf::DW_LANG_C_plus_plus_14, File, "marker-unit", false, "", 0);
  DISubroutineType *DebugType =
      Debug.createSubroutineType(Debug.getOrCreateTypeArray({}));
  DISubprogram *KernelScope = Debug.createFunction(
      Unit, "ledger_kernel", "ledger_kernel", File, 10, DebugType, 10,
      DINode::FlagZero, DISubprogram::SPFlagDefinition);
  DISubprogram *InlinedScope = Debug.createFunction(
      Unit, "inlined_load", "inlined_load", File, 7, DebugType, 7,
      DINode::FlagZero, DISubprogram::SPFlagDefinition);
  Kernel->setSubprogram(KernelScope);

  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", Kernel));
  llvm::Function *Enter =
      makeNamedMarkerSentinel(*TestModule, "sqtt_marker_enter");
  llvm::Function *Exit =
      makeNamedMarkerSentinel(*TestModule, "sqtt_marker_exit");
  GlobalVariable *ScopeName = makeMarkerString(*TestModule, "ledger_scope");
  Builder.CreateCall(Enter, {ScopeName});
  Instruction *GlobalLoad = Builder.CreateLoad(I32, Kernel->getArg(0));
  Instruction *LdsStore = Builder.CreateStore(GlobalLoad, Kernel->getArg(1));
  Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
      TestModule.get(), Intrinsic::amdgcn_s_barrier));
  Builder.CreateCall(Exit, {ScopeName});
  Builder.CreateRetVoid();

  DILocation *CallSite = DILocation::get(Ctx, 20, 1, KernelScope);
  GlobalLoad->setDebugLoc(DILocation::get(Ctx, 7, 1, InlinedScope, CallSite));
  LdsStore->setDebugLoc(DILocation::get(Ctx, 30, 1, KernelScope));
  Debug.finalize();

  SQTTConfig Config = fullScopeConfig();
  Config.FunctionThreshold = 1;
  Config.InstrumentBarriers = true;
  Config.TraceMemoryAddrs = Config.TraceLDSAddrs = true;
  const std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);

  EXPECT_FALSE(verifyModule(*TestModule));
  const size_t FunctionRow = FuncMap.find("F:4:ledger_device");
  const size_t KernelRow = FuncMap.find("K:ledger_kernel@ledger.hip:10");
  const size_t NamedRow = FuncMap.find("U:5:ledger_scope");
  const size_t SystemRow = FuncMap.find("P:1:barrier_signal");
  const size_t WaveRow = FuncMap.find("W:32");
  const size_t GlobalRow =
      FuncMap.find("P:6:addr_trace_load@ledger.hip:7 -> ledger.hip:20");
  const size_t LdsRow = FuncMap.find("P:7:addr_trace_lds_store@ledger.hip:30");
  for (size_t Row : {FunctionRow, KernelRow, NamedRow, SystemRow, WaveRow,
                     GlobalRow, LdsRow})
    ASSERT_NE(Row, std::string::npos) << FuncMap;
  EXPECT_LT(FunctionRow, KernelRow);
  EXPECT_LT(KernelRow, NamedRow);
  EXPECT_LT(NamedRow, SystemRow);
  EXPECT_LT(SystemRow, WaveRow);
  EXPECT_LT(WaveRow, GlobalRow);
  EXPECT_LT(GlobalRow, LdsRow);
  EXPECT_EQ(extraPayloadCountForId(FuncMap, 6), std::optional<unsigned>(66));
  EXPECT_EQ(extraPayloadCountForId(FuncMap, 7), std::optional<unsigned>(34));
}

TEST_F(MarkerPass, AddressTracingCoversBufferProtocolsAcrossWaveSizes) {
  struct Case {
    const char *Cpu;
    const char *WaveSize;
    const char *PermuteName;
    unsigned Lanes;
    bool Barriers;
  };
  for (const Case &Test :
       {Case{"gfx1100", "W:32", "addr_trace_ds_bpermute", 32, true},
        Case{"gfx90a", "W:64", "addr_trace_ds_permute", 64, false}}) {
    SCOPED_TRACE(Test.Cpu);
    LLVMContext Ctx;
    std::unique_ptr<Module> BufferModule = makeModule(Ctx);
    llvm::Function *Function =
        makeBufferTraceFunction(*BufferModule, Test.Cpu, Test.Barriers);
    SQTTConfig Config = fullScopeConfig();
    Config.InstrumentBarriers = Test.Barriers;
    Config.TraceMemoryAddrs = Config.TraceLDSAddrs = true;

    const std::string FuncMap = runPassAndGetFuncMap(*BufferModule, Config);
    expectContains(FuncMap, Test.WaveSize);
    expectPointEntryWithPayload(FuncMap, "addr_trace_buffer_load",
                                5 + Test.Lanes);
    expectPointEntryWithPayload(FuncMap, "addr_trace_struct_buffer_store",
                                5 + 2 * Test.Lanes);
    expectPointEntryWithPayload(FuncMap, "addr_trace_buffer_atomic",
                                5 + Test.Lanes);
    expectPointEntryWithPayload(FuncMap, Test.PermuteName, 2 + Test.Lanes);
    EXPECT_EQ(countPointEntries(FuncMap, "addr_trace_load"), 1u);
    EXPECT_EQ(countPointEntries(FuncMap, "addr_trace_store"), 1u);
    EXPECT_EQ(countPointEntries(FuncMap, "addr_trace_atomic"), 0u);
    if (Test.Barriers) {
      EXPECT_LT(FuncMap.find("barrier_signal"), FuncMap.find(Test.WaveSize));
      EXPECT_LT(FuncMap.find(Test.WaveSize),
                FuncMap.find("addr_trace_buffer_load"));
    }

    const std::string Ir = printModule(*BufferModule);
    expectContains(Ir, "sqtt.lanes.loop");
    expectContains(Ir, "s_mov_b32 m0, exec_lo");
    expectContains(Ir, "s_nop 0");
    expectContains(Ir, "s_ttracedata");
    expectContains(Ir, "={m0}");
    bool HasDescriptorPtrToInt = false;
    for (const llvm::Function &Candidate : *BufferModule)
      for (const BasicBlock &Block : Candidate)
        for (const Instruction &Instruction : Block)
          HasDescriptorPtrToInt |=
              Instruction.getOpcode() == Instruction::PtrToInt &&
              Instruction.getType()->getIntegerBitWidth() == 128 &&
              Instruction.getOperand(0)->getType()->getPointerAddressSpace() ==
                  8;
    EXPECT_TRUE(HasDescriptorPtrToInt);
    EXPECT_EQ(countIntrinsicCalls(*BufferModule, Intrinsic::amdgcn_readlane),
              9u);
    std::vector<uint32_t> Readlanes;
    forEachCall(*Function, [&](const CallInst &Call) {
      if (const llvm::Function *Callee = Call.getCalledFunction();
          Callee && Callee->getIntrinsicID() == Intrinsic::amdgcn_readlane)
        if (const auto *Value = dyn_cast<ConstantInt>(Call.getArgOperand(0)))
          Readlanes.push_back(Value->getZExtValue());
    });
    EXPECT_EQ(Readlanes, (std::vector<uint32_t>{11, 7, 5, 4, 16}));
  }

  LLVMContext ScopeCtx;
  std::unique_ptr<Module> ScopeModule = makeModule(ScopeCtx);
  llvm::Function *Scoped =
      makeGlobalLoadFunction(*ScopeModule, "scoped_address_trace", "gfx1100");
  SQTTConfig ScopeConfig = fullScopeConfig();
  ScopeConfig.CuMask = 0x1;
  ScopeConfig.MemBarrier = MemBarrierMode::Fence;
  ScopeConfig.TraceMemoryAddrs = true;
  runPass(*ScopeModule, ScopeConfig);
  EXPECT_EQ(countFences(*Scoped), 2u);
  EXPECT_EQ(countIntrinsicCalls(*Scoped, Intrinsic::amdgcn_sched_barrier), 0u);
}

TEST_F(MarkerPass, PayloadMarkersControlGfx12ClockPacking) {
  llvm::Function *Function =
      makeGlobalLoadFunction(*TestModule, "gfx12_address_trace", "gfx1200");
  SQTTConfig Config = fullScopeConfig();
  Config.MemBarrier = MemBarrierMode::Fence;
  Config.TraceMemoryAddrs = true;

  const std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);
  expectContains(FuncMap, "W:32");
  expectPointEntryWithPayload(FuncMap, "addr_trace_load", 66);
  expectNotContains(FuncMap, "M:shader_clock_bits=");
  EXPECT_EQ(countIntrinsicCalls(*Function, Intrinsic::amdgcn_s_getreg), 0u);
  EXPECT_EQ(countFences(*Function), 2u);
  EXPECT_EQ(countIntrinsicCalls(*Function, Intrinsic::amdgcn_sched_barrier),
            2u);

  // buffer.load.lds must not create an address payload block, so packing
  // remains valid.
  LLVMContext LdsCtx;
  std::unique_ptr<Module> LdsModule = makeModule(LdsCtx);
  llvm::Function *LdsFunction =
      makeVoidFunction(*LdsModule, "buffer_load_lds", "gfx1200");
  IRBuilder<> LdsBuilder(LdsFunction->getEntryBlock().getTerminator());
  llvm::Function *Point =
      makeNamedMarkerSentinel(*LdsModule, "sqtt_marker_point");
  LdsBuilder.CreateCall(declareFunction(*LdsModule,
                                        "llvm.amdgcn.raw.buffer.load.lds",
                                        Type::getVoidTy(LdsCtx), {}));
  LdsBuilder.CreateCall(Point,
                        {makeMarkerString(*LdsModule, "ordinary_point")});
  // An ordinary numeric shaderdata value must stay legacy even while
  // pass-owned headers carry clock bits.
  insertTraceCallBefore(LdsFunction->getEntryBlock().getTerminator(),
                        encodeMarker(1u << 20, false, false));
  llvm::Function *U64Function =
      makeVoidFunction(*LdsModule, "shader_cycles_u64", "gfx1250");
  IRBuilder<>(U64Function->getEntryBlock().getTerminator())
      .CreateCall(Point, {makeMarkerString(*LdsModule, "ordinary_point_u64")});
  SQTTConfig LdsConfig = fullScopeConfig();
  LdsConfig.TraceMemoryAddrs = true;
  LdsConfig.ShaderClockBits = 12;
  const std::string LdsMap = runPassAndGetFuncMap(*LdsModule, LdsConfig);
  expectContains(LdsMap, "M:shader_clock_bits=12;shader_clock_shift=4");
  expectNotContains(LdsMap, "addr_trace_buffer_load");
  expectNotContains(LdsMap, "W:");
  EXPECT_EQ(countIntrinsicCalls(*LdsFunction, Intrinsic::amdgcn_s_getreg), 1u);
  expectContains(printModule(*LdsModule), "s_get_shader_cycles_u64 $0");

  for (bool NamedData : {false, true}) {
    SCOPED_TRACE(NamedData ? "named data" : "address block");
    EXPECT_DEATH(
        {
          LLVMContext LocalCtx;
          std::unique_ptr<Module> LocalModule = makeModule(LocalCtx);
          SQTTConfig LocalConfig = fullScopeConfig();
          if (NamedData) {
            Type *I32 = Type::getInt32Ty(LocalCtx);
            llvm::Function *Function =
                makeVoidFunction(*LocalModule, "gfx12_named_data", "gfx1200");
            IRBuilder<>(Function->getEntryBlock().getTerminator())
                .CreateCall(
                    declareFunction(*LocalModule, "sqtt_marker_data",
                                    Type::getVoidTy(LocalCtx),
                                    {PointerType::get(LocalCtx, 0), I32}),
                    {makeMarkerString(*LocalModule, "one_payload"),
                     ConstantInt::get(I32, 17)});
          } else {
            makeGlobalLoadFunction(*LocalModule, "forced_clock_address_trace",
                                   "gfx1200");
            LocalConfig.TraceMemoryAddrs = true;
          }
          LocalConfig.ShaderClockBits = 12;
          runPass(*LocalModule, LocalConfig);
        },
        "sqtt payload markers require SQTT_SHADER_CLOCK_BITS=0");
  }
}

TEST_F(MarkerPass, ShaderClockPackingLeavesUnregisteredNumericDataUntouched) {
  Type *I32 = Type::getInt32Ty(Ctx);
  llvm::Function *Function =
      makeFunction(*TestModule, "mixed_clock_headers", "gfx1200",
                   FunctionType::get(I32, {I32}, false));
  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", Function));
  Value *DynamicValue =
      Builder.CreateAdd(Function->getArg(0), ConstantInt::get(I32, 1));
  Value *Result = Builder.CreateAdd(DynamicValue, ConstantInt::get(I32, 2));
  ReturnInst *Ret = Builder.CreateRet(Result);

  const uint32_t NumericValue = encodeMarker(1u << 20, false, false);
  insertTraceCallBefore(Ret, NumericValue);
  IRBuilder<>(Ret).CreateCall(
      Intrinsic::getOrInsertDeclaration(TestModule.get(),
                                        Intrinsic::amdgcn_s_ttracedata),
      {DynamicValue});

  SQTTConfig Config = fullScopeConfig();
  Config.FunctionThreshold = 1;
  Config.ShaderClockBits = 12;
  const std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);

  // Only the pass-generated function entry and exit are clock-packed. The
  // constant numeric ID is deliberately too large for the packed layout,
  // and the dynamic value has no funcmap identity at all.
  EXPECT_EQ(countIntrinsicCalls(*Function, Intrinsic::amdgcn_s_getreg), 2u);
  expectContains(FuncMap, "M:shader_clock_bits=12;shader_clock_shift=4");
  expectContains(FuncMap, "F:1:mixed_clock_headers");

  bool FoundConstant = false, FoundDynamic = false;
  forEachCall(*Function, [&](const CallInst &Call) {
    const auto *AsmCall = dyn_cast<InlineAsm>(Call.getCalledOperand());
    if (!AsmCall || !AsmCall->getAsmString().contains("s_ttracedata"))
      return;
    if (const auto *Value = dyn_cast<ConstantInt>(Call.getArgOperand(0)))
      FoundConstant |= Value->getZExtValue() == NumericValue;
    FoundDynamic |= Call.getArgOperand(0) == DynamicValue;
  });
  EXPECT_TRUE(FoundConstant);
  EXPECT_TRUE(FoundDynamic);
}

TEST_F(MarkerPass,
       ShaderClockPackingRejectsInvalidLayoutsAndOversizedHeaderIds) {
  struct LayoutCase {
    unsigned Bits;
    unsigned Shift;
    const char *Message;
  };
  for (const LayoutCase &Test :
       {LayoutCase{30, 0, "leave at least one marker ID bit"},
        LayoutCase{12, 21, "window must fit"}}) {
    SCOPED_TRACE(Test.Message);
    EXPECT_DEATH(
        {
          LLVMContext LocalCtx;
          std::unique_ptr<Module> LocalModule = makeModule(LocalCtx);
          makeVoidFunction(*LocalModule, "invalid_clock_layout", "gfx1200");
          SQTTConfig LocalConfig = fullScopeConfig();
          LocalConfig.ShaderClockBits = Test.Bits;
          LocalConfig.ShaderClockShift = Test.Shift;
          runPass(*LocalModule, LocalConfig);
        },
        Test.Message);
  }

  EXPECT_DEATH(
      {
        LLVMContext LocalCtx;
        std::unique_ptr<Module> LocalModule = makeModule(LocalCtx);
        llvm::Function *Function =
            makeVoidFunction(*LocalModule, "oversized_clock_id", "gfx1200");
        insertTraceCallBefore(Function->getEntryBlock().getTerminator(),
                              encodeMarker(1u << 18, false, false), true);
        SQTTConfig LocalConfig = fullScopeConfig();
        LocalConfig.ShaderClockBits = 12;
        runPass(*LocalModule, LocalConfig);
      },
      "marker ID .* does not fit");
}

TEST_F(MarkerPass, NumericMarkerLoweringAndBoundaries) {
  const uint32_t MarkerValue = encodeMarker(64, false, false);
  MDNode *TraceMetadata = MDNode::get(Ctx, {});

  std::vector<llvm::Function *> Functions;
  for (const auto &[Name, Cpu] :
       {std::pair{"gfx9_trace", "gfx90a"}, std::pair{"gfx10_trace", "gfx1030"},
        std::pair{"gfx12_trace", "gfx1200"}}) {
    llvm::Function *Function = makeVoidFunction(*TestModule, Name, Cpu);
    CallInst *Trace = insertTraceCallBefore(
        Function->getEntryBlock().getTerminator(), MarkerValue);
    Trace->setMetadata("sqtt.test.trace", TraceMetadata);
    Functions.push_back(Function);
  }

  SQTTConfig Config = fullScopeConfig();

  runPass(*TestModule, Config);

  for (llvm::Function *Function : Functions) {
    const CallInst *Trace = findM0NopTrace(*Function);
    ASSERT_NE(Trace, nullptr) << Function->getName().str();
    auto *TraceAsm = dyn_cast<InlineAsm>(Trace->getCalledOperand());
    ASSERT_NE(TraceAsm, nullptr);
    EXPECT_EQ(TraceAsm->getConstraintString(), "={m0},i");
    EXPECT_EQ(Trace->getMetadata("sqtt.test.trace"), TraceMetadata);
  }

  LLVMContext FenceCtx;
  std::unique_ptr<Module> FenceModule = makeModule(FenceCtx);
  llvm::Function *Fenced =
      makeVoidFunction(*FenceModule, "numeric_marker", "gfx1100");
  insertTraceCallBefore(Fenced->getEntryBlock().getTerminator(),
                        encodeMarker(17, false, false));
  SQTTConfig FenceConfig = fullScopeConfig();
  FenceConfig.MemBarrier = MemBarrierMode::Fence;
  runPass(*FenceModule, FenceConfig);
  EXPECT_EQ(countFences(*Fenced), 2u);

  LLVMContext ScopeCtx;
  std::unique_ptr<Module> ScopeModule = makeModule(ScopeCtx);
  llvm::Function *Scoped =
      makeVoidFunction(*ScopeModule, "scoped_numeric_markers", "gfx1100");
  Type *ScopedI32 = Type::getInt32Ty(ScopeCtx);
  IRBuilder<> ScopedBuilder(Scoped->getEntryBlock().getTerminator());
  FunctionCallee Hint = Intrinsic::getOrInsertDeclaration(
      ScopeModule.get(), Intrinsic::amdgcn_sched_barrier);
  FunctionCallee Trace = Intrinsic::getOrInsertDeclaration(
      ScopeModule.get(), Intrinsic::amdgcn_s_ttracedata);
  for (uint32_t Id : {17u, 18u}) {
    ScopedBuilder.CreateCall(Hint, {ConstantInt::get(ScopedI32, 0)});
    ScopedBuilder.CreateCall(
        Trace, {ConstantInt::get(ScopedI32, encodeMarker(Id, false, false))});
    ScopedBuilder.CreateCall(Hint, {ConstantInt::get(ScopedI32, 0)});
  }
  ScopedBuilder.CreateCall(Intrinsic::getOrInsertDeclaration(
      ScopeModule.get(), Intrinsic::amdgcn_s_barrier));
  SQTTConfig ScopeConfig;
  ScopeConfig.CuMask = 0x1;
  ScopeConfig.MemBarrier = MemBarrierMode::Fence;
  runPass(*ScopeModule, ScopeConfig);
  size_t TraceBlocks = 0;
  for (const BasicBlock &Block : *Scoped)
    TraceBlocks += Block.getName() == "sqtt.trace";
  EXPECT_EQ(TraceBlocks, 1u);
  EXPECT_EQ(countFences(*Scoped), 4u);
  EXPECT_EQ(countIntrinsicCalls(*Scoped, Intrinsic::amdgcn_sched_barrier), 1u);
  expectScopedSkipPin(*Scoped);
}

TEST_F(MarkerPass,
       ScopedMarkerCoalescingKeepsUserSchedulerBarriersUnconditional) {
  llvm::Function *Function =
      makeVoidFunction(*TestModule, "scoped_user_sched_barrier", "gfx1100");
  Instruction *Ret = Function->getEntryBlock().getTerminator();
  IRBuilder<> Builder(Ret);
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionCallee Trace = Intrinsic::getOrInsertDeclaration(
      TestModule.get(), Intrinsic::amdgcn_s_ttracedata);
  FunctionCallee Sched = Intrinsic::getOrInsertDeclaration(
      TestModule.get(), Intrinsic::amdgcn_sched_barrier);
  Builder.CreateCall(Trace,
                     {ConstantInt::get(I32, encodeMarker(17, false, false))});
  CallInst *UserBarrier = Builder.CreateCall(Sched, {ConstantInt::get(I32, 1)});
  Builder.CreateCall(Trace,
                     {ConstantInt::get(I32, encodeMarker(18, false, false))});

  SQTTConfig Config = fullScopeConfig();
  Config.CuMask = 0x1;
  runPass(*TestModule, Config);

  EXPECT_FALSE(verifyModule(*TestModule));
  size_t TraceBlocks = 0;
  for (const BasicBlock &Block : *Function)
    TraceBlocks += Block.getName().starts_with("sqtt.trace");
  EXPECT_EQ(TraceBlocks, 2u);
  EXPECT_FALSE(UserBarrier->getParent()->getName().starts_with("sqtt.trace"));
  EXPECT_EQ(countIntrinsicCalls(*Function, Intrinsic::amdgcn_sched_barrier),
            1u);
}

TEST_F(MarkerPass, ScopedMarkerBoundariesStayUniformAndPayloadsStayAtomic) {
  SQTTConfig Config = fullScopeConfig();
  Config.CuMask = 0x1;
  for (const auto &[Early, Sync] :
       {std::tuple{false, false}, std::tuple{true, false},
        std::tuple{false, true}, std::tuple{true, true}})
    expectScopedMarkerCase(Early, Sync);

  struct PayloadCase {
    bool Scoped, TailUse;
    const char *Name;
  };
  for (const PayloadCase &Test :
       {PayloadCase{false, false, "full data"},
        PayloadCase{true, false, "scoped data"},
        PayloadCase{true, true, "scoped data with tail use"}}) {
    SCOPED_TRACE(Test.Name);
    LLVMContext PayloadCtx;
    std::unique_ptr<Module> PayloadModule = makeModule(PayloadCtx);
    Type *I32 = Type::getInt32Ty(PayloadCtx);
    llvm::Function *PayloadFunction = makeFunction(
        *PayloadModule, "named_data", "gfx1100",
        FunctionType::get(Test.TailUse ? I32 : Type::getVoidTy(PayloadCtx),
                          {I32}, false));
    IRBuilder<> PayloadBuilder(
        BasicBlock::Create(PayloadCtx, "entry", PayloadFunction));
    PayloadBuilder.CreateCall(
        declareFunction(*PayloadModule, "sqtt_marker_data",
                        Type::getVoidTy(PayloadCtx),
                        {PointerType::get(PayloadCtx, 0), I32}),
        {makeMarkerString(*PayloadModule, "payload"),
         PayloadFunction->getArg(0)});
    ReturnInst *Tail =
        Test.TailUse ? PayloadBuilder.CreateRet(PayloadFunction->getArg(0))
                     : PayloadBuilder.CreateRetVoid();
    SQTTConfig PayloadConfig = Test.Scoped ? Config : fullScopeConfig();
    PayloadConfig.MemBarrier = MemBarrierMode::Fence;
    runPass(*PayloadModule, PayloadConfig, SQTTInstrumentPass::Mode::Early);
    Instruction *UserBetween = nullptr;
    if (Test.TailUse) {
      CallInst *EarlyPayload =
          findTraceWithMetadata(*PayloadFunction, "sqtt.raw_payload");
      ASSERT_NE(EarlyPayload, nullptr);
      UserBetween = cast<Instruction>(
          IRBuilder<>(EarlyPayload)
              .CreateAdd(PayloadFunction->getArg(0), ConstantInt::get(I32, 1)));
      Tail->setOperand(0, UserBetween);
    }
    runPass(*PayloadModule, PayloadConfig);

    EXPECT_FALSE(verifyModule(*PayloadModule));
    CallInst *Header =
        findTraceWithMetadata(*PayloadFunction, "sqtt.marker_header");
    CallInst *Payload =
        findTraceWithMetadata(*PayloadFunction, "sqtt.raw_payload");
    ASSERT_NE(Header, nullptr);
    ASSERT_NE(Payload, nullptr);
    if (Test.TailUse) {
      EXPECT_FALSE(
          UserBetween->getParent()->getName().starts_with("sqtt.trace"));
      continue;
    }
    ASSERT_EQ(Header->getParent(), Payload->getParent());
    EXPECT_EQ(countFences(*PayloadFunction), 2u);
    EXPECT_EQ(
        countIntrinsicCalls(*PayloadFunction, Intrinsic::amdgcn_sched_barrier),
        Test.Scoped ? 0u : 2u);
    for (Instruction *Instruction = Header->getNextNode();
         Instruction != Payload; Instruction = Instruction->getNextNode()) {
      ASSERT_NE(Instruction, nullptr);
      EXPECT_FALSE(isa<FenceInst>(Instruction));
      auto *Call = dyn_cast<CallInst>(Instruction);
      llvm::Function *Callee = Call ? Call->getCalledFunction() : nullptr;
      EXPECT_FALSE(Callee &&
                   Callee->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier);
      EXPECT_FALSE(
          Callee &&
          (Callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata ||
           Callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata_imm));
      if (auto *AsmCall =
              Call ? dyn_cast<InlineAsm>(Call->getCalledOperand()) : nullptr) {
        EXPECT_EQ(AsmCall->getAsmString().find("s_ttracedata"),
                  StringRef::npos);
      }
    }
  }
}

TEST_F(MarkerPass, MemoryInstrumentationPreservesChunksKindsAndGaps) {
  Type *I32 = Type::getInt32Ty(Ctx);
  llvm::Function *Function =
      makeFunction(*TestModule, "memory_chunks", "gfx1100",
                   FunctionType::get(Type::getVoidTy(Ctx),
                                     {PointerType::get(Ctx, 1)}, false));
  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", Function));
  Value *Pointer = Function->getArg(0);
  Instruction *Load1 = Builder.CreateLoad(I32, Pointer);
  Instruction *Load2 = Builder.CreateLoad(I32, Pointer);
  Instruction *Store1 = Builder.CreateStore(Load1, Pointer);
  Instruction *Store2 = Builder.CreateStore(Load2, Pointer);
  Instruction *GapLoad1 = Builder.CreateLoad(I32, Pointer);
  Builder.CreateAdd(GapLoad1, ConstantInt::get(I32, 1));
  Instruction *GapLoad2 = Builder.CreateLoad(I32, Pointer);
  Builder.CreateRetVoid();

  SQTTConfig Config = fullScopeConfig();
  Config.MemoryChunkSize = 2;
  Config.MemoryMaxGap = 0;
  const std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);
  std::optional<unsigned> LoadId = pointEntryId(FuncMap, "vmem_load");
  std::optional<unsigned> StoreId = pointEntryId(FuncMap, "vmem_store");
  ASSERT_TRUE(LoadId.has_value());
  ASSERT_TRUE(StoreId.has_value());
  EXPECT_EQ(*LoadId, 1u);
  EXPECT_EQ(*StoreId, 2u);

  uint32_t LoadMarker = encodeMarker(*LoadId, false, false);
  uint32_t StoreMarker = encodeMarker(*StoreId, false, false);
  EXPECT_EQ(
      traceMarkerValues(*Function),
      (std::vector<uint32_t>{LoadMarker, StoreMarker, LoadMarker, LoadMarker}));
  EXPECT_FALSE(markerAfter(Load1));
  EXPECT_EQ(markerAfter(Load2), std::optional<uint32_t>(LoadMarker));
  EXPECT_FALSE(markerAfter(Store1));
  EXPECT_EQ(markerAfter(Store2), std::optional<uint32_t>(StoreMarker));
  EXPECT_EQ(markerAfter(GapLoad1), std::optional<uint32_t>(LoadMarker));
  EXPECT_EQ(markerAfter(GapLoad2), std::optional<uint32_t>(LoadMarker));

  LLVMContext ScopeCtx;
  std::unique_ptr<Module> ScopeModule = makeModule(ScopeCtx);
  llvm::Function *Scoped =
      makeGlobalLoadFunction(*ScopeModule, "scoped_memory_chunks", "gfx1100");
  IRBuilder<> ScopedBuilder(Scoped->getEntryBlock().getTerminator());
  Type *ScopedI32 = Type::getInt32Ty(ScopeCtx);
  ScopedBuilder.CreateLoad(ScopedI32, Scoped->getArg(0));
  ScopedBuilder.CreateLoad(ScopedI32, Scoped->getArg(0));
  SQTTConfig ScopeConfig = fullScopeConfig();
  ScopeConfig.CuMask = 0x1;
  ScopeConfig.MemoryChunkSize = 1;
  runPass(*ScopeModule, ScopeConfig);
  EXPECT_EQ(traceMarkerValues(*Scoped).size(), 3u);
}

TEST_F(MarkerPass, BarrierInstrumentationHandlesSplitAndStandaloneBarriers) {
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I16 = Type::getInt16Ty(Ctx);

  llvm::Function *Function =
      makeVoidFunction(*TestModule, "barrier_traces", "gfx1100");
  Instruction *Ret = Function->getEntryBlock().getTerminator();
  IRBuilder<> Builder(Ret);

  FunctionCallee Signal = Intrinsic::getOrInsertDeclaration(
      TestModule.get(), Intrinsic::amdgcn_s_barrier_signal);
  FunctionCallee Wait = Intrinsic::getOrInsertDeclaration(
      TestModule.get(), Intrinsic::amdgcn_s_barrier_wait);
  FunctionCallee Full = Intrinsic::getOrInsertDeclaration(
      TestModule.get(), Intrinsic::amdgcn_s_barrier);
  llvm::Function *Work = llvm::Function::Create(
      FunctionType::get(Type::getVoidTy(Ctx), false),
      GlobalValue::ExternalLinkage, "barrier_work", TestModule.get());

  Builder.CreateCall(Signal, {ConstantInt::get(I32, 0)});
  Builder.CreateCall(Wait, {ConstantInt::get(I16, 0)});
  Builder.CreateCall(Signal, {ConstantInt::get(I32, 0)});
  Builder.CreateCall(Work);
  Builder.CreateCall(Wait, {ConstantInt::get(I16, 0)});
  Builder.CreateCall(Full);

  SQTTConfig Config = fullScopeConfig();
  Config.InstrumentBarriers = true;
  Config.MemoryChunkSize = 1;

  std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);
  std::optional<unsigned> SignalId = pointEntryId(FuncMap, "barrier_signal");
  std::optional<unsigned> WaitId = pointEntryId(FuncMap, "barrier_wait");
  std::optional<unsigned> FullId = pointEntryId(FuncMap, "barrier");
  ASSERT_TRUE(SignalId.has_value());
  ASSERT_TRUE(WaitId.has_value());
  ASSERT_TRUE(FullId.has_value());
  EXPECT_EQ(*SignalId, 1u);
  EXPECT_EQ(*WaitId, 2u);
  EXPECT_EQ(*FullId, 3u);
  EXPECT_EQ(pointEntryId(FuncMap, "vmem_load"), std::optional<unsigned>(4));
  EXPECT_EQ(pointEntryId(FuncMap, "vmem_store"), std::optional<unsigned>(5));

  size_t TraceCount =
      countIntrinsicCalls(*TestModule, Intrinsic::amdgcn_s_ttracedata) +
      countIntrinsicCalls(*TestModule, Intrinsic::amdgcn_s_ttracedata_imm);
  EXPECT_EQ(TraceCount, 4u);
}

TEST_F(MarkerPass, NamedExitEnterFusionRequiresDirectAdjacency) {
  Type *I32 = Type::getInt32Ty(Ctx);
  llvm::Function *Exit =
      makeNamedMarkerSentinel(*TestModule, "sqtt_marker_exit");
  llvm::Function *Enter =
      makeNamedMarkerSentinel(*TestModule, "sqtt_marker_enter");
  GlobalVariable *OldName = makeMarkerString(*TestModule, "old");
  GlobalVariable *NewName = makeMarkerString(*TestModule, "new");

  llvm::Function *Separated =
      makeFunction(*TestModule, "separated_named_markers", "gfx1100",
                   FunctionType::get(Type::getVoidTy(Ctx), {I32}, false));
  BasicBlock *SeparatedEntry = BasicBlock::Create(Ctx, "entry", Separated);
  IRBuilder<> SeparatedBuilder(SeparatedEntry);
  SeparatedBuilder.CreateCall(Exit, {OldName});
  SeparatedBuilder.CreateAdd(Separated->getArg(0), ConstantInt::get(I32, 1));
  SeparatedBuilder.CreateCall(Enter, {NewName});
  SeparatedBuilder.CreateRetVoid();

  llvm::Function *Adjacent =
      makeVoidFunction(*TestModule, "adjacent_named_markers", "gfx1100");
  Instruction *AdjacentRet = Adjacent->getEntryBlock().getTerminator();
  IRBuilder<> AdjacentBuilder(AdjacentRet);
  AdjacentBuilder.CreateCall(Exit, {OldName});
  AdjacentBuilder.CreateCall(Enter, {NewName});

  SQTTConfig Config = fullScopeConfig();

  runPass(*TestModule, Config);

  std::vector<uint32_t> SeparatedMarkers = traceMarkerValues(*Separated);
  ASSERT_EQ(SeparatedMarkers.size(), 2u);
  EXPECT_EQ(SeparatedMarkers[0], FlagExitPrev);
  EXPECT_EQ(SeparatedMarkers[1] & FlagMask, FlagEnter);

  std::vector<uint32_t> AdjacentMarkers = traceMarkerValues(*Adjacent);
  ASSERT_EQ(AdjacentMarkers.size(), 1u);
  EXPECT_EQ(AdjacentMarkers[0] & FlagMask, FlagEnter | FlagExitPrev);
}

TEST_F(MarkerPass, NamedMarkerWrappersInlineAtEveryCallSiteBeforeResolution) {
  llvm::Function *Point =
      makeNamedMarkerSentinel(*TestModule, "sqtt_marker_point");
  Type *Ptr = PointerType::get(Ctx, 0);
  llvm::Function *Wrapper =
      makeFunction(*TestModule, "user_marker_wrapper", "gfx1100",
                   FunctionType::get(Type::getVoidTy(Ctx), {Ptr}, false));
  Wrapper->setLinkage(GlobalValue::InternalLinkage);
  IRBuilder<> WrapperBuilder(BasicBlock::Create(Ctx, "entry", Wrapper));
  WrapperBuilder.CreateCall(Point, {Wrapper->getArg(0)});
  WrapperBuilder.CreateRetVoid();

  GlobalVariable *Name = makeMarkerString(*TestModule, "wrapped_point");
  std::vector<llvm::Function *> Callers;
  for (StringRef CallerName :
       {StringRef("wrapper_caller_one"), StringRef("wrapper_caller_two")}) {
    llvm::Function *Caller =
        makeVoidFunction(*TestModule, CallerName, "gfx1100");
    IRBuilder<>(Caller->getEntryBlock().getTerminator())
        .CreateCall(Wrapper, {Name});
    Callers.push_back(Caller);
  }

  SQTTConfig Config = fullScopeConfig();
  runPass(*TestModule, Config, SQTTInstrumentPass::Mode::Early);

  for (llvm::Function *Caller : Callers) {
    EXPECT_EQ(traceMarkerValues(*Caller),
              (std::vector<uint32_t>{encodeMarker(1, false, false)}));
    forEachCall(*Caller, [&](const CallInst &Call) {
      EXPECT_NE(Call.getCalledFunction(), Wrapper);
    });
  }
  const NamedMDNode *Early = TestModule->getNamedMetadata("sqtt.markers.early");
  ASSERT_NE(Early, nullptr);
  EXPECT_EQ(Early->getNumOperands(), 1u);
  EXPECT_FALSE(verifyModule(*TestModule));
}

TEST_F(MarkerPass, DirectFunctionInstrumentationHandlesO0Fallback) {
  Type *I32 = Type::getInt32Ty(Ctx);

  llvm::Function *Large = makeFunction(*TestModule, "direct_large", "gfx1100",
                                       FunctionType::get(I32, {I32}, false));
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Large);
  BasicBlock *ThenBlock = BasicBlock::Create(Ctx, "then", Large);
  BasicBlock *ElseBlock = BasicBlock::Create(Ctx, "else", Large);
  IRBuilder<> Builder(Entry);
  Value *Arg = Large->getArg(0);
  Builder.CreateCondBr(Builder.CreateICmpUGT(Arg, ConstantInt::get(I32, 10)),
                       ThenBlock, ElseBlock);
  Builder.SetInsertPoint(ThenBlock);
  Builder.CreateRet(Builder.CreateAdd(Arg, ConstantInt::get(I32, 1)));
  Builder.SetInsertPoint(ElseBlock);
  Builder.CreateRet(Builder.CreateSub(Arg, ConstantInt::get(I32, 1)));

  llvm::Function *Small =
      makeVoidFunction(*TestModule, "direct_small", "gfx1100");
  llvm::Function *Kernel =
      makeVoidFunction(*TestModule, "direct_kernel", "gfx1100");
  Kernel->setCallingConv(CallingConv::AMDGPU_KERNEL);
  llvm::Function *MustTail = makeMustTailFunction(*TestModule, "late_musttail");

  SQTTConfig Config = fullScopeConfig();
  Config.FunctionThreshold = 3;

  std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);
  expectContains(FuncMap, "F:1:direct_large");
  expectContains(FuncMap, "K:direct_kernel");
  expectNotContains(FuncMap, "direct_small");
  expectNotContains(FuncMap, "late_musttail");

  std::vector<uint32_t> LargeMarkers = traceMarkerValues(*Large);
  EXPECT_EQ(llvm::count(LargeMarkers, encodeMarker(1, true, false)), 1);
  EXPECT_EQ(llvm::count(LargeMarkers, FlagExitPrev), 2);
  EXPECT_TRUE(traceMarkerValues(*Small).empty());
  EXPECT_TRUE(traceMarkerValues(*Kernel).empty());
  EXPECT_TRUE(traceMarkerValues(*MustTail).empty());
}

TEST_F(MarkerPass,
       FunctionThresholdIgnoresPassMarkersBeforeAndAfterOptimization) {
  llvm::Function *Function =
      makeVoidFunction(*TestModule, "small_function", "gfx1100");

  SQTTConfig Config = fullScopeConfig();
  Config.FunctionThreshold = 1;

  runPass(*TestModule, Config, SQTTInstrumentPass::Mode::Early);

  const NamedMDNode *Early = TestModule->getNamedMetadata("sqtt.markers.early");
  ASSERT_NE(Early, nullptr);
  auto *Size =
      mdconst::dyn_extract<ConstantInt>(Early->getOperand(0)->getOperand(3));
  ASSERT_NE(Size, nullptr);
  EXPECT_EQ(Size->getZExtValue(), 1u);

  runPass(*TestModule, Config);

  EXPECT_TRUE(traceMarkerValues(*Function).empty());
  expectNotContains(getFuncMap(*TestModule), "small_function");

  LLVMContext MustTailCtx;
  std::unique_ptr<Module> MustTailModule = makeModule(MustTailCtx);
  llvm::Function *MustTail =
      makeMustTailFunction(*MustTailModule, "early_musttail");
  runPass(*MustTailModule, Config, SQTTInstrumentPass::Mode::Early);
  EXPECT_TRUE(traceMarkerValues(*MustTail).empty());
  EXPECT_FALSE(MustTailModule->getNamedMetadata("sqtt.markers.early"));
}

TEST_F(MarkerPass, FunctionThresholdPrunesMarkersAndPreservesExistingLlvmUsed) {
  Type *I32 = Type::getInt32Ty(Ctx);
  constexpr uint32_t SmallId = 7, LargeId = 8;

  llvm::Function *Small =
      makeVoidFunction(*TestModule, "small_function", "gfx1100");
  addPassOwnedFunctionMarkers(*Small, SmallId);
  IRBuilder<> SmallBuilder(&*Small->getEntryBlock().begin());
  SmallBuilder.CreateCall(
      Intrinsic::getOrInsertDeclaration(TestModule.get(),
                                        Intrinsic::amdgcn_sched_barrier),
      {ConstantInt::get(I32, 0)});
  addEarlyFunctionMetadata(*Small, SmallId, 1, "small.hip:3");

  makeLargePassOwnedFunction(*TestModule, "large_function", LargeId,
                             "large.hip:17");

  constexpr uint32_t CloneId = 101;
  llvm::Function *LargeClone = makeLargePassOwnedFunction(
      *TestModule, "large_clone", CloneId, "clone.hip:10");
  llvm::Function *SmallClone =
      makeVoidFunction(*TestModule, "small_clone", "gfx1100");
  addPassOwnedFunctionMarkers(*SmallClone, CloneId);
  SmallClone->setMetadata(
      "sqtt.func.id",
      MDNode::get(Ctx,
                  {ConstantAsMetadata::get(ConstantInt::get(I32, CloneId))}));

  // This unregistered numeric marker happens to use the pruned function's
  // old ID. It must not be removed or rewritten with pass-owned headers.
  llvm::Function *Numeric =
      makeVoidFunction(*TestModule, "numeric_marker", "gfx90a");
  insertTraceCallBefore(Numeric->getEntryBlock().getTerminator(),
                        encodeMarker(SmallId, true, false));

  addEarlyFunctionMapEntry(*TestModule, 99, "inlined_large_function", 40,
                           "inlined.hip:21");
  addEarlyFunctionMapEntry(*TestModule, 100, "inlined_small_function", 1,
                           "inlined.hip:4");
  addExistingLlvmUsed(*TestModule);

  SQTTConfig Config = fullScopeConfig();
  Config.FunctionThreshold = 20;

  std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);
  expectContains(FuncMap, "F:1:large_function@large.hip:17");
  expectContains(FuncMap, "F:2:inlined_large_function@inlined.hip:21");
  expectNotContains(FuncMap, "small_function");
  expectNotContains(FuncMap, "inlined_small_function");
  expectNotContains(FuncMap, "large_clone");
  const GlobalVariable *Used = TestModule->getGlobalVariable("llvm.used");
  ASSERT_NE(Used, nullptr);
  const auto *UsedValues = dyn_cast<ConstantArray>(Used->getInitializer());
  ASSERT_NE(UsedValues, nullptr);
  EXPECT_EQ(UsedValues->getNumOperands(), 2u);

  EXPECT_EQ(countIntrinsicCalls(*Small, Intrinsic::amdgcn_s_ttracedata), 0u);
  EXPECT_EQ(countIntrinsicCalls(*Small, Intrinsic::amdgcn_s_ttracedata_imm),
            0u);
  EXPECT_EQ(countIntrinsicCalls(*Small, Intrinsic::amdgcn_sched_barrier), 1u);
  EXPECT_EQ(countIntrinsicCalls(*LargeClone, Intrinsic::amdgcn_s_ttracedata),
            0u);
  EXPECT_EQ(countIntrinsicCalls(*SmallClone, Intrinsic::amdgcn_s_ttracedata),
            0u);
  EXPECT_EQ(findM0NopTrace(*LargeClone), nullptr);
  EXPECT_EQ(findM0NopTrace(*SmallClone), nullptr);
  const CallInst *NumericTrace = findM0NopTrace(*Numeric);
  ASSERT_NE(NumericTrace, nullptr);
  auto *NumericValue = dyn_cast<ConstantInt>(NumericTrace->getArgOperand(0));
  ASSERT_NE(NumericValue, nullptr);
  EXPECT_EQ(NumericValue->getZExtValue(), encodeMarker(SmallId, true, false));
}

TEST_F(MarkerPass, LateNamedMarkersReuseTheirCompactedEarlyID) {
  Type *I32 = Type::getInt32Ty(Ctx);
  llvm::Function *Point =
      makeNamedMarkerSentinel(*TestModule, "sqtt_marker_point");
  GlobalVariable *Name = makeMarkerString(*TestModule, "reused_point");
  llvm::Function *Function =
      makeFunction(*TestModule, "late_named_marker", "gfx1100",
                   FunctionType::get(Type::getVoidTy(Ctx), {I32}, false));
  IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", Function));
  Builder.CreateCall(Point, {Name});
  Value *Value = Function->getArg(0);
  for (unsigned I = 0; I < 8; ++I)
    Value = Builder.CreateAdd(Value, ConstantInt::get(I32, I));
  Builder.CreateRetVoid();

  SQTTConfig Config = fullScopeConfig();
  Config.FunctionThreshold = 1;
  runPass(*TestModule, Config, SQTTInstrumentPass::Mode::Early);

  // This models a literal marker exposed only after the early pass. The
  // original declaration has been erased along with its resolved call.
  Point = makeNamedMarkerSentinel(*TestModule, "sqtt_marker_point");
  Builder.SetInsertPoint(Function->getEntryBlock().getTerminator());
  Builder.CreateCall(Point, {Name});

  const std::string FuncMap = runPassAndGetFuncMap(*TestModule, Config);
  const std::optional<unsigned> Id = pointEntryId(FuncMap, "reused_point");
  ASSERT_TRUE(Id.has_value());
  EXPECT_EQ(countPointEntries(FuncMap, "reused_point"), 1u);
  const std::vector<uint32_t> Markers = traceMarkerValues(*Function);
  EXPECT_EQ(llvm::count(Markers, encodeMarker(*Id, false, false)), 2);
}

} // namespace
