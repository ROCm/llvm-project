//===-- AMDGPUTargetMachine.h - AMDGPU TargetMachine Interface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The AMDGPU TargetMachine interface definition for hw codegen targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H

#include "GCNSubtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include <optional>
#include <utility>

namespace llvm {

//===----------------------------------------------------------------------===//
// AMDGPU Target Machine (R600+)
//===----------------------------------------------------------------------===//

class AMDGPUTargetMachine : public CodeGenTargetMachineImpl {
protected:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

  StringRef getGPUName(const Function &F) const;
  StringRef getFeatureString(const Function &F) const;

public:
  static bool EnableFunctionCalls;
  static bool EnableLowerModuleLDS;

  AMDGPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                      StringRef FS, const TargetOptions &Options,
                      std::optional<Reloc::Model> RM,
                      std::optional<CodeModel::Model> CM, CodeGenOptLevel OL);
  ~AMDGPUTargetMachine() override;

  const TargetSubtargetInfo *getSubtargetImpl() const;
  const TargetSubtargetInfo *
  getSubtargetImpl(const Function &) const override = 0;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  void registerPassBuilderCallbacks(PassBuilder &PB) override;
  void registerDefaultAliasAnalyses(AAManager &) override;

  /// Get the integer value of a null pointer in the given address space.
  static int64_t getNullPointerValue(unsigned AddrSpace);

  bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const override;

  std::optional<dwarf::AddressSpace>
  mapToDWARFAddrSpace(unsigned LLVMAddrSpace) const override;

  unsigned getAssumedAddrSpace(const Value *V) const override;

  std::pair<const Value *, unsigned>
  getPredicatedAddrSpace(const Value *V) const override;

  unsigned getAddressSpaceForPseudoSourceKind(unsigned Kind) const override;

  bool splitModule(Module &M, unsigned NumParts,
                   function_ref<void(std::unique_ptr<Module> MPart)>
                       ModuleCallback) override;
  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override;
};

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

class GCNTargetMachine final : public AMDGPUTargetMachine {
private:
  mutable StringMap<std::unique_ptr<GCNSubtarget>> SubtargetMap;

public:
  GCNTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                   std::optional<Reloc::Model> RM,
                   std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                   bool JIT);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  const TargetSubtargetInfo *getSubtargetImpl(const Function &) const override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  bool useIPRA() const override { return true; }

  Error buildCodeGenPipeline(ModulePassManager &MPM, raw_pwrite_stream &Out,
                             raw_pwrite_stream *DwoOut,
                             CodeGenFileType FileType,
                             const CGPassBuilderOption &Opts,
                             PassInstrumentationCallbacks *PIC) override;

  void registerMachineRegisterInfoCallback(MachineFunction &MF) const override;

  MachineFunctionInfo *
  createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                            const TargetSubtargetInfo *STI) const override;

  yaml::MachineFunctionInfo *createDefaultFuncInfoYAML() const override;
  yaml::MachineFunctionInfo *
  convertFuncInfoToYAML(const MachineFunction &MF) const override;
  bool parseMachineFunctionInfo(const yaml::MachineFunctionInfo &,
                                PerFunctionMIParsingState &PFS,
                                SMDiagnostic &Error,
                                SMRange &SourceRange) const override;
  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override;
  ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const override;
};

//===----------------------------------------------------------------------===//
// AMDGPU Pass Setup - For Legacy Pass Manager.
//===----------------------------------------------------------------------===//

class AMDGPUPassConfig : public TargetPassConfig {
public:
  AMDGPUPassConfig(TargetMachine &TM, PassManagerBase &PM);

  AMDGPUTargetMachine &getAMDGPUTargetMachine() const {
    return getTM<AMDGPUTargetMachine>();
  }
  void addEarlyCSEOrGVNPass();
  void addStraightLineScalarOptimizationPasses();
  void addIRPasses() override;
  void addCodeGenPrepare() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  bool addGCPasses() override;

  std::unique_ptr<CSEConfigBase> getCSEConfig() const override;

  /// Check if a pass is enabled given \p Opt option. The option always
  /// overrides defaults if explicitly used. Otherwise its default will
  /// be used given that a pass shall work at an optimization \p Level
  /// minimum.
  bool isPassEnabled(const cl::opt<bool> &Opt,
                     CodeGenOptLevel Level = CodeGenOptLevel::Default) const {
    if (Opt.getNumOccurrences())
      return Opt;
    if (TM->getOptLevel() < Level)
      return false;
    return Opt;
  }
};

//===----------------------------------------------------------------------===//
// AMDGPU CodeGen Pass Builder interface.
//===----------------------------------------------------------------------===//

class AMDGPUCodeGenPassBuilder
    : public CodeGenPassBuilder<AMDGPUCodeGenPassBuilder, GCNTargetMachine> {
  using Base = CodeGenPassBuilder<AMDGPUCodeGenPassBuilder, GCNTargetMachine>;

public:
  AMDGPUCodeGenPassBuilder(GCNTargetMachine &TM,
                           const CGPassBuilderOption &Opts,
                           PassInstrumentationCallbacks *PIC);

  void addIRPasses(AddIRPass &) const;
  void addCodeGenPrepare(AddIRPass &) const;
  void addPreISel(AddIRPass &addPass) const;
  void addILPOpts(AddMachinePass &) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
  Error addInstSelector(AddMachinePass &) const;
  void addPreRewrite(AddMachinePass &) const;
  void addMachineSSAOptimization(AddMachinePass &) const;
  void addPostRegAlloc(AddMachinePass &) const;
  void addPreEmitPass(AddMachinePass &) const;
  void addPreEmitRegAlloc(AddMachinePass &) const;
  Error addRegAssignmentOptimized(AddMachinePass &) const;
  void addPreRegAlloc(AddMachinePass &) const;
  void addOptimizedRegAlloc(AddMachinePass &) const;
  void addPreSched2(AddMachinePass &) const;

  /// Check if a pass is enabled given \p Opt option. The option always
  /// overrides defaults if explicitly used. Otherwise its default will be used
  /// given that a pass shall work at an optimization \p Level minimum.
  bool isPassEnabled(const cl::opt<bool> &Opt,
                     CodeGenOptLevel Level = CodeGenOptLevel::Default) const;
  void addEarlyCSEOrGVNPass(AddIRPass &) const;
  void addStraightLineScalarOptimizationPasses(AddIRPass &) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETMACHINE_H
