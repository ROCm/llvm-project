//===- ocml-runtime.cpp - Hotswap OCML device-library linking -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OCML-backed handlers initially emit declarations such as
// `__ocml_tanh_f32`. This file resolves those declarations from COMGR's
// embedded device-library bitcode, inlines the imported helper bodies, and
// removes unused library code before later raise pipeline stages inspect the
// IR. Final lowered IR must not depend on a device-call ABI for these helpers.
//
//===----------------------------------------------------------------------===//

#include "ocml-runtime.h"

#include "comgr-device-libs.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <optional>
#include <string>
#include <tuple>

namespace COMGR::hotswap {

namespace {

struct DeviceLibRef {
  llvm::StringRef Name;
  llvm::StringRef Contents;
};

// `LinkOnlyNeeded` reports the globals imported from each device-library module
// through an internalization callback. Keep that set so later inlining is
// limited to linked device-library bodies, not arbitrary local functions in the
// raised kernel module.
struct DeviceLibLinkState {
  llvm::StringSet<> LinkedSymbols;
};

llvm::FunctionType *ocmlTanhF32FnTy(llvm::LLVMContext &C) {
  llvm::Type *F32 = llvm::Type::getFloatTy(C);
  return llvm::FunctionType::get(F32, {F32}, /*isVarArg=*/false);
}

llvm::FunctionType *ocmlTanhF16FnTy(llvm::LLVMContext &C) {
  llvm::Type *F16 = llvm::Type::getHalfTy(C);
  return llvm::FunctionType::get(F16, {F16}, /*isVarArg=*/false);
}

std::optional<DeviceLibRef> findDeviceLibrary(llvm::StringRef Name) {
  for (const auto &Lib : COMGR::getDeviceLibraries()) {
    if (std::get<0>(Lib) == Name)
      return DeviceLibRef{std::get<0>(Lib), std::get<1>(Lib)};
  }
  return std::nullopt;
}

llvm::Error linkDeviceLibrary(llvm::Module &M, DeviceLibRef Lib,
                              DeviceLibLinkState &State) {
  llvm::MemoryBufferRef Buf(Lib.Contents, Lib.Name);
  llvm::Expected<std::unique_ptr<llvm::Module>> ModOrErr =
      llvm::parseBitcodeFile(Buf, M.getContext());
  if (!ModOrErr) {
    return ModOrErr.takeError();
  }

  std::unique_ptr<llvm::Module> LibModule = std::move(*ModOrErr);
  LibModule->setTargetTriple(M.getTargetTriple());
  LibModule->setDataLayout(M.getDataLayout());

  if (llvm::Linker::linkModules(
          M, std::move(LibModule), llvm::Linker::LinkOnlyNeeded,
          [&State](llvm::Module &LinkedM,
                   const llvm::StringSet<> &LinkedSymbols) {
            for (const auto &Entry : LinkedSymbols)
              State.LinkedSymbols.insert(Entry.getKey());
            llvm::internalizeModule(LinkedM, [&LinkedSymbols](
                                                 const llvm::GlobalValue &GV) {
              return !GV.hasName() || LinkedSymbols.count(GV.getName()) == 0;
            });
          })) {
    return llvm::createStringError("failed to link embedded device library '" +
                                   Lib.Name + "' into module '" + M.getName() +
                                   "'");
  }

  return llvm::Error::success();
}

llvm::Error linkOCMLAndSupportLibraries(llvm::Module &M,
                                        llvm::StringRef TargetProcessor,
                                        unsigned TargetWaveSize,
                                        DeviceLibLinkState &State) {
  const bool NeedsTanhF32 = M.getFunction(kOCMLTanhF32Symbol) != nullptr;
  const bool NeedsTanhF16 = M.getFunction(kOCMLTanhF16Symbol) != nullptr;

  llvm::SmallVector<std::string, 8> DeviceLibNames;
  if (llvm::Error LibNameErr = COMGR::getOCMLDeviceLibraryNames(
          TargetProcessor, TargetWaveSize, DeviceLibNames))
    return LibNameErr;

  for (llvm::StringRef Name : DeviceLibNames) {
    std::optional<DeviceLibRef> Lib = findDeviceLibrary(Name);
    if (!Lib)
      return llvm::createStringError("required OCML device library '" + Name +
                                     "' is not embedded in this COMGR build");

    if (llvm::Error LinkErr = linkDeviceLibrary(M, *Lib, State))
      return LinkErr;
  }

  auto CheckResolved = [&](llvm::StringRef Symbol) {
    llvm::Function *TanhFn = M.getFunction(Symbol);
    return TanhFn && !TanhFn->isDeclaration();
  };

  if ((NeedsTanhF32 && !CheckResolved(kOCMLTanhF32Symbol)) ||
      (NeedsTanhF16 && !CheckResolved(kOCMLTanhF16Symbol))) {
    llvm::StringRef Missing = NeedsTanhF32 && !CheckResolved(kOCMLTanhF32Symbol)
                                  ? kOCMLTanhF32Symbol
                                  : kOCMLTanhF16Symbol;
    return llvm::createStringError(
        "embedded OCML bitcode does not define " + Missing +
        "; cannot lower requested OCML helper without a resolved "
        "device-library body");
  }

  return llvm::Error::success();
}

bool isKernel(const llvm::Function &F) {
  return F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

bool isInlineableDeviceLibCallee(llvm::Function &F,
                                 const DeviceLibLinkState &State) {
  if (F.isDeclaration() || F.isIntrinsic() || isKernel(F))
    return false;
  return F.hasName() && F.hasLocalLinkage() &&
         State.LinkedSymbols.count(F.getName()) != 0;
}

llvm::Error inlineDeviceLibraryCallSites(llvm::Module &M,
                                         const DeviceLibLinkState &State) {
  // Inline imported helper calls explicitly instead of relying on a later
  // optimizer pipeline: HotSwap needs a hard failure if a device-library call
  // would remain in final IR, and `State` scopes the transformation to symbols
  // imported by the linker above.
  bool Changed = true;
  while (Changed) {
    Changed = false;
    llvm::SmallVector<llvm::CallBase *, 16> Calls;
    for (llvm::Function &F : M) {
      if (F.isDeclaration())
        continue;
      for (llvm::BasicBlock &BB : F) {
        for (llvm::Instruction &I : BB) {
          auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
          if (!CB)
            continue;
          llvm::Function *Callee = CB->getCalledFunction();
          if (!Callee || Callee == &F ||
              !isInlineableDeviceLibCallee(*Callee, State))
            continue;
          Calls.push_back(CB);
        }
      }
    }

    for (llvm::CallBase *CB : Calls) {
      llvm::Function *Callee = CB->getCalledFunction();
      llvm::InlineFunctionInfo IFI;
      llvm::InlineResult Inlined = llvm::InlineFunction(*CB, IFI);
      if (!Inlined.isSuccess()) {
        return llvm::createStringError(
            "failed to inline OCML device-library helper '" +
            Callee->getName() + "': " + Inlined.getFailureReason());
      }
      Changed = true;
    }
  }

  return llvm::Error::success();
}

bool hasDirectCallTo(llvm::Module &M, llvm::StringRef Name) {
  llvm::Function *F = M.getFunction(Name);
  if (!F)
    return false;
  for (llvm::User *U : F->users()) {
    auto *CB = llvm::dyn_cast<llvm::CallBase>(U);
    if (CB && CB->getCalledOperand()->stripPointerCasts() == F)
      return true;
  }
  return false;
}

bool isDeviceLibrarySymbol(llvm::StringRef Name) {
  return Name.starts_with("__ocml_") || Name.starts_with("__ockl_") ||
         Name.starts_with("__oclc_");
}

llvm::Error verifyDeviceLibraryReferences(llvm::Module &M) {
  for (llvm::GlobalValue &GV : M.global_values()) {
    if (!GV.isDeclaration() || GV.use_empty() || !GV.hasName() ||
        !isDeviceLibrarySymbol(GV.getName()))
      continue;
    return llvm::createStringError("unresolved device-library symbol '" +
                                   GV.getName() +
                                   "' remains referenced after OCML linking");
  }
  return llvm::Error::success();
}

void runGlobalDCE(llvm::Module &M) {
  llvm::legacy::PassManager PM;
  PM.add(llvm::createGlobalDCEPass());
  PM.run(M);
}

} // namespace

llvm::FunctionCallee declareOCMLTanhF32(llvm::Module &M) {
  return M.getOrInsertFunction(kOCMLTanhF32Symbol,
                               ocmlTanhF32FnTy(M.getContext()));
}

llvm::FunctionCallee declareOCMLTanhF16(llvm::Module &M) {
  return M.getOrInsertFunction(kOCMLTanhF16Symbol,
                               ocmlTanhF16FnTy(M.getContext()));
}

bool moduleUsesOCMLRuntime(const llvm::Module &M) {
  return M.getFunction(kOCMLTanhF32Symbol) != nullptr ||
         M.getFunction(kOCMLTanhF16Symbol) != nullptr;
}

llvm::Error linkOCMLRuntime(llvm::Module &M, llvm::StringRef TargetProcessor,
                            unsigned TargetWaveSize) {
  DeviceLibLinkState State;
  if (llvm::Error LinkErr = linkOCMLAndSupportLibraries(M, TargetProcessor,
                                                        TargetWaveSize, State))
    return LinkErr;

  if (llvm::Error InlineError = inlineDeviceLibraryCallSites(M, State))
    return InlineError;

  runGlobalDCE(M);

  if (hasDirectCallTo(M, kOCMLTanhF32Symbol) ||
      hasDirectCallTo(M, kOCMLTanhF16Symbol)) {
    return llvm::createStringError("OCML helper call remained after inlining; "
                                   "refusing to leave a device-call ABI");
  }

  if (llvm::Error LibRefErr = verifyDeviceLibraryReferences(M))
    return LibRefErr;

  return llvm::Error::success();
}

} // namespace COMGR::hotswap
