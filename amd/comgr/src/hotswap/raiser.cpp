//===- raiser.cpp - Hotswap MC -> LLVM IR raiser scaffolding --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raiser.h"

#include "hotswap-error.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace COMGR::hotswap {

namespace {

constexpr llvm::StringLiteral AMDGPUTriple = "amdgcn-amd-amdhsa";

// Reject obviously-bad inputs before constructing IR. Mirrors the
// preconditions the full pipeline enforces in subsequent commits.
//
// Two operating modes:
//
//   * Scaffolding mode (both `SourceISA` and `KernelName` empty):
//     validation is bypassed entirely. Useful for stubbing the raiser
//     without standing up a real ISA / metadata pair.
//
//   * Non-empty mode (anything else): the input is treated as a real
//     lift request. Both strings must be non-empty, `SourceISA` must
//     parse via `llvm::AMDGPU::parseArchAMDGCN`, and the kernel
//     descriptor must have been parsed successfully (signaled by
//     `Meta.HasKernelDescriptor`). A partially-empty input
//     (e.g. empty kernel name with a real ISA) is rejected as
//     malformed.
//
// Ideally we would reuse `COMGR::parseTargetIdentifier`, but that helper
// currently lives behind the comgr-metadata layer in `src/comgr.cpp` and
// is not reachable from the hotswap subproject. As a stop-gap, validate
// the AMDGPU processor name through `llvm::AMDGPU::parseArchAMDGCN`.
llvm::Error validateInputs(llvm::StringRef SourceISA,
                           llvm::StringRef KernelName, const KernelMeta &Meta) {
  if (SourceISA.empty() && KernelName.empty())
    return llvm::Error::success();

  if (SourceISA.empty())
    return makeHotswapError(
        "raiseToIR: source ISA empty for non-empty kernel name '" + KernelName +
        "'");
  if (KernelName.empty())
    return makeHotswapError(
        "raiseToIR: kernel name empty for non-empty source ISA '" + SourceISA +
        "'");

  // The disassembler-facing identifier is `<arch>-<vendor>-<os>-<env>-<gfx>`;
  // `parseArchAMDGCN` inspects the trailing component.
  llvm::StringRef GfxName = SourceISA.rsplit('-').second;
  if (GfxName.empty())
    GfxName = SourceISA;
  if (llvm::AMDGPU::parseArchAMDGCN(GfxName) == llvm::AMDGPU::GK_NONE)
    return makeHotswapError("raiseToIR: source ISA '" + SourceISA +
                            "' does not name an AMDGPU GPU");

  if (!Meta.HasKernelDescriptor)
    return makeHotswapError("raiseToIR: kernel '" + KernelName +
                            "' has no parsed kernel descriptor");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<RaiseResult> raiseToIR(llvm::StringRef SourceISA,
                                      llvm::StringRef KernelName,
                                      const KernelMeta &Meta) {
  using namespace llvm;

  if (Error E = validateInputs(SourceISA, KernelName, Meta))
    return std::move(E);

  RaiseResult Result;
  Result.Ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *Result.Ctx;
  Result.Module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple(AMDGPUTriple));

  FunctionType *FuncTy =
      FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(Entry);
  B.CreateRetVoid();

  return Result;
}

} // namespace COMGR::hotswap
