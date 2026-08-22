//===- raiser.cpp - Hotswap MC -> LLVM IR raiser scaffolding --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raiser.h"

#include "hotswap/raiser/raise_failure.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace COMGR::hotswap {

namespace {

constexpr llvm::StringLiteral AMDGPUTriple = "amdgcn-amd-amdhsa";

// Address space the kernarg segment lives in.
constexpr unsigned ConstantAddressSpace = 4;

// Minimum kernarg segment alignment the AMDGPU ABI mandates.
constexpr llvm::Align KernargSegmentAlign = llvm::Align::Constant<16>();

// Reject obviously-bad inputs before constructing IR. Mirrors the
// preconditions the full pipeline enforces in subsequent commits.
//
// Ideally we would reuse `COMGR::parseTargetIdentifier`, but that helper
// currently lives behind the comgr-metadata layer in `src/comgr.cpp` and
// is not reachable from the hotswap subproject. As a stop-gap, validate
// the AMDGPU processor name through `llvm::AMDGPU::parseArchAMDGCN`.
llvm::Error validateInputs(llvm::StringRef SourceISA,
                           llvm::StringRef KernelName) {
  if (SourceISA.empty())
    return RaiseFailure::general(RaiseFailureReason::BadInput,
                                 "source ISA string is empty");
  // The disassembler-facing identifier is `<arch>-<vendor>-<os>-<env>-<gfx>`;
  // `parseArchAMDGCN` inspects the trailing component.
  llvm::StringRef GfxName = SourceISA.rsplit('-').second;
  if (GfxName.empty()) {
    GfxName = SourceISA;
  }
  if (llvm::AMDGPU::parseArchAMDGCN(GfxName) == llvm::AMDGPU::GK_NONE)
    return RaiseFailure::general(RaiseFailureReason::BadInput,
                                 "source ISA '" + SourceISA +
                                     "' does not name an AMDGPU GPU");
  if (KernelName.empty())
    return RaiseFailure::general(RaiseFailureReason::BadInput,
                                 "kernel name is empty");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<RaiseResult> raiseToIR(llvm::StringRef SourceISA,
                                      llvm::StringRef KernelName,
                                      const KernelMeta &Meta) {
  using namespace llvm;

  if (Error E = validateInputs(SourceISA, KernelName))
    return std::move(E);

  RaiseResult Result;
  Result.Ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *Result.Ctx;
  Result.Module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple(AMDGPUTriple));

  // A single opaque parameter spanning the source kernarg segment, so the
  // emitted kernel descriptor reports the source segment size and the ABI
  // alignment. The raised body reads arguments as ordinary loads off the
  // kernarg pointer, at the byte offsets the source metadata gives them, so it
  // needs no typed view of the source signature.
  SmallVector<Type *> ParamTys;
  Type *KernargSegmentTy = nullptr;
  if (Meta.KernargSegmentSize > 0) {
    KernargSegmentTy =
        ArrayType::get(Type::getInt8Ty(C), Meta.KernargSegmentSize);
    ParamTys.push_back(PointerType::get(C, ConstantAddressSpace));
  }

  FunctionType *FuncTy =
      FunctionType::get(Type::getVoidTy(C), ParamTys, /*isVarArg=*/false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  // AMDGPULowerKernelArguments honors the `align` parameter attribute only on a
  // byref kernel argument; without `byref` the segment would take the array
  // type's natural one-byte alignment.
  if (KernargSegmentTy) {
    F->addParamAttr(0, Attribute::getWithByRefType(C, KernargSegmentTy));
    F->addParamAttr(0, Attribute::getWithAlignment(C, KernargSegmentAlign));
  }

  // The host fills the kernarg buffer from the source metadata and leaves no
  // room past the source segment, so the target ABI's hidden-argument block
  // must not be appended to it.
  F->addFnAttr("amdgpu-no-implicitarg-ptr");

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(Entry);
  B.CreateRetVoid();

  return Result;
}

} // namespace COMGR::hotswap
