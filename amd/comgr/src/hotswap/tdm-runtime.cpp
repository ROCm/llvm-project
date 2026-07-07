//===- tdm-runtime.cpp - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tdm-runtime.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

// HOTSWAP_HAVE_TDM_RUNTIME is set to 1 by `CMakeLists.txt` when hipcc is
// available at configure time and the embedded bitcode blob is generated
// into `tdm_runtime_blob.cpp`. When unset (no hipcc), the linker would
// otherwise fail to resolve `tdm_runtime_bc_data` / `tdm_runtime_bc_size`
// -- we provide a one-byte stub here so the build succeeds and
// `tdmRuntimeAvailable()` reports false at runtime.
#ifndef HOTSWAP_HAVE_TDM_RUNTIME
#define HOTSWAP_HAVE_TDM_RUNTIME 0
#endif

extern "C" {
#if HOTSWAP_HAVE_TDM_RUNTIME
extern const unsigned char tdm_runtime_bc_data[];
extern const unsigned long tdm_runtime_bc_size;
#else
const unsigned char tdm_runtime_bc_data[1] = {0};
const unsigned long tdm_runtime_bc_size = 0;
#endif
}

namespace COMGR::hotswap {

namespace {

llvm::FunctionType *tdmHelperFnTy(llvm::LLVMContext &C) {
  using namespace llvm;
  auto *I32 = Type::getInt32Ty(C);
  auto *V4i = FixedVectorType::get(I32, 4);
  auto *V8i = FixedVectorType::get(I32, 8);
  // Helper takes the four D# groups the walker consumes plus the source
  // wave size. The LLVM intrinsic's trailing <8 x i32> `grp4` is reserved,
  // and `i32 cpol` is a cache-policy immediate with no target-side encoding
  // in the helper path. See `tdm-runtime.h` and the matching handler-side
  // elision in `handle-vimage.cpp`.
  return FunctionType::get(Type::getVoidTy(C), {V4i, V8i, V4i, V4i, I32},
                           /*isVarArg=*/false);
}

llvm::Error inlineTDMHelperCallSites(llvm::Module &M, llvm::StringRef Name) {
  llvm::Function *F = M.getFunction(Name);
  if (!F || F->isDeclaration())
    return llvm::Error::success();

  llvm::SmallVector<llvm::CallBase *, 8> Calls;
  for (llvm::User *U : F->users()) {
    auto *CB = llvm::dyn_cast<llvm::CallBase>(U);
    if (!CB || CB->getCalledOperand()->stripPointerCasts() != F)
      continue;
    Calls.push_back(CB);
  }

  for (llvm::CallBase *CB : Calls) {
    llvm::InlineFunctionInfo IFI;
    llvm::InlineResult Inlined = llvm::InlineFunction(*CB, IFI);
    if (!Inlined.isSuccess()) {
      return llvm::createStringError(
          "transpiler: failed to inline TDM helper '" + Name +
          "': " + Inlined.getFailureReason());
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::FunctionCallee declareTDMLoad(llvm::Module &M) {
  return M.getOrInsertFunction(kTDMLoadSymbol, tdmHelperFnTy(M.getContext()));
}

llvm::FunctionCallee declareTDMStore(llvm::Module &M) {
  return M.getOrInsertFunction(kTDMStoreSymbol, tdmHelperFnTy(M.getContext()));
}

bool tdmRuntimeAvailable() { return tdm_runtime_bc_size > 0; }

bool moduleUsesTDMRuntime(const llvm::Module &M) {
  return M.getFunction(kTDMLoadSymbol) != nullptr ||
         M.getFunction(kTDMStoreSymbol) != nullptr;
}

llvm::Error linkTDMRuntime(llvm::Module &M, llvm::StringRef /*targetISA*/) {
  if (!tdmRuntimeAvailable()) {
    return llvm::createStringError(
        "transpiler: TDM runtime requested but unavailable "
        "(transpiler was built without hipcc -- re-run CMake with "
        "hipcc on PATH to enable cross-target gfx1250->gfx942 TDM "
        "emulation).");
  }

  auto Buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(tdm_runtime_bc_data),
                      tdm_runtime_bc_size),
      "hotswap_tdm_runtime.bc",
      /*RequiresNullTerminator=*/false);

  auto ModOrErr =
      llvm::parseBitcodeFile(Buf->getMemBufferRef(), M.getContext());
  if (!ModOrErr) {
    return ModOrErr.takeError();
  }
  std::unique_ptr<llvm::Module> Rt = std::move(*ModOrErr);

  // Inherit the kernel module's triple/datalayout. The helper bitcode is
  // architecture-neutral at the IR level (no MFMA, no target-private
  // intrinsics beyond `mbcnt`/`atomicrmw add` on addrspace(3)) and we
  // want `llc -mcpu=<targetISA>` to drive lowering, not whatever default
  // triple hipcc baked into the .bc.
  Rt->setTargetTriple(M.getTargetTriple());
  Rt->setDataLayout(M.getDataLayout());

  if (llvm::Linker::linkModules(M, std::move(Rt))) {
    return llvm::createStringError(
        "transpiler: failed to link TDM runtime into module '" + M.getName() +
        "'");
  }

  // The helper bitcode is a semantic lowering aid, not a device-call ABI
  // contract.  Leave direct helper calls out of the final kernel IR: otherwise
  // LLVM lowers the calls through a private-segment/dynamic-stack path that the
  // original gfx1250 kernel did not require, and large Triton TensorDescriptor
  // kernels can fault before any useful numerical comparison.
  if (llvm::Error Err = inlineTDMHelperCallSites(M, kTDMLoadSymbol)) {
    return Err;
  }

  if (llvm::Error Err = inlineTDMHelperCallSites(M, kTDMStoreSymbol)) {
    return Err;
  }

  return llvm::Error::success();
}

} // namespace COMGR::hotswap
