//===- raise_cli.cpp - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Per-file raiser CLI used by the hotswap-raise/ lit fixtures.
//
// Usage:
//   raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>]
//             [--isa=<arch>] [--target-isa=<arch>]
//
// --emit-ir mode runs `raiseToIR` in-process, dumps the raised LLVM IR
// for a single kernel on stdout, and leaves stderr alone so FileCheck
// can match warnings / abort-gate diagnostics. Selects the only kernel
// when the code object has one, or requires the `=<kernel>` form when
// there are multiple. Exits 0 iff the kernel raised successfully;
// non-zero otherwise.
//
// Subsequent commits add the default fork-mode (per-kernel crash
// isolation for the kerneldex runner) and `--write-hsaco` mode (full
// raise + llc + lld pipeline) once `hotswap/pipeline.h` lands.

#include "comgr-metadata.h"
#include "hotswap/code-object-utils.h"
#include "hotswap/raiser.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string autoDetectIsa(llvm::StringRef Path) {
  for (size_t I = 0; I + 3 < Path.size(); ++I) {
    if (Path[I] == 'g' && Path[I + 1] == 'f' && Path[I + 2] == 'x') {
      size_t J = I + 3;
      while (J < Path.size() &&
             std::isdigit(static_cast<unsigned char>(Path[J])))
        ++J;
      if (J > I + 3) {
        if (J < Path.size() && Path[J] >= 'a' && Path[J] <= 'z')
          ++J;
        return Path.substr(I, J - I).str();
      }
    }
  }
  return {};
}

int usage() {
  llvm::errs()
      << "usage:\n"
         "  raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] "
         "[--isa=<arch>] [--target-isa=<arch>]\n"
         "\n"
         "--emit-ir mode dumps raised LLVM IR for a single kernel on stdout.\n"
         "  No fork; stderr left alone for FileCheck.\n"
         "ISA is inferred from the filename when --isa is not given.\n";
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  std::string CoPath;
  std::string Isa;
  std::string TargetIsa;
  bool EmitIr = false;
  std::string EmitIrKernel;
  for (int I = 1; I < argc; ++I) {
    std::string A = argv[I];
    if (A.rfind("--isa=", 0) == 0) {
      Isa = A.substr(6);
    } else if (A == "--isa") {
      if (I + 1 >= argc)
        return usage();
      Isa = argv[++I];
    } else if (A.rfind("--target-isa=", 0) == 0) {
      TargetIsa = A.substr(13);
    } else if (A == "--target-isa") {
      if (I + 1 >= argc)
        return usage();
      TargetIsa = argv[++I];
    } else if (A == "--emit-ir") {
      EmitIr = true;
    } else if (A.rfind("--emit-ir=", 0) == 0) {
      EmitIr = true;
      EmitIrKernel = A.substr(10);
    } else if (!A.empty() && A[0] == '-') {
      llvm::errs() << "raise_cli: unknown flag: " << A << "\n";
      return usage();
    } else if (CoPath.empty()) {
      CoPath = A;
    } else {
      llvm::errs() << "raise_cli: unexpected positional arg: " << A << "\n";
      return usage();
    }
  }
  if (CoPath.empty())
    return usage();
  if (!EmitIr) {
    llvm::errs() << "raise_cli: this build only supports --emit-ir mode\n";
    return usage();
  }

  std::vector<uint8_t> CoData = COMGR::hotswap::readFile(CoPath);
  if (CoData.empty()) {
    llvm::errs() << "raise_cli: cannot read " << CoPath << "\n";
    return 2;
  }
  llvm::MemoryBufferRef CoBuf(
      llvm::StringRef(reinterpret_cast<const char *>(CoData.data()),
                      CoData.size()),
      "");

  if (Isa.empty()) {
    Isa = autoDetectIsa(CoPath);
    if (Isa.empty()) {
      std::string ElfIsa;
      if (COMGR::metadata::getElfIsaName(CoBuf, ElfIsa) ==
          AMD_COMGR_STATUS_SUCCESS)
        Isa = std::move(ElfIsa);
    }
    if (Isa.empty()) {
      llvm::errs() << "raise_cli: could not infer ISA from " << CoPath
                   << "; pass --isa=<arch>\n";
      return 2;
    }
  }

  auto KernelNamesOrErr = COMGR::hotswap::listKernelNames(CoBuf);
  if (!KernelNamesOrErr) {
    llvm::errs() << "raise_cli: no kernels in " << CoPath << ": "
                 << llvm::toString(KernelNamesOrErr.takeError()) << "\n";
    return 2;
  }
  llvm::SmallVector<std::string> KernelNames = std::move(*KernelNamesOrErr);
  if (KernelNames.empty()) {
    llvm::errs() << "raise_cli: no kernels in " << CoPath << "\n";
    return 2;
  }

  auto TextOrErr = COMGR::hotswap::extractTextSection(CoBuf);
  if (!TextOrErr) {
    llvm::errs() << "raise_cli: could not extract .text from " << CoPath
                 << ": " << llvm::toString(TextOrErr.takeError()) << "\n";
    return 2;
  }
  // Note: text bytes are unused at P6's smaller raise_cli (raiseToIR takes
  // only ISA / kernel name / metadata at this point); later commits wire
  // the extracted bytes through.
  (void)*TextOrErr;

  std::string Target;
  if (EmitIrKernel.empty()) {
    if (KernelNames.size() != 1) {
      llvm::errs() << "raise_cli: --emit-ir requires =<kernel> when the "
                      "code object has "
                   << KernelNames.size() << " kernels\n";
      return 2;
    }
    Target = KernelNames.front();
  } else {
    bool Found = false;
    for (const auto &Kn : KernelNames)
      if (Kn == EmitIrKernel) {
        Target = Kn;
        Found = true;
        break;
      }
    if (!Found) {
      llvm::errs() << "raise_cli: kernel '" << EmitIrKernel
                   << "' not found in " << CoPath << "\n";
      return 2;
    }
  }

  auto MetaOrErr = COMGR::hotswap::extractKernelMeta(CoBuf, Target);
  if (!MetaOrErr) {
    llvm::errs() << "raise_cli: kernel '" << Target << "' metadata: "
                 << llvm::toString(MetaOrErr.takeError()) << "\n";
    return 1;
  }
  COMGR::hotswap::KernelMeta Meta = std::move(*MetaOrErr);
  auto Raised = COMGR::hotswap::raiseToIR(Isa, Target, Meta);
  if (!Raised.Success) {
    llvm::errs() << "raise_cli: kernel '" << Target << "' failed to raise: "
                 << (Raised.Failure.Detail.empty()
                         ? "unknown"
                         : Raised.Failure.Detail.c_str())
                 << "\n";
    return 1;
  }
  llvm::raw_fd_ostream Os(/*fd=*/1, /*shouldClose=*/false);
  Raised.Module->print(Os, /*AAW=*/nullptr);
  return 0;
}
