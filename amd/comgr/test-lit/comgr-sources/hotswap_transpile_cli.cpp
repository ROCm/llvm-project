//===- hotswap_transpile_cli.cpp - Hotswap transpiler test driver ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command-line front end for the hotswap transpiler, used by the lit tests
// under test-lit/hotswap/raiser. Its modes grow with the stack; this milestone
// supports --dump-meta, which prints the metadata extracted from a code object
// so the extraction can be checked without any MC or raiser machinery.
//
//===----------------------------------------------------------------------===//

#include "comgr-metadata.h"
#include "hotswap/loader/code-object-utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace {

namespace cl = llvm::cl;

cl::opt<std::string> CoPathOpt(cl::Positional, cl::Required,
                               cl::desc("<code-object.co|.hsaco>"));

cl::opt<std::string> IsaOpt("isa", cl::value_desc("arch"),
                            cl::desc("Source ISA; defaults to the ELF e_flags "
                                     "when not given."));

cl::opt<std::string> KernelOpt(
    "kernel", cl::value_desc("name"),
    cl::desc("Restrict output to this kernel instead of every kernel."));

cl::opt<bool> DumpMetaOpt(
    "dump-meta",
    cl::desc(
        "Print the metadata extracted from the code object (per-kernel ABI "
        "surface, kernel-descriptor fields, and .text extent) and exit."));

// Print the ABI and descriptor fields for one kernel, in a form the lit tests
// FileCheck.
int dumpKernel(const COMGR::hotswap::CodeObjectInfo &Info,
               llvm::StringRef Name) {
  llvm::Expected<const COMGR::hotswap::KernelMeta *> MetaOrErr =
      Info.kernel(Name);
  if (!MetaOrErr) {
    llvm::errs() << "hotswap_transpile_cli: kernel '" << Name
                 << "': " << llvm::toString(MetaOrErr.takeError()) << "\n";
    return 1;
  }
  const COMGR::hotswap::KernelMeta &Meta = **MetaOrErr;

  llvm::Expected<COMGR::hotswap::KernelSymbolExtent> ExtOrErr =
      Info.kernelSymbolExtent(Name);
  if (!ExtOrErr) {
    llvm::errs() << "hotswap_transpile_cli: kernel '" << Name
                 << "' extent: " << llvm::toString(ExtOrErr.takeError())
                 << "\n";
    return 1;
  }

  // has_kd is always 1: create() refuses a code object whose descriptor it
  // cannot read and validate, so the register fields below are always present.
  llvm::outs() << "kernel: " << Meta.Name
               << " kernarg=" << Meta.KernargSegmentSize
               << " group=" << Meta.GroupSegmentFixedSize
               << " maxflat=" << Meta.MaxFlatWorkgroupSize << " has_kd=1"
               << " rsrc1=" << llvm::format_hex(Meta.ComputePgmRsrc1, 10)
               << " rsrc2=" << llvm::format_hex(Meta.ComputePgmRsrc2, 10)
               << " code_props="
               << llvm::format_hex(Meta.KernelCodeProperties, 6)
               << " preload=" << llvm::format_hex(Meta.KernargPreload, 6)
               << " extent_size=" << ExtOrErr->Size << "\n";
  for (const COMGR::hotswap::KernelArgMeta &Arg : Meta.Args)
    llvm::outs() << "arg: name=" << Arg.Name << " offset=" << Arg.Offset
                 << " size=" << Arg.Size << " kind=" << Arg.ValueKind
                 << " address_space="
                 << (Arg.AddressSpace.empty() ? "<none>" : Arg.AddressSpace)
                 << "\n";
  return 0;
}

} // namespace

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "Hotswap transpiler test driver.\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> CoBufOrErr =
      llvm::MemoryBuffer::getFile(CoPathOpt, /*IsText=*/false);
  if (!CoBufOrErr) {
    llvm::errs() << "hotswap_transpile_cli: cannot read " << CoPathOpt << ": "
                 << CoBufOrErr.getError().message() << "\n";
    return 2;
  }
  llvm::MemoryBufferRef CoData = (*CoBufOrErr)->getMemBufferRef();

  if (!DumpMetaOpt) {
    llvm::errs()
        << "hotswap_transpile_cli: no mode selected; pass --dump-meta\n";
    return 2;
  }

  // Validate and load the code object before interpreting anything else, so a
  // structural or metadata refusal is reported rather than a downstream error.
  llvm::Expected<COMGR::hotswap::CodeObjectInfo> InfoOrErr =
      COMGR::hotswap::CodeObjectInfo::create(CoData);
  if (!InfoOrErr) {
    llvm::errs() << "hotswap_transpile_cli: " << CoPathOpt << ": "
                 << llvm::toString(InfoOrErr.takeError()) << "\n";
    return 1;
  }
  COMGR::hotswap::CodeObjectInfo &Info = *InfoOrErr;

  // ISA: explicit --isa overrides, otherwise the ELF e_flags are authoritative.
  std::string Isa = IsaOpt;
  if (Isa.empty()) {
    llvm::Expected<std::string> ElfIsa = COMGR::metadata::getElfIsaName(CoData);
    if (!ElfIsa) {
      llvm::errs() << "hotswap_transpile_cli: cannot read ISA from "
                   << CoPathOpt << ": " << llvm::toString(ElfIsa.takeError())
                   << "\n";
      return 2;
    }
    Isa = std::move(*ElfIsa);
  }

  llvm::outs() << "isa: " << Isa << "\n";
  if (!KernelOpt.empty()) {
    if (int Rc = dumpKernel(Info, KernelOpt))
      return Rc;
  } else {
    for (llvm::StringRef Name : Info.kernelNames())
      if (int Rc = dumpKernel(Info, Name))
        return Rc;
  }

  llvm::Expected<COMGR::hotswap::TextSection> TsOrErr = Info.textSection();
  if (!TsOrErr) {
    llvm::errs() << "hotswap_transpile_cli: .text: "
                 << llvm::toString(TsOrErr.takeError()) << "\n";
    return 1;
  }
  llvm::outs() << "text_bytes: " << TsOrErr->Bytes.size() << "\n";
  return 0;
}
