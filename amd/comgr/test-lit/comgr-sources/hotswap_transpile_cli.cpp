//===- hotswap_transpile_cli.cpp - Hotswap transpiler test driver ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command-line front end for the hotswap transpiler, used by the lit tests
// under test-lit/hotswap/raiser. Its modes grow with the stack:
//   --dump-meta      print the metadata extracted from a code object.
//   --dump-decoded   print the decoded canonical-op instruction listing.
//   --emit-ir        raise the selected kernels and print the LLVM IR.
// Diagnostics go to stderr and results to stdout, so a refuse test can
// FileCheck stderr under `not ... 2>&1` while a raise test checks stdout.
//
//===----------------------------------------------------------------------===//

#include "comgr-metadata.h"
#include "comgr.h"
#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decode.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/opcode-map.h"
#include "hotswap/loader/code-object-utils.h"
#include "hotswap/raiser/raiser.h"

// raiser.h forward-declares llvm::LLVMContext and llvm::Module, but RaiseResult
// holds them by unique_ptr, so the destructor synthesized here needs the
// complete types.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

cl::opt<std::string>
    EmitIrOpt("emit-ir", cl::ValueOptional, cl::value_desc("kernel[,kernel...]"),
              cl::desc("Raise the selected kernels and print the LLVM IR on "
                       "stdout. Bare or absent = all kernels; =<k>[,<k>...] "
                       "selects a subset in order."));

cl::opt<std::string> DumpDecodedOpt(
    "dump-decoded", cl::ValueOptional, cl::value_desc("kernel[,kernel...]"),
    cl::desc("Print the decoded instruction listing (offset, canonical op, "
             "disassembly) instead of raising. Same kernel selection as "
             "--emit-ir."));

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

// Resolve a --emit-ir / --dump-decoded value into the ordered list of kernels
// to process: empty selects every kernel in code-object order; a comma list
// selects the named kernels in order. Reports unknown names on stderr.
bool resolveTargets(llvm::StringRef Requested,
                    llvm::ArrayRef<std::string> KernelNames,
                    llvm::StringRef CoPath,
                    llvm::SmallVectorImpl<std::string> &Targets) {
  if (Requested.empty()) {
    Targets.assign(KernelNames.begin(), KernelNames.end());
    return true;
  }
  llvm::SmallVector<llvm::StringRef> RequestedNames;
  Requested.split(RequestedNames, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef Name : RequestedNames) {
    Name = Name.trim();
    if (!llvm::is_contained(KernelNames, Name)) {
      llvm::errs() << "hotswap_transpile_cli: kernel '" << Name
                   << "' not found in " << CoPath << "\n";
      return false;
    }
    Targets.push_back(Name.str());
  }
  return true;
}

// --dump-meta: print the metadata the loader extracted for every (or the
// selected) kernel, then the .text size. Needs no MC or raiser machinery.
int runDumpMeta(const COMGR::hotswap::CodeObjectInfo &Info,
                llvm::StringRef Isa) {
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

// --dump-decoded: decode each selected kernel's .text to a canonical
// instruction listing without raising. Exercises the MC stack, opcode map, and
// decoder.
int runDumpDecoded(const COMGR::hotswap::CodeObjectInfo &Info,
                   const COMGR::hotswap::TextSection &Text, llvm::StringRef Isa,
                   llvm::ArrayRef<std::string> Targets) {
  // initMCState wants the bare AMDGPU processor (e.g. gfx942); the --isa / ELF
  // form may be a full target id like "amdgcn-amd-amdhsa--gfx942:xnack-".
  llvm::StringRef Cpu = Isa.rsplit('-').second;
  if (Cpu.empty())
    Cpu = Isa;
  Cpu = Cpu.take_until([](char C) { return C == ':'; });

  llvm::Expected<COMGR::hotswap::MCState> McOrErr =
      COMGR::hotswap::initMCState(Cpu);
  if (!McOrErr) {
    llvm::errs() << "hotswap_transpile_cli: MC init failed for ISA '" << Isa
                 << "': " << llvm::toString(McOrErr.takeError()) << "\n";
    return 2;
  }
  COMGR::hotswap::MCState Mc = std::move(*McOrErr);
  COMGR::hotswap::OpcodeMap OpcMap;
  OpcMap.build(*Mc.InstrInfo);

  bool Multi = Targets.size() > 1;
  bool AnyFailed = false;
  for (const std::string &Target : Targets) {
    llvm::Expected<COMGR::hotswap::KernelSymbolExtent> ExtentOrErr =
        Info.kernelSymbolExtent(Target);
    if (!ExtentOrErr) {
      llvm::errs() << "hotswap_transpile_cli: kernel '" << Target
                   << "' extent: " << llvm::toString(ExtentOrErr.takeError())
                   << "\n";
      AnyFailed = true;
      continue;
    }
    llvm::Expected<COMGR::hotswap::DecodeResult> DecodedOrErr =
        COMGR::hotswap::decodeKernel(Mc, OpcMap, Text.Bytes,
                                     ExtentOrErr->Offset,
                                     ExtentOrErr->Offset + ExtentOrErr->Size);
    if (!DecodedOrErr) {
      llvm::errs() << "hotswap_transpile_cli: kernel '" << Target
                   << "' decode: " << llvm::toString(DecodedOrErr.takeError())
                   << "\n";
      AnyFailed = true;
      continue;
    }
    if (Multi)
      llvm::outs() << "; === hotswap_transpile_cli kernel: " << Target
                   << " ===\n";
    for (const COMGR::hotswap::DecodedInst &Di : DecodedOrErr->Insts) {
      llvm::outs() << "0x";
      llvm::outs().write_hex(Di.Offset);
      llvm::outs() << "  " << COMGR::hotswap::canonicalOpName(Di.CanonOp) << "  "
                   << COMGR::hotswap::printInst(Mc, Di.Inst) << "\n";
    }
  }
  return AnyFailed ? 1 : 0;
}

// --emit-ir: raise each selected kernel and print its LLVM IR.
int runEmitIr(const COMGR::hotswap::CodeObjectInfo &Info,
              const COMGR::hotswap::TextSection &Text, llvm::StringRef Isa,
              llvm::ArrayRef<std::string> Targets) {
  bool Multi = Targets.size() > 1;
  bool AnyFailed = false;
  for (const std::string &Target : Targets) {
    llvm::Expected<const COMGR::hotswap::KernelMeta *> MetaOrErr =
        Info.kernel(Target);
    if (!MetaOrErr) {
      llvm::errs() << "hotswap_transpile_cli: kernel '" << Target
                   << "' metadata: " << llvm::toString(MetaOrErr.takeError())
                   << "\n";
      AnyFailed = true;
      continue;
    }

    llvm::Expected<COMGR::hotswap::KernelSymbolExtent> ExtentOrErr =
        Info.kernelSymbolExtent(Target);
    if (!ExtentOrErr) {
      llvm::errs() << "hotswap_transpile_cli: kernel '" << Target
                   << "' extent: " << llvm::toString(ExtentOrErr.takeError())
                   << "\n";
      AnyFailed = true;
      continue;
    }

    llvm::Expected<COMGR::hotswap::RaiseResult> RaisedOrErr =
        COMGR::hotswap::raiseToIR(Text.Bytes, Isa, Target, **MetaOrErr,
                                  ExtentOrErr->Offset, ExtentOrErr->Size,
                                  /*CompilationTargetIsa=*/"",
                                  /*EnableWritelaneRewrite=*/true,
                                  /*EnableWaveNative=*/true,
                                  /*AssumeHipGlobalOffsetZero=*/false,
                                  /*ForceModrepDoubled=*/false, Text.Address,
                                  Text.ImageSections);
    if (!RaisedOrErr) {
      // The raiser only returns a module on success, so a failure has no
      // partial IR to dump; report the structured reason on stderr.
      llvm::errs() << "hotswap_transpile_cli: kernel '" << Target
                   << "' failed to raise: "
                   << llvm::toString(RaisedOrErr.takeError()) << "\n";
      AnyFailed = true;
      continue;
    }

    if (Multi)
      llvm::outs() << "; === hotswap_transpile_cli kernel: " << Target
                   << " ===\n";
    RaisedOrErr->Module->print(llvm::outs(), nullptr);
  }
  return AnyFailed ? 1 : 0;
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

  bool DumpDecoded = DumpDecodedOpt.getNumOccurrences() > 0;
  bool EmitIr = EmitIrOpt.getNumOccurrences() > 0;
  if (!DumpMetaOpt && !DumpDecoded && !EmitIr) {
    llvm::errs()
        << "hotswap_transpile_cli: no mode selected; pass --dump-meta, "
           "--dump-decoded, or --emit-ir\n";
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
      llvm::errs() << "hotswap_transpile_cli: cannot read ISA from " << CoPathOpt
                   << ": " << llvm::toString(ElfIsa.takeError()) << "\n";
      return 2;
    }
    Isa = std::move(*ElfIsa);
  }

  if (DumpMetaOpt)
    return runDumpMeta(Info, Isa);

  // The decode and raise modes work over the kernel .text and register AMDGPU
  // into this binary's own LLVM (see standalone-init.cpp for why not the
  // amd_comgr copy).
  COMGR::ensureLLVMInitialized();

  llvm::SmallVector<std::string> Targets;
  if (!resolveTargets(DumpDecoded ? llvm::StringRef(DumpDecodedOpt)
                                  : llvm::StringRef(EmitIrOpt),
                      Info.kernelNames(), CoPathOpt, Targets))
    return 2;

  llvm::Expected<COMGR::hotswap::TextSection> TextOrErr = Info.textSection();
  if (!TextOrErr) {
    llvm::errs() << "hotswap_transpile_cli: could not extract .text from "
                 << CoPathOpt << ": " << llvm::toString(TextOrErr.takeError())
                 << "\n";
    return 2;
  }

  if (DumpDecoded)
    return runDumpDecoded(Info, *TextOrErr, Isa, Targets);
  return runEmitIr(Info, *TextOrErr, Isa, Targets);
}
