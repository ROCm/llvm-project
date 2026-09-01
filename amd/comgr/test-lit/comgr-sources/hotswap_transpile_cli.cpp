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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace llvm;
using namespace COMGR::hotswap;

namespace {

namespace cl = llvm::cl;

cl::opt<std::string> CoPathOpt(cl::Positional, cl::Required,
                               cl::desc("<code-object.co|.hsaco>"));

cl::opt<std::string> IsaOpt("isa", cl::value_desc("arch"),
                            cl::desc("Source ISA; defaults to the ELF e_flags "
                                     "when not given."));

cl::opt<std::string>
    TargetIsaOpt("target-isa", cl::value_desc("arch"),
                 cl::desc("ISA to raise onto; defaults to the source ISA, "
                          "which raises the code object back onto the GPU it "
                          "was compiled for."));

// The three modes are mutually exclusive and each names the kernels it runs
// over the same way: bare selects every kernel in code-object order,
// =<k>[,<k>...] selects those kernels in the order given.
cl::opt<std::string> DumpMetaOpt(
    "dump-meta", cl::ValueOptional, cl::value_desc("kernel[,kernel...]"),
    cl::desc(
        "Print the metadata extracted from the code object (per-kernel ABI "
        "surface, kernel-descriptor fields, and .text extent) and exit."));

cl::opt<std::string>
    EmitIrOpt("emit-ir", cl::ValueOptional,
              cl::value_desc("kernel[,kernel...]"),
              cl::desc("Raise the selected kernels and print the LLVM IR on "
                       "stdout."));

cl::opt<std::string> DumpDecodedOpt(
    "dump-decoded", cl::ValueOptional, cl::value_desc("kernel[,kernel...]"),
    cl::desc("Print the decoded instruction listing (offset, canonical op, "
             "disassembly) instead of raising."));

// Print the ABI and descriptor fields for one kernel, in a form the lit tests
// FileCheck.
int dumpKernel(const CodeObjectInfo &Info, StringRef Name) {
  Expected<const KernelMeta *> MetaOrErr = Info.kernel(Name);
  if (!MetaOrErr) {
    errs() << "hotswap_transpile_cli: kernel '" << Name
           << "': " << toString(MetaOrErr.takeError()) << "\n";
    return 1;
  }
  const KernelMeta &Meta = **MetaOrErr;

  Expected<KernelSymbolExtent> ExtOrErr = Info.kernelSymbolExtent(Name);
  if (!ExtOrErr) {
    errs() << "hotswap_transpile_cli: kernel '" << Name
           << "' extent: " << toString(ExtOrErr.takeError()) << "\n";
    return 1;
  }

  // has_kd is always 1: create() refuses a code object whose descriptor it
  // cannot read and validate, so the register fields below are always present.
  outs() << "kernel: " << Meta.Name << " kernarg=" << Meta.KernargSegmentSize
         << " group=" << Meta.GroupSegmentFixedSize
         << " maxflat=" << Meta.MaxFlatWorkgroupSize << " has_kd=1"
         << " rsrc1=" << format_hex(Meta.ComputePgmRsrc1, 10)
         << " rsrc2=" << format_hex(Meta.ComputePgmRsrc2, 10)
         << " code_props=" << format_hex(Meta.KernelCodeProperties, 6)
         << " preload=" << format_hex(Meta.KernargPreload, 6)
         << " extent_size=" << ExtOrErr->Size << "\n";
  for (const KernelArgMeta &Arg : Meta.Args)
    outs() << "arg: name=" << Arg.Name << " offset=" << Arg.Offset
           << " size=" << Arg.Size << " kind=" << Arg.ValueKind
           << " address_space="
           << (Arg.AddressSpace.empty() ? "<none>" : Arg.AddressSpace) << "\n";
  return 0;
}

// Resolve a mode's value into the ordered list of kernels to process: empty
// selects every kernel in code-object order; a comma list selects the named
// kernels in order. Reports unknown names on stderr.
bool resolveTargets(StringRef Requested, ArrayRef<std::string> KernelNames,
                    StringRef CoPath, SmallVectorImpl<std::string> &Targets) {
  if (Requested.empty()) {
    Targets.assign(KernelNames.begin(), KernelNames.end());
    return true;
  }
  SmallVector<StringRef> RequestedNames;
  Requested.split(RequestedNames, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Name : RequestedNames) {
    Name = Name.trim();
    if (!is_contained(KernelNames, Name)) {
      errs() << "hotswap_transpile_cli: kernel '" << Name << "' not found in "
             << CoPath << "\n";
      return false;
    }
    Targets.push_back(Name.str());
  }
  return true;
}

// --dump-meta: print the metadata the loader extracted for each selected
// kernel, then the .text size. Needs no MC or raiser machinery.
int runDumpMeta(const CodeObjectInfo &Info, StringRef Isa,
                ArrayRef<std::string> Targets) {
  outs() << "isa: " << Isa << "\n";
  for (StringRef Name : Targets)
    if (int Rc = dumpKernel(Info, Name))
      return Rc;

  Expected<TextSection> TsOrErr = Info.textSection();
  if (!TsOrErr) {
    errs() << "hotswap_transpile_cli: .text: " << toString(TsOrErr.takeError())
           << "\n";
    return 1;
  }
  outs() << "text_bytes: " << TsOrErr->Bytes.size() << "\n";
  return 0;
}

// --dump-decoded: decode each selected kernel's .text to a canonical
// instruction listing without raising. Exercises the MC stack, opcode map, and
// decoder.
int runDumpDecoded(const CodeObjectInfo &Info, const TextSection &Text,
                   StringRef Isa, ArrayRef<std::string> Targets) {
  // initMCState wants the bare AMDGPU processor (e.g. gfx942); the --isa / ELF
  // form may be a full target id like "amdgcn-amd-amdhsa--gfx942:xnack-".
  StringRef Cpu = Isa;
  COMGR::TargetIdentifier Ident;
  if (COMGR::parseTargetIdentifier(Isa, Ident) == AMD_COMGR_STATUS_SUCCESS)
    Cpu = Ident.Processor;

  Expected<MCState> MCOrErr = initMCState(Cpu);
  if (!MCOrErr) {
    errs() << "hotswap_transpile_cli: MC init failed for ISA '" << Isa
           << "': " << toString(MCOrErr.takeError()) << "\n";
    return 2;
  }
  MCState MC = std::move(*MCOrErr);
  OpcodeMap OpcMap;
  OpcMap.build(*MC.InstrInfo);

  bool Multi = Targets.size() > 1;
  bool AnyFailed = false;
  for (const std::string &Target : Targets) {
    Expected<KernelSymbolExtent> ExtentOrErr = Info.kernelSymbolExtent(Target);
    if (!ExtentOrErr) {
      errs() << "hotswap_transpile_cli: kernel '" << Target
             << "' extent: " << toString(ExtentOrErr.takeError()) << "\n";
      AnyFailed = true;
      continue;
    }
    Expected<DecodeResult> DecodedOrErr =
        decodeKernel(MC, OpcMap, Text.Bytes, ExtentOrErr->Offset,
                     ExtentOrErr->Offset + ExtentOrErr->Size);
    if (!DecodedOrErr) {
      errs() << "hotswap_transpile_cli: kernel '" << Target
             << "' decode: " << toString(DecodedOrErr.takeError()) << "\n";
      AnyFailed = true;
      continue;
    }
    if (Multi)
      outs() << "; === hotswap_transpile_cli kernel: " << Target << " ===\n";
    for (const DecodedInst &Di : DecodedOrErr->Insts) {
      outs() << "0x";
      outs().write_hex(Di.Offset);
      outs() << "  " << canonicalOpName(Di.CanonOp) << "  "
             << printInst(MC, Di.Inst) << "\n";
    }
  }
  return AnyFailed ? 1 : 0;
}

// --emit-ir: raise the selected kernels from SourceIsa onto TargetIsa, into one
// module, and print its IR.
int runEmitIr(const CodeObjectInfo &Info, const TextSection &Text,
              StringRef SourceIsa, StringRef TargetIsa,
              ArrayRef<std::string> Targets) {
  SmallVector<KernelRequest> Kernels;
  for (const std::string &Target : Targets) {
    Expected<const KernelMeta *> MetaOrErr = Info.kernel(Target);
    if (!MetaOrErr) {
      errs() << "hotswap_transpile_cli: kernel '" << Target
             << "' metadata: " << toString(MetaOrErr.takeError()) << "\n";
      return 1;
    }

    Expected<KernelSymbolExtent> ExtentOrErr = Info.kernelSymbolExtent(Target);
    if (!ExtentOrErr) {
      errs() << "hotswap_transpile_cli: kernel '" << Target
             << "' extent: " << toString(ExtentOrErr.takeError()) << "\n";
      return 1;
    }

    Kernels.push_back(KernelRequest{Target, **MetaOrErr, ExtentOrErr->Offset,
                                    ExtentOrErr->Offset + ExtentOrErr->Size});
  }

  Expected<RaiseResult> RaisedOrErr =
      raiseToIR(Text, SourceIsa, TargetIsa, Kernels);
  if (!RaisedOrErr) {
    // The raiser only returns a module on success, so a failure has no partial
    // IR to dump; report the structured reason on stderr.
    errs() << "hotswap_transpile_cli: failed to raise: "
           << toString(RaisedOrErr.takeError()) << "\n";
    return 1;
  }
  RaisedOrErr->Module->print(outs(), nullptr);
  return 0;
}

} // namespace

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "Hotswap transpiler test driver.\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> CoBufOrErr =
      MemoryBuffer::getFile(CoPathOpt, /*IsText=*/false);
  if (!CoBufOrErr) {
    errs() << "hotswap_transpile_cli: cannot read " << CoPathOpt << ": "
           << CoBufOrErr.getError().message() << "\n";
    return 2;
  }
  MemoryBufferRef CoData = (*CoBufOrErr)->getMemBufferRef();

  bool DumpMeta = DumpMetaOpt.getNumOccurrences() > 0;
  bool DumpDecoded = DumpDecodedOpt.getNumOccurrences() > 0;
  bool EmitIr = EmitIrOpt.getNumOccurrences() > 0;
  if (!DumpMeta && !DumpDecoded && !EmitIr) {
    errs() << "hotswap_transpile_cli: no mode selected; pass --dump-meta, "
              "--dump-decoded, or --emit-ir\n";
    return 2;
  }

  // Validate and load the code object before interpreting anything else, so a
  // structural or metadata refusal is reported rather than a downstream error.
  Expected<CodeObjectInfo> InfoOrErr = CodeObjectInfo::create(CoData);
  if (!InfoOrErr) {
    errs() << "hotswap_transpile_cli: " << CoPathOpt << ": "
           << toString(InfoOrErr.takeError()) << "\n";
    return 1;
  }
  CodeObjectInfo &Info = *InfoOrErr;

  // ISA: explicit --isa overrides, otherwise the ELF e_flags are authoritative.
  std::string Isa = IsaOpt;
  if (Isa.empty()) {
    Expected<std::string> ElfIsa = COMGR::metadata::getElfIsaName(CoData);
    if (!ElfIsa) {
      errs() << "hotswap_transpile_cli: cannot read ISA from " << CoPathOpt
             << ": " << toString(ElfIsa.takeError()) << "\n";
      return 2;
    }
    Isa = std::move(*ElfIsa);
  }

  // Every mode names its kernels the same way, so the selection is resolved
  // once from whichever mode's value carries it.
  StringRef Requested = DumpMeta      ? StringRef(DumpMetaOpt)
                        : DumpDecoded ? StringRef(DumpDecodedOpt)
                                      : StringRef(EmitIrOpt);
  SmallVector<std::string> Targets;
  if (!resolveTargets(Requested, Info.kernelNames(), CoPathOpt, Targets))
    return 2;

  if (DumpMeta)
    return runDumpMeta(Info, Isa, Targets);

  // The decode and raise modes work over the kernel .text, so AMDGPU has to be
  // registered before the MC stack is built.
  COMGR::ensureLLVMInitialized();

  Expected<TextSection> TextOrErr = Info.textSection();
  if (!TextOrErr) {
    errs() << "hotswap_transpile_cli: could not extract .text from "
           << CoPathOpt << ": " << toString(TextOrErr.takeError()) << "\n";
    return 2;
  }

  if (DumpDecoded)
    return runDumpDecoded(Info, *TextOrErr, Isa, Targets);

  // Only the raise reads a target ISA; the decode is written in source terms.
  return runEmitIr(
      Info, *TextOrErr, Isa,
      TargetIsaOpt.empty() ? StringRef(Isa) : StringRef(TargetIsaOpt), Targets);
}
