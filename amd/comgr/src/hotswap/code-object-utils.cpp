//===- code-object-utils.cpp - Hotswap transpiler -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "code-object-utils.h"

#include "../comgr-metadata.h"
#include "../comgr-symbol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
namespace COMGR::hotswap {

namespace {

// Copy `<kernelName>.kd`'s 64 KD bytes from .rodata into `Out`. The KD
// symbol is *always* in the .rodata section for amdhsa code objects (the
// AMDGPU asm printer emits it there); we map the symbol's virtual
// address to its file-level byte offset within the section's contents
// and copy the canonical 64-byte structure. Any mismatch (missing
// symbol, wrong size, address not within .rodata) is reported and
// produces `false`.
//
// We deliberately key off the symbol rather than the MsgPack metadata:
// the MsgPack notes do not include kernarg_preload_length /
// preload_offset, and that information is essential for modelling the
// gfx1250 user-SGPR ABI in Phase 4 of the raiser.
bool readKernelDescriptorBytes(llvm::object::ObjectFile &Obj,
                               llvm::StringRef KernelName,
                               llvm::MutableArrayRef<uint8_t> Out) {
  constexpr size_t KdSize = sizeof(llvm::amdhsa::kernel_descriptor_t);
  assert(Out.size() == KdSize &&
         "kernel descriptor is sizeof(kernel_descriptor_t) bytes");
  std::string KdSymName = (KernelName + ".kd").str();

  std::optional<llvm::object::SectionRef> RodataSec;
  for (const auto &Sec : Obj.sections()) {
    auto NameOrErr = Sec.getName();
    if (!NameOrErr) {
      llvm::errs()
          << "transpiler: readKernelDescriptorBytes: failed to read section "
             "name while scanning for .rodata: "
          << llvm::toString(NameOrErr.takeError()) << "\n";
      continue;
    }
    if (*NameOrErr == ".rodata") {
      RodataSec = Sec;
      break;
    }
  }
  if (!RodataSec) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: no .rodata "
                    "section in code object\n";
    return false;
  }

  uint64_t RodataAddr = RodataSec->getAddress();
  uint64_t RodataSize = RodataSec->getSize();
  auto RodataContentsOrErr = RodataSec->getContents();
  if (!RodataContentsOrErr) {
    llvm::errs()
        << "transpiler: readKernelDescriptorBytes: failed to read .rodata "
           "section contents: "
        << llvm::toString(RodataContentsOrErr.takeError()) << "\n";
    return false;
  }
  auto RodataContents = *RodataContentsOrErr;

  auto SymOrErr = COMGR::lookupSymbolByName(Obj, KdSymName);
  if (!SymOrErr) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: "
                 << llvm::toString(SymOrErr.takeError()) << "\n";
    return false;
  }
  auto AddrOrErr = SymOrErr->getAddress();
  if (!AddrOrErr) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: failed to read "
                    "address of symbol '"
                 << KdSymName
                 << "': " << llvm::toString(AddrOrErr.takeError()) << "\n";
    return false;
  }
  uint64_t SymAddr = *AddrOrErr;

  if (SymAddr < RodataAddr || SymAddr + KdSize > RodataAddr + RodataSize) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                 << KdSymName << "' at 0x" << llvm::utohexstr(SymAddr)
                 << " is not contained within .rodata [0x"
                 << llvm::utohexstr(RodataAddr) << ", 0x"
                 << llvm::utohexstr(RodataAddr + RodataSize) << ")\n";
    return false;
  }

  uint64_t Off = SymAddr - RodataAddr;
  if (Off + KdSize > RodataContents.size()) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                 << KdSymName << "' offset 0x" << llvm::utohexstr(Off) << " + "
                 << KdSize << " exceeds .rodata contents size 0x"
                 << llvm::utohexstr(RodataContents.size()) << "\n";
    return false;
  }

  llvm::ArrayRef<uint8_t> Src(
      RodataContents.bytes_begin() + Off, KdSize);
  llvm::copy(Src, Out.begin());
  return true;
}

// Parse the four KD register fields we care about into `meta`. The
// 64-byte block is read straight into a `kernel_descriptor_t` so each
// field comes from its struct member instead of an offset + read32le
// call against a raw byte buffer.
void populateKernelDescriptorFields(llvm::object::ObjectFile &Obj,
                                    KernelMeta &Meta) {
  llvm::amdhsa::kernel_descriptor_t Kd{};
  llvm::MutableArrayRef<uint8_t> KdBytes(reinterpret_cast<uint8_t *>(&Kd),
                                         sizeof(Kd));
  if (!readKernelDescriptorBytes(Obj, Meta.Name, KdBytes)) {
    Meta.HasKernelDescriptor = false;
    return;
  }
  Meta.PrivateSegmentFixedSize = Kd.private_segment_fixed_size;
  Meta.ComputePgmRsrc1 = Kd.compute_pgm_rsrc1;
  Meta.ComputePgmRsrc2 = Kd.compute_pgm_rsrc2;
  Meta.KernelCodeProperties = Kd.kernel_code_properties;
  Meta.KernargPreload = Kd.kernarg_preload;
  Meta.HasKernelDescriptor = true;
}

// Look up `Key` in `Map`. Returns null when the key is absent.
// `MapDocNode::find(StringRef)` allocates the lookup key on `Map`'s
// owning document, so callers need only pass the literal string.
inline llvm::msgpack::DocNode *findInMap(llvm::msgpack::MapDocNode &Map,
                                         llvm::StringRef Key) {
  auto It = Map.find(Key);
  return (It == Map.end()) ? nullptr : &It->second;
}

// Pull a 64-bit integer value from a MsgPack node, accepting either
// signed or unsigned encoding (different toolchains emit either).
inline int64_t nodeAsInt(const llvm::msgpack::DocNode &N) {
  if (N.getKind() == llvm::msgpack::Type::Int)
    return N.getInt();
  if (N.getKind() == llvm::msgpack::Type::UInt)
    return static_cast<int64_t>(N.getUInt());
  return 0;
}

// Iterate the `amdhsa.kernels` array of a parsed AMDGPU MsgPack document
// and invoke `Fn` on each kernel map node. Stops on the first non-map
// child silently (matches the existing comgr metadata walker's tolerance).
template <class Fn>
void forEachKernelNode(llvm::msgpack::Document &Doc, Fn &&CB) {
  llvm::msgpack::DocNode &Root = Doc.getRoot();
  if (!Root.isMap())
    return;
  llvm::msgpack::DocNode *Kernels =
      findInMap(Root.getMap(), "amdhsa.kernels");
  if (!Kernels || !Kernels->isArray())
    return;
  for (auto &K : Kernels->getArray()) {
    if (!K.isMap())
      continue;
    CB(K.getMap());
  }
}

} // namespace

TextSection extractTextSection(llvm::MemoryBufferRef ElfData) {
  TextSection Result;
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(ElfData);
  if (!ObjOrErr) {
    llvm::errs() << "transpiler: Failed to parse ELF: "
                 << llvm::toString(ObjOrErr.takeError()) << "\n";
    return Result;
  }
  for (const auto &Sec : (*ObjOrErr)->sections()) {
    auto NameOrErr = Sec.getName();
    if (!NameOrErr) {
      llvm::errs() << "transpiler: extractTextSection: failed to read section "
                      "name while scanning for .text: "
                   << llvm::toString(NameOrErr.takeError()) << "\n";
      continue;
    }
    if (*NameOrErr != ".text")
      continue;
    auto ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr) {
      llvm::errs() << "transpiler: extractTextSection: failed to read .text "
                      "section contents: "
                   << llvm::toString(ContentsOrErr.takeError()) << "\n";
      continue;
    }
    Result.Bytes.assign(ContentsOrErr->begin(), ContentsOrErr->end());
    Result.Offset = Sec.getAddress();
    Result.Size = Sec.getSize();
    Result.Valid = true;
    return Result;
  }
  llvm::errs() << "transpiler: .text section not found in ELF\n";
  return Result;
}

llvm::SmallVector<std::string> listKernelNames(llvm::MemoryBufferRef ElfData) {
  llvm::SmallVector<std::string> Names;
  COMGR::DataMeta Meta;
  Meta.MetaDoc = std::make_shared<COMGR::MetaDocument>();
  Meta.DocNode = Meta.MetaDoc->Document.getRoot();
  if (COMGR::metadata::getMetadataRoot(ElfData, &Meta) !=
      AMD_COMGR_STATUS_SUCCESS) {
    llvm::errs() << "transpiler: listKernelNames: no AMDGPU metadata note\n";
    return Names;
  }

  forEachKernelNode(Meta.MetaDoc->Document,
                    [&](llvm::msgpack::MapDocNode &KMap) {
                      if (auto *N = findInMap(KMap, ".name"))
                        Names.push_back(N->toString());
                    });
  return Names;
}

KernelMeta extractKernelMeta(llvm::MemoryBufferRef ElfData,
                             llvm::StringRef KernelName) {
  KernelMeta Meta;

  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(ElfData);
  if (!ObjOrErr) {
    llvm::errs() << "transpiler: extractKernelMeta: Failed to parse ELF: "
                 << llvm::toString(ObjOrErr.takeError()) << "\n";
    return Meta;
  }

  COMGR::DataMeta MetaDoc;
  MetaDoc.MetaDoc = std::make_shared<COMGR::MetaDocument>();
  MetaDoc.DocNode = MetaDoc.MetaDoc->Document.getRoot();
  if (COMGR::metadata::getMetadataRoot(ElfData, &MetaDoc) !=
      AMD_COMGR_STATUS_SUCCESS) {
    llvm::errs() << "transpiler: extractKernelMeta: no AMDGPU metadata note\n";
    return Meta;
  }
  bool MatchedKernel = false;
  forEachKernelNode(MetaDoc.MetaDoc->Document,
                    [&](llvm::msgpack::MapDocNode &KMap) {
    if (MatchedKernel)
      return;
    auto *NameNode = findInMap(KMap, ".name");
    if (!NameNode || NameNode->toString() != KernelName)
      return;
    MatchedKernel = true;
    Meta.Name = NameNode->toString();

    if (auto *N = findInMap(KMap, ".kernarg_segment_size"))
      Meta.KernargSegmentSize = nodeAsInt(*N);
    if (auto *N = findInMap(KMap, ".group_segment_fixed_size"))
      Meta.GroupSegmentFixedSize = nodeAsInt(*N);
    if (auto *N = findInMap(KMap, ".private_segment_fixed_size"))
      Meta.PrivateSegmentFixedSize = nodeAsInt(*N);
    if (auto *N = findInMap(KMap, ".max_flat_workgroup_size"))
      Meta.MaxFlatWorkgroupSize = nodeAsInt(*N);

    if (auto *Args = findInMap(KMap, ".args");
        Args && Args->isArray()) {
      for (auto &ArgNode : Args->getArray()) {
        if (!ArgNode.isMap())
          continue;
        auto &AMap = ArgNode.getMap();
        KernelArgMeta Am;
        if (auto *N = findInMap(AMap, ".name"))
          Am.Name = N->toString();
        if (auto *N = findInMap(AMap, ".offset"))
          Am.Offset = nodeAsInt(*N);
        if (auto *N = findInMap(AMap, ".size"))
          Am.Size = nodeAsInt(*N);
        if (auto *N = findInMap(AMap, ".value_kind"))
          Am.ValueKind = N->toString();
        if (auto *N = findInMap(AMap, ".address_space"))
          Am.AddressSpace = nodeAsInt(*N);
        Meta.Args.push_back(Am);
      }
    }
  });

  if (!MatchedKernel) {
    llvm::errs() << "transpiler: extractKernelMeta: kernel '" << KernelName
                 << "' not found in metadata\n";
    return Meta;
  }

  // Fill the KD-register fields from .rodata. Sets Meta.HasKernelDescriptor
  // on success and emits a diagnostic on failure; the caller (raiser /
  // Phase-4 init) is responsible for refusing the lift if the field is
  // false rather than silently assuming a hardcoded SGPR layout.
  populateKernelDescriptorFields(*ObjOrErr->get(), Meta);
  return Meta;
}

llvm::Expected<uint64_t>
findKernelSymbolOffset(llvm::MemoryBufferRef ElfData,
                       llvm::StringRef KernelName) {
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(ElfData);
  if (!ObjOrErr)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "findKernelSymbolOffset: Failed to parse ELF: " +
            llvm::toString(ObjOrErr.takeError()));

  uint64_t TextBase = UINT64_MAX;
  for (const auto &Sec : (*ObjOrErr)->sections()) {
    auto NameOrErr = Sec.getName();
    if (NameOrErr && *NameOrErr == ".text") {
      TextBase = Sec.getAddress();
      break;
    }
  }
  if (TextBase == UINT64_MAX)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no .text section in ELF");

  auto SymOrErr = COMGR::lookupSymbolByName(**ObjOrErr, KernelName);
  if (!SymOrErr)
    return SymOrErr.takeError();
  auto AddrOrErr = SymOrErr->getAddress();
  if (!AddrOrErr)
    return AddrOrErr.takeError();
  if (*AddrOrErr < TextBase)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "symbol '" + KernelName + "' address < .text base");
  return *AddrOrErr - TextBase;
}

} // namespace COMGR::hotswap
