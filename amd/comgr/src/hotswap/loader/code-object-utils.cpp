//===- code-object-utils.cpp - Hotswap transpiler -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "code-object-utils.h"

#include "comgr-metadata.h"
#include "comgr-symbol.h"
#include "hotswap/common/hotswap-error.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <optional>

using namespace llvm;

namespace COMGR::hotswap {

//===----------------------------------------------------------------------===//
// Section / symbol helpers
//===----------------------------------------------------------------------===//

// Return the section named `Name`, or nullopt when absent. Forwards
// getName() failures.
static Expected<std::optional<object::SectionRef>>
findSection(object::ObjectFile &Obj, StringRef Name) {
  for (const object::SectionRef &Section : Obj.sections()) {
    Expected<StringRef> SecName = Section.getName();
    if (!SecName)
      return SecName.takeError();
    if (*SecName == Name)
      return std::optional<object::SectionRef>(Section);
  }
  return std::optional<object::SectionRef>(std::nullopt);
}

//===----------------------------------------------------------------------===//
// MsgPack field readers: absent optional fields keep the default, present
// fields must be well-typed, and required fields must be present.
//===----------------------------------------------------------------------===//

static msgpack::DocNode *findInMap(msgpack::MapDocNode &Map, StringRef Key) {
  auto It = Map.find(Key);
  return It == Map.end() ? nullptr : &It->second;
}

static std::optional<int64_t> nodeAsInt(const msgpack::DocNode &Node) {
  if (Node.getKind() == msgpack::Type::Int)
    return Node.getInt();
  if (Node.getKind() == msgpack::Type::UInt)
    return static_cast<int64_t>(Node.getUInt());
  return std::nullopt;
}

// Read `Key` as a uint32. `Required` controls whether an absent key is an
// error; a present-but-not-uint32 value is always an error.
static Error readUInt32(msgpack::MapDocNode &Map, StringRef Key,
                        StringRef KernelName, bool Required, uint32_t &Out) {
  msgpack::DocNode *Node = findInMap(Map, Key);
  if (!Node) {
    if (Required)
      return makeHotswapError(
          formatv("kernel '{0}': required metadata field '{1}' is missing",
                  KernelName, Key));
    return Error::success();
  }
  std::optional<int64_t> Value = nodeAsInt(*Node);
  if (!Value || *Value < 0 || *Value > UINT32_MAX)
    return makeHotswapError(formatv(
        "kernel '{0}': metadata field '{1}' is not a uint32", KernelName, Key));
  Out = static_cast<uint32_t>(*Value);
  return Error::success();
}

// Read `Key` as a string. `Required` controls whether an absent key is an
// error; a present-but-not-string value is always an error. `toString()` is
// avoided because it accepts non-string scalars and asserts on arrays / maps.
static Error readString(msgpack::MapDocNode &Map, StringRef Key,
                        StringRef KernelName, bool Required, std::string &Out) {
  msgpack::DocNode *Node = findInMap(Map, Key);
  if (!Node) {
    if (Required)
      return makeHotswapError(
          formatv("kernel '{0}': required metadata field '{1}' is missing",
                  KernelName, Key));
    return Error::success();
  }
  if (!Node->isString())
    return makeHotswapError(formatv(
        "kernel '{0}': metadata field '{1}' is not a string", KernelName, Key));
  Out = Node->getString().str();
  return Error::success();
}

// Invoke `Callback` on each kernel map node of the required `amdhsa.kernels`
// array. Malformed structure is an error rather than silently skipped.
static Error
forEachKernelNode(msgpack::Document &Doc,
                  function_ref<Error(msgpack::MapDocNode &)> Callback) {
  msgpack::DocNode &Root = Doc.getRoot();
  if (!Root.isMap())
    return makeHotswapError("AMDGPU metadata root is not a map");
  msgpack::DocNode *Kernels = findInMap(Root.getMap(), "amdhsa.kernels");
  if (!Kernels)
    return makeHotswapError("AMDGPU metadata has no amdhsa.kernels");
  if (!Kernels->isArray())
    return makeHotswapError("amdhsa.kernels is not an array");
  for (msgpack::DocNode &Kernel : Kernels->getArray()) {
    if (!Kernel.isMap())
      return makeHotswapError("amdhsa.kernels entry is not a map");
    if (Error Err = Callback(Kernel.getMap()))
      return Err;
  }
  return Error::success();
}

// Reject an AMDGPU metadata document whose schema version this loader does not
// understand. `amdhsa.version` is [major, minor]; only major version 1 is
// supported.
static Error checkMetadataVersion(msgpack::Document &Doc) {
  msgpack::DocNode &Root = Doc.getRoot();
  if (!Root.isMap())
    return makeHotswapError("AMDGPU metadata root is not a map");
  msgpack::DocNode *Version = findInMap(Root.getMap(), "amdhsa.version");
  if (!Version)
    return makeHotswapError("AMDGPU metadata has no amdhsa.version");
  if (!Version->isArray() || Version->getArray().size() != 2)
    return makeHotswapError("amdhsa.version is not a [major, minor] array");
  std::optional<int64_t> Major = nodeAsInt(Version->getArray()[0]);
  if (!Major)
    return makeHotswapError("amdhsa.version major is not an integer");
  if (*Major != 1)
    return makeHotswapError(
        formatv("unsupported AMDGPU metadata version {0}", *Major));
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Kernel descriptor
//===----------------------------------------------------------------------===//

// Read and validate the 64-byte `<symbol>` kernel descriptor from .rodata. The
// AMDGPU asm printer emits it as an STT_OBJECT there; the descriptor is read
// rather than derived from the MsgPack notes because those omit the kernarg
// preload spec the gfx1250 user-SGPR ABI needs. Fields are read as explicit
// little-endian values since `kernel_descriptor_t` has native integer members.
static Expected<amdhsa::kernel_descriptor_t>
readKernelDescriptor(object::ObjectFile &Obj, StringRef DescriptorSymbolName) {
  constexpr uint64_t DescriptorSize = sizeof(amdhsa::kernel_descriptor_t);

  Expected<std::optional<object::SectionRef>> Rodata =
      findSection(Obj, ".rodata");
  if (!Rodata)
    return Rodata.takeError();
  if (!*Rodata)
    return makeHotswapError("readKernelDescriptor: no .rodata section");

  Expected<StringRef> Contents = (*Rodata)->getContents();
  if (!Contents)
    return Contents.takeError();

  Expected<object::SymbolRef> Symbol =
      COMGR::lookupSymbolByName(Obj, DescriptorSymbolName);
  if (!Symbol)
    return Symbol.takeError();

  // The descriptor must be a defined 64-byte, 64-byte-aligned data object in
  // the selected .rodata; anything else is not an AMDHSA kernel descriptor.
  object::ELFSymbolRef ELFSym(*Symbol);
  Expected<object::section_iterator> SymbolSection = Symbol->getSection();
  if (!SymbolSection)
    return SymbolSection.takeError();
  Expected<uint64_t> Address = Symbol->getAddress();
  if (!Address)
    return Address.takeError();
  if (*SymbolSection == Obj.section_end() || **SymbolSection != **Rodata ||
      ELFSym.getELFType() != ELF::STT_OBJECT ||
      ELFSym.getSize() != DescriptorSize || *Address % DescriptorSize != 0)
    return makeHotswapError(
        formatv("readKernelDescriptor: symbol '{0}' is not a valid kernel "
                "descriptor (wrong type, section, size, or alignment)",
                DescriptorSymbolName));

  uint64_t RodataAddress = (*Rodata)->getAddress();
  if (*Address < RodataAddress)
    return makeHotswapError(
        formatv("readKernelDescriptor: symbol '{0}' at {1:x} precedes .rodata "
                "base {2:x}",
                DescriptorSymbolName, *Address, RodataAddress));
  uint64_t Offset = *Address - RodataAddress;
  if (Offset > Contents->size() || DescriptorSize > Contents->size() - Offset)
    return makeHotswapError(formatv(
        "readKernelDescriptor: symbol '{0}' at {1:x} is outside "
        ".rodata at {2:x} with size {3:x}",
        DescriptorSymbolName, *Address, RodataAddress, Contents->size()));

  const uint8_t *Bytes =
      reinterpret_cast<const uint8_t *>(Contents->data()) + Offset;

  // Reserved bytes must be zero: a nonzero reserved region means the blob is
  // not a descriptor this loader understands.
  auto ReservedZero = [&](uint32_t Start, uint32_t Len) {
    return llvm::all_of(ArrayRef<uint8_t>(Bytes + Start, Len),
                        [](uint8_t B) { return B == 0; });
  };
  if (!ReservedZero(amdhsa::RESERVED0_OFFSET,
                    amdhsa::KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET -
                        amdhsa::RESERVED0_OFFSET) ||
      !ReservedZero(amdhsa::RESERVED1_OFFSET, amdhsa::COMPUTE_PGM_RSRC3_OFFSET -
                                                  amdhsa::RESERVED1_OFFSET) ||
      !ReservedZero(amdhsa::RESERVED3_OFFSET,
                    DescriptorSize - amdhsa::RESERVED3_OFFSET))
    return makeHotswapError(
        formatv("readKernelDescriptor: symbol '{0}' has nonzero reserved bytes",
                DescriptorSymbolName));

  amdhsa::kernel_descriptor_t Descriptor = {};
  Descriptor.group_segment_fixed_size = support::endian::read32le(
      Bytes + amdhsa::GROUP_SEGMENT_FIXED_SIZE_OFFSET);
  Descriptor.private_segment_fixed_size = support::endian::read32le(
      Bytes + amdhsa::PRIVATE_SEGMENT_FIXED_SIZE_OFFSET);
  Descriptor.kernarg_size =
      support::endian::read32le(Bytes + amdhsa::KERNARG_SIZE_OFFSET);
  Descriptor.compute_pgm_rsrc1 =
      support::endian::read32le(Bytes + amdhsa::COMPUTE_PGM_RSRC1_OFFSET);
  Descriptor.compute_pgm_rsrc2 =
      support::endian::read32le(Bytes + amdhsa::COMPUTE_PGM_RSRC2_OFFSET);
  Descriptor.kernel_code_properties =
      support::endian::read16le(Bytes + amdhsa::KERNEL_CODE_PROPERTIES_OFFSET);
  Descriptor.kernarg_preload =
      support::endian::read16le(Bytes + amdhsa::KERNARG_PRELOAD_OFFSET);
  return Descriptor;
}

//===----------------------------------------------------------------------===//
// Per-kernel metadata parsing and validation
//===----------------------------------------------------------------------===//

// Cross-check the descriptor against the MsgPack fields that describe the same
// ABI. `PrivatePresent` records whether .private_segment_fixed_size appeared in
// the metadata; it is authoritative from the descriptor either way but a
// present-and-disagreeing value is a malformed code object.
static Error checkDescriptorAgrees(const amdhsa::kernel_descriptor_t &Desc,
                                   const KernelMeta &Meta,
                                   bool PrivatePresent) {
  if (Desc.group_segment_fixed_size != Meta.GroupSegmentFixedSize)
    return makeHotswapError(formatv(
        "kernel '{0}': metadata and descriptor disagree on group "
        "segment size ({1} vs {2})",
        Meta.Name, Meta.GroupSegmentFixedSize, Desc.group_segment_fixed_size));
  if (PrivatePresent &&
      Desc.private_segment_fixed_size != Meta.PrivateSegmentFixedSize)
    return makeHotswapError(
        formatv("kernel '{0}': metadata and descriptor disagree on private "
                "segment size ({1} vs {2})",
                Meta.Name, Meta.PrivateSegmentFixedSize,
                Desc.private_segment_fixed_size));
  // A zero descriptor kernarg size means "unspecified".
  if (Desc.kernarg_size != 0 && Desc.kernarg_size != Meta.KernargSegmentSize)
    return makeHotswapError(
        formatv("kernel '{0}': metadata and descriptor disagree on kernarg "
                "size ({1} vs {2})",
                Meta.Name, Meta.KernargSegmentSize, Desc.kernarg_size));
  return Error::success();
}

// Validate the completed ABI model. The metadata verifier checks schema and
// field types but not that argument ranges fit within the kernarg segment or
// that constrained values are semantically valid.
static Error validateKernelAbi(const KernelMeta &Meta) {
  SmallVector<std::pair<uint32_t, uint32_t>> Ranges;
  for (const KernelArgMeta &Arg : Meta.Args) {
    if (Arg.Offset > Meta.KernargSegmentSize ||
        Arg.Size > Meta.KernargSegmentSize - Arg.Offset)
      return makeHotswapError(
          formatv("kernel '{0}': argument '{1}' [{2}, {3}) extends beyond the "
                  "kernarg segment of size {4}",
                  Meta.Name, Arg.Name, Arg.Offset,
                  static_cast<uint64_t>(Arg.Offset) + Arg.Size,
                  Meta.KernargSegmentSize));
    Ranges.emplace_back(Arg.Offset, Arg.Offset + Arg.Size);
  }
  llvm::sort(Ranges);
  for (size_t I = 1; I < Ranges.size(); ++I)
    if (Ranges[I].first < Ranges[I - 1].second)
      return makeHotswapError(
          formatv("kernel '{0}': argument ranges overlap", Meta.Name));

  if (Meta.MaxFlatWorkgroupSize == 0)
    return makeHotswapError(
        formatv("kernel '{0}': max_flat_workgroup_size must be at least one",
                Meta.Name));
  return Error::success();
}

// Parse one kernel node into `Meta`, then read, validate, and cross-check its
// descriptor. `Obj` is used only for the descriptor read.
static Error parseKernel(object::ObjectFile &Obj, msgpack::MapDocNode &Kernel,
                         KernelMeta &Meta) {
  if (Error E = readString(Kernel, ".name", "<unnamed>", /*Required=*/true,
                           Meta.Name))
    return E;
  StringRef Name = Meta.Name;
  if (Error E =
          readString(Kernel, ".symbol", Name, /*Required=*/true, Meta.Symbol))
    return E;

  if (Error E = readUInt32(Kernel, ".kernarg_segment_size", Name,
                           /*Required=*/true, Meta.KernargSegmentSize))
    return E;
  if (Error E = readUInt32(Kernel, ".group_segment_fixed_size", Name,
                           /*Required=*/true, Meta.GroupSegmentFixedSize))
    return E;
  if (Error E = readUInt32(Kernel, ".max_flat_workgroup_size", Name,
                           /*Required=*/true, Meta.MaxFlatWorkgroupSize))
    return E;
  // .private_segment_fixed_size is authoritative from the descriptor, so it is
  // optional here and only cross-checked when present.
  bool PrivatePresent = findInMap(Kernel, ".private_segment_fixed_size");
  if (Error E = readUInt32(Kernel, ".private_segment_fixed_size", Name,
                           /*Required=*/false, Meta.PrivateSegmentFixedSize))
    return E;

  if (msgpack::DocNode *Dims = findInMap(Kernel, ".cluster_dims")) {
    if (!Dims->isArray() || Dims->getArray().size() != 3)
      return makeHotswapError(
          formatv("kernel '{0}' has malformed .cluster_dims", Name));
    std::array<uint32_t, 3> Parsed = {};
    unsigned I = 0;
    for (msgpack::DocNode &Dim : Dims->getArray()) {
      std::optional<int64_t> Value = nodeAsInt(Dim);
      if (!Value || *Value < 0 || *Value > UINT32_MAX)
        return makeHotswapError(
            formatv("kernel '{0}' has malformed .cluster_dims", Name));
      Parsed[I++] = static_cast<uint32_t>(*Value);
    }
    Meta.ClusterDims = Parsed;
  }

  if (msgpack::DocNode *Args = findInMap(Kernel, ".args")) {
    if (!Args->isArray())
      return makeHotswapError(
          formatv("kernel '{0}': .args is not an array", Name));
    for (msgpack::DocNode &ArgNode : Args->getArray()) {
      if (!ArgNode.isMap())
        return makeHotswapError(
            formatv("kernel '{0}' has a non-map .args entry", Name));
      msgpack::MapDocNode &ArgMap = ArgNode.getMap();
      KernelArgMeta Arg;
      if (Error E =
              readString(ArgMap, ".name", Name, /*Required=*/false, Arg.Name))
        return E;
      if (Error E = readUInt32(ArgMap, ".offset", Name, /*Required=*/true,
                               Arg.Offset))
        return E;
      if (Error E =
              readUInt32(ArgMap, ".size", Name, /*Required=*/true, Arg.Size))
        return E;
      if (Error E = readString(ArgMap, ".value_kind", Name, /*Required=*/true,
                               Arg.ValueKind))
        return E;
      if (Error E = readString(ArgMap, ".address_space", Name,
                               /*Required=*/false, Arg.AddressSpace))
        return E;
      Meta.Args.push_back(std::move(Arg));
    }
  }

  Expected<amdhsa::kernel_descriptor_t> Descriptor =
      readKernelDescriptor(Obj, Meta.Symbol);
  if (!Descriptor)
    return Descriptor.takeError();
  if (Error E = checkDescriptorAgrees(*Descriptor, Meta, PrivatePresent))
    return E;
  Meta.PrivateSegmentFixedSize = Descriptor->private_segment_fixed_size;
  Meta.ComputePgmRsrc1 = Descriptor->compute_pgm_rsrc1;
  Meta.ComputePgmRsrc2 = Descriptor->compute_pgm_rsrc2;
  Meta.KernelCodeProperties = Descriptor->kernel_code_properties;
  Meta.KernargPreload = Descriptor->kernarg_preload;

  return validateKernelAbi(Meta);
}

//===----------------------------------------------------------------------===//
// CodeObjectInfo
//===----------------------------------------------------------------------===//

Expected<CodeObjectInfo> CodeObjectInfo::create(MemoryBufferRef ElfData) {
  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createELFObjectFile(ElfData);
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  // The raiser's decode and ABI reconstruction assume a little-endian 64-bit
  // AMDGPU HSA code object; reject anything else at the boundary rather than
  // letting later queries misinterpret it.
  auto *ELFObj = dyn_cast<object::ELF64LEObjectFile>(ObjOrErr->get());
  if (!ELFObj)
    return makeHotswapError("code object is not a little-endian 64-bit ELF");
  const auto &Header = ELFObj->getELFFile().getHeader();
  if (Header.e_machine != ELF::EM_AMDGPU)
    return makeHotswapError("code object is not an AMDGPU ELF");
  if (Header.e_ident[ELF::EI_OSABI] != ELF::ELFOSABI_AMDGPU_HSA)
    return makeHotswapError("code object does not use the AMDGPU HSA OS ABI");

  // Symbol lookup walks only .symtab, so a stripped object would degrade into
  // a misleading missing-descriptor result. Refuse it explicitly instead.
  bool HasSymtab = false;
  for (const object::SectionRef &Section : ELFObj->sections())
    if (object::ELFSectionRef(Section).getType() == ELF::SHT_SYMTAB) {
      HasSymtab = true;
      break;
    }
  if (!HasSymtab)
    return makeHotswapError(
        "stripped code objects are not supported: no .symtab section");

  DataMeta Meta;
  Meta.MetaDoc = std::make_shared<COMGR::MetaDocument>();
  Meta.DocNode = Meta.MetaDoc->Document.getRoot();
  if (Error E = COMGR::metadata::getMetadataRoot(ElfData, &Meta))
    return std::move(E);
  msgpack::Document &Doc = Meta.MetaDoc->Document;

  if (Error E = checkMetadataVersion(Doc))
    return std::move(E);

  CodeObjectInfo Info;
  Info.Obj = std::move(*ObjOrErr);

  if (Error E =
          forEachKernelNode(Doc, [&](msgpack::MapDocNode &Kernel) -> Error {
            KernelMeta KM;
            if (Error E = parseKernel(*Info.Obj, Kernel, KM))
              return E;
            if (Info.Kernels.count(KM.Name))
              return makeHotswapError(
                  formatv("duplicate kernel '{0}' in metadata", KM.Name));
            Info.KernelOrder.push_back(KM.Name);
            Info.Kernels[KM.Name] = std::move(KM);
            return Error::success();
          }))
    return std::move(E);

  return Info;
}

Expected<const KernelMeta *>
CodeObjectInfo::kernel(StringRef KernelName) const {
  auto It = Kernels.find(KernelName);
  if (It == Kernels.end())
    return makeHotswapError(
        formatv("kernel '{0}' not found in metadata", KernelName));
  return &It->second;
}

Expected<TextSection> CodeObjectInfo::textSection() const {
  TextSection Result;
  bool FoundText = false;
  for (const object::SectionRef &Section : Obj->sections()) {
    Expected<StringRef> Name = Section.getName();
    if (!Name)
      return Name.takeError();
    if (*Name != ".rodata" && *Name != ".text")
      continue;
    Expected<StringRef> Contents = Section.getContents();
    if (!Contents)
      return Contents.takeError();
    ArrayRef<uint8_t> Bytes = arrayRefFromStringRef(*Contents);
    Result.ImageSections.push_back({Section.getAddress(), Bytes});
    if (*Name == ".text") {
      Result.Address = Section.getAddress();
      Result.Bytes = Bytes;
      FoundText = true;
    }
  }
  if (!FoundText)
    return makeHotswapError("textSection: .text section not found in ELF");
  return Result;
}

// Collect the addresses and sizes of every STT_FUNC symbol inside `.text`,
// sorted by ascending address. Shared by the extent queries so a zero-sized
// symbol is always bounded by the next distinct function address.
namespace {
struct FunctionSymbol {
  uint64_t Address;
  uint64_t Size;
};
} // namespace

static Expected<SmallVector<FunctionSymbol>>
collectTextFunctions(object::ObjectFile &Obj, const object::SectionRef &Text,
                     uint64_t TextBase, uint64_t TextEnd) {
  SmallVector<FunctionSymbol> Functions;
  for (const object::SymbolRef &Symbol : Obj.symbols()) {
    Expected<object::SymbolRef::Type> Type = Symbol.getType();
    if (!Type)
      return Type.takeError();
    if (*Type != object::SymbolRef::ST_Function)
      continue;
    Expected<object::section_iterator> Section = Symbol.getSection();
    if (!Section)
      return Section.takeError();
    if (*Section == Obj.section_end() || **Section != Text)
      continue;
    Expected<uint64_t> Address = Symbol.getAddress();
    if (!Address)
      return Address.takeError();
    if (*Address < TextBase || *Address >= TextEnd)
      continue;
    Functions.push_back({*Address, object::ELFSymbolRef(Symbol).getSize()});
  }
  llvm::sort(Functions, [](const FunctionSymbol &A, const FunctionSymbol &B) {
    return A.Address < B.Address;
  });
  return Functions;
}

// Scanning forward from `Start` in the ascending-sorted `Functions`, the first
// address strictly greater than `Address` (skipping aliases at `Address`), or
// `TextEnd` when none follows.
static uint64_t nextDistinctAddress(ArrayRef<FunctionSymbol> Functions,
                                    size_t Start, uint64_t Address,
                                    uint64_t TextEnd) {
  for (size_t I = Start, E = Functions.size(); I < E; ++I)
    if (Functions[I].Address > Address)
      return Functions[I].Address;
  return TextEnd;
}

Expected<KernelSymbolExtent>
CodeObjectInfo::kernelSymbolExtent(StringRef KernelName) const {
  Expected<std::optional<object::SectionRef>> Text = findSection(*Obj, ".text");
  if (!Text)
    return Text.takeError();
  if (!*Text)
    return makeHotswapError("kernelSymbolExtent: no .text section");
  uint64_t TextBase = (*Text)->getAddress();
  uint64_t TextEnd = TextBase + (*Text)->getSize();

  Expected<object::SymbolRef> Symbol =
      COMGR::lookupSymbolByName(*Obj, KernelName);
  if (!Symbol)
    return Symbol.takeError();
  Expected<object::section_iterator> Section = Symbol->getSection();
  if (!Section)
    return Section.takeError();
  if (*Section == Obj->section_end() || **Section != **Text)
    return makeHotswapError(formatv(
        "kernelSymbolExtent: symbol '{0}' is not in .text", KernelName));
  Expected<uint64_t> Address = Symbol->getAddress();
  if (!Address)
    return Address.takeError();
  if (*Address < TextBase || *Address >= TextEnd)
    return makeHotswapError(
        formatv("kernelSymbolExtent: symbol '{0}' address is outside .text",
                KernelName));

  KernelSymbolExtent Extent;
  Extent.Offset = *Address - TextBase;

  uint64_t SymbolSize = object::ELFSymbolRef(*Symbol).getSize();
  if (SymbolSize != 0) {
    if (SymbolSize > TextEnd - *Address)
      return makeHotswapError(
          formatv("kernelSymbolExtent: symbol '{0}' size extends past .text",
                  KernelName));
    Extent.Size = SymbolSize;
    return Extent;
  }

  // A zero-sized kernel symbol is bounded by the next distinct function-symbol
  // address: symbol placement does not establish ownership, so an intervening
  // helper caps the extent rather than being absorbed into it.
  Expected<SmallVector<FunctionSymbol>> Functions =
      collectTextFunctions(*Obj, **Text, TextBase, TextEnd);
  if (!Functions)
    return Functions.takeError();
  Extent.Size =
      nextDistinctAddress(*Functions, /*Start=*/0, *Address, TextEnd) -
      *Address;
  return Extent;
}

Expected<SmallVector<KernelSymbolExtent>>
CodeObjectInfo::textFunctionExtents() const {
  Expected<std::optional<object::SectionRef>> Text = findSection(*Obj, ".text");
  if (!Text)
    return Text.takeError();
  if (!*Text)
    return makeHotswapError("textFunctionExtents: no .text section");
  uint64_t TextBase = (*Text)->getAddress();
  uint64_t TextEnd = TextBase + (*Text)->getSize();

  Expected<SmallVector<FunctionSymbol>> Functions =
      collectTextFunctions(*Obj, **Text, TextBase, TextEnd);
  if (!Functions)
    return Functions.takeError();

  SmallVector<KernelSymbolExtent> Extents;
  Extents.reserve(Functions->size());
  for (size_t I = 0, E = Functions->size(); I < E; ++I) {
    uint64_t Address = (*Functions)[I].Address;
    uint64_t Size = (*Functions)[I].Size;
    if (Size == 0) {
      // No recorded size: bound by the next greater address (Functions is
      // sorted, so scan forward from I + 1, skipping aliases at this address),
      // or the end of .text for the last one.
      Size = nextDistinctAddress(*Functions, I + 1, Address, TextEnd) - Address;
    } else if (Size > TextEnd - Address) {
      return makeHotswapError(
          formatv("textFunctionExtents: function at {0:x} has size {1:x}, "
                  "extending past .text end {2:x}",
                  Address, Size, TextEnd));
    }
    Extents.push_back({Address - TextBase, Size});
  }
  return Extents;
}

} // namespace COMGR::hotswap
