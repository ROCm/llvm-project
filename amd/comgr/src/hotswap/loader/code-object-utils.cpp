//===- code-object-utils.cpp - Hotswap transpiler -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/loader/code-object-utils.h"

#include "comgr-metadata.h"
#include "comgr-symbol.h"
#include "hotswap/common/hotswap-error.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/AMDGPUMetadataVerifier.h"
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

// Return the section named `Name` or error.
static Expected<object::SectionRef> findSection(object::ObjectFile &Obj,
                                                StringRef Name) {
  for (const object::SectionRef &Section : Obj.sections()) {
    Expected<StringRef> SecName = Section.getName();
    if (!SecName)
      return SecName.takeError();
    if (*SecName == Name)
      return Section;
  }
  return makeHotswapError(formatv("findSection: no section named '{0}'", Name));
}

// Return named DocNode or nullptr if absent.
static msgpack::DocNode *findInMap(msgpack::MapDocNode &Map, StringRef Key) {
  auto It = Map.find(Key);
  return It == Map.end() ? nullptr : &It->second;
}

// Return Node's value as an int64_t or nullopt if not an integer.
static std::optional<int64_t> nodeAsInt(const msgpack::DocNode &Node) {
  if (Node.getKind() == msgpack::Type::Int)
    return Node.getInt();
  if (Node.getKind() == msgpack::Type::UInt)
    return static_cast<int64_t>(Node.getUInt());
  return std::nullopt;
}

// Read `Key` as a uint32. `Required` controls whether an absent key is an
// error; a present-but-not-uint32 value is always an error.
static Error readKernelMetaAsUInt32(msgpack::MapDocNode &Map, StringRef Key,
                                    StringRef KernelName, bool Required,
                                    uint32_t &Out) {
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
static Error readKernelMetaAsString(msgpack::MapDocNode &Map, StringRef Key,
                                    StringRef KernelName, bool Required,
                                    std::string &Out) {
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

// Map e_ident[EI_ABIVERSION] to the AMDGPU code object version. The transpiler
// models the gfx1250 user-SGPR ABI, which appears in code object V4 and later;
// earlier versions are refused at the boundary rather than misparsed with the
// wrong descriptor and metadata layout.
static Expected<uint8_t> codeObjectVersion(uint8_t AbiVersion) {
  switch (AbiVersion) {
  case ELF::ELFABIVERSION_AMDGPU_HSA_V4:
    return 4;
  case ELF::ELFABIVERSION_AMDGPU_HSA_V5:
    return 5;
  case ELF::ELFABIVERSION_AMDGPU_HSA_V6:
    return 6;
  default:
    return makeHotswapError(
        formatv("unsupported AMDGPU code object ABI version {0}; the "
                "transpiler models code object V4 through V6",
                AbiVersion));
  }
}

// Read and validate the `amdhsa.version` [major, minor] pair, returning the
// minor version. Only major version 1 is supported; the minor value and type
// are validated so an unsupported schema is rejected before interpretation.
static Expected<uint32_t> readMetadataMinorVersion(msgpack::Document &Doc) {
  msgpack::DocNode &Root = Doc.getRoot();
  if (!Root.isMap())
    return makeHotswapError("AMDGPU metadata root is not a map");
  msgpack::DocNode *Version = findInMap(Root.getMap(), "amdhsa.version");
  if (!Version)
    return makeHotswapError("AMDGPU metadata has no amdhsa.version");
  if (!Version->isArray() || Version->getArray().size() != 2)
    return makeHotswapError("amdhsa.version is not a [major, minor] array");
  std::optional<int64_t> Major = nodeAsInt(Version->getArray()[0]);
  std::optional<int64_t> Minor = nodeAsInt(Version->getArray()[1]);
  if (!Major || !Minor)
    return makeHotswapError("amdhsa.version major/minor is not an integer");
  if (*Major != 1)
    return makeHotswapError(
        formatv("unsupported AMDGPU metadata version {0}.{1}", *Major, *Minor));
  if (*Minor < 0 || *Minor > UINT32_MAX)
    return makeHotswapError(
        formatv("amdhsa.version minor {0} is out of range", *Minor));
  return static_cast<uint32_t>(*Minor);
}

// Run LLVM's strict AMDGPU metadata verifier as the single source of truth for
// schema and field types: it proves required fields are present and well-typed
// so the extraction below does not re-derive that check. Hotswap keeps only the
// semantic checks the verifier cannot express (version pair, descriptor
// agreement, kernarg ranges).
static Error verifyMetadataSchema(msgpack::Document &Doc) {
  AMDGPU::HSAMD::V3::MetadataVerifier Verifier(/*Strict=*/true);
  if (!Verifier.verify(Doc.getRoot()))
    return makeHotswapError(
        "AMDGPU code-object metadata failed strict schema verification");
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Kernel descriptor
//===----------------------------------------------------------------------===//

// The kernel descriptor together with the absolute source entry address it
// points at (descriptor address + signed kernel_code_entry_byte_offset).
struct DescriptorLoad {
  amdhsa::kernel_descriptor_t Descriptor;
  uint64_t EntryAddress;
};

// Read and validate the 64-byte `<symbol>` kernel descriptor from .rodata. The
// AMDGPU asm printer emits it as an STT_OBJECT there; the descriptor is read
// rather than derived from the MsgPack notes because those omit the kernarg
// preload spec the gfx1250 user-SGPR ABI needs. Fields are read as explicit
// little-endian values since `kernel_descriptor_t` has native integer members.
static Expected<DescriptorLoad>
readKernelDescriptor(object::ObjectFile &Obj, StringRef DescriptorSymbolName) {
  constexpr uint64_t DescriptorSize = sizeof(amdhsa::kernel_descriptor_t);

  Expected<object::SectionRef> Rodata = findSection(Obj, ".rodata");
  if (!Rodata)
    return makeHotswapError(
        formatv("readKernelDescriptor: descriptor '{0}' requires a .rodata "
                "section: {1}",
                DescriptorSymbolName, toString(Rodata.takeError())));

  Expected<StringRef> Contents = Rodata->getContents();
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
  if (*SymbolSection == Obj.section_end() || **SymbolSection != *Rodata ||
      ELFSym.getELFType() != ELF::STT_OBJECT ||
      ELFSym.getSize() != DescriptorSize || *Address % DescriptorSize != 0)
    return makeHotswapError(
        formatv("readKernelDescriptor: symbol '{0}' is not a valid kernel "
                "descriptor (wrong type, section, size, or alignment)",
                DescriptorSymbolName));

  uint64_t RodataAddress = Rodata->getAddress();
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
  Descriptor.kernel_code_entry_byte_offset =
      static_cast<int64_t>(support::endian::read64le(
          Bytes + amdhsa::KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET));
  Descriptor.compute_pgm_rsrc1 =
      support::endian::read32le(Bytes + amdhsa::COMPUTE_PGM_RSRC1_OFFSET);
  Descriptor.compute_pgm_rsrc2 =
      support::endian::read32le(Bytes + amdhsa::COMPUTE_PGM_RSRC2_OFFSET);
  Descriptor.kernel_code_properties =
      support::endian::read16le(Bytes + amdhsa::KERNEL_CODE_PROPERTIES_OFFSET);
  Descriptor.kernarg_preload =
      support::endian::read16le(Bytes + amdhsa::KERNARG_PRELOAD_OFFSET);

  // The entry address is the descriptor address plus its signed code-entry
  // offset: this, not the source `.name`, binds the kernel's ABI to its code.
  // A negative offset larger than the descriptor address would wrap past zero,
  // so reject it here; a positive offset that overshoots is caught later by the
  // .text bounds check on EntryAddress.
  int64_t EntryOffset = Descriptor.kernel_code_entry_byte_offset;
  if (EntryOffset < 0 && static_cast<uint64_t>(-EntryOffset) > *Address)
    return makeHotswapError(formatv(
        "readKernelDescriptor: symbol '{0}' code-entry offset {1} precedes the "
        "descriptor address {2:x}",
        DescriptorSymbolName, EntryOffset, *Address));
  uint64_t EntryAddress = *Address + static_cast<uint64_t>(EntryOffset);
  return DescriptorLoad{Descriptor, EntryAddress};
}

//===----------------------------------------------------------------------===//
// Per-kernel metadata parsing and validation
//===----------------------------------------------------------------------===//

// Cross-check the descriptor against the MsgPack fields that describe the same
// ABI. The strict verifier guarantees .private_segment_fixed_size is present,
// so it is always compared; a disagreement is a malformed code object.
static Error checkDescriptorAgrees(const amdhsa::kernel_descriptor_t &Desc,
                                   const KernelMeta &Meta) {
  if (Desc.group_segment_fixed_size != Meta.GroupSegmentFixedSize)
    return makeHotswapError(formatv(
        "kernel '{0}': metadata and descriptor disagree on group "
        "segment size ({1} vs {2})",
        Meta.Name, Meta.GroupSegmentFixedSize, Desc.group_segment_fixed_size));
  if (Desc.private_segment_fixed_size != Meta.PrivateSegmentFixedSize)
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
// descriptor. `Obj` is used only for the descriptor read. The strict schema
// verifier has already run, so required fields are present and well-typed.
// Every required read below is a `Required=true` extraction rather than a
// schema check, kept so a value's type is never assumed.
static Error parseKernel(object::ObjectFile &Obj, uint8_t CodeObjectVersion,
                         msgpack::MapDocNode &Kernel, KernelMeta &Meta) {
  Meta.CodeObjectVersion = CodeObjectVersion;
  if (Error E = readKernelMetaAsString(Kernel, ".name", "<unnamed>",
                                       /*Required=*/true, Meta.Name))
    return E;
  StringRef Name = Meta.Name;
  if (Error E = readKernelMetaAsString(Kernel, ".symbol", Name,
                                       /*Required=*/true, Meta.Symbol))
    return E;

  if (Error E =
          readKernelMetaAsUInt32(Kernel, ".kernarg_segment_size", Name,
                                 /*Required=*/true, Meta.KernargSegmentSize))
    return E;
  if (Error E =
          readKernelMetaAsUInt32(Kernel, ".group_segment_fixed_size", Name,
                                 /*Required=*/true, Meta.GroupSegmentFixedSize))
    return E;
  if (Error E =
          readKernelMetaAsUInt32(Kernel, ".max_flat_workgroup_size", Name,
                                 /*Required=*/true, Meta.MaxFlatWorkgroupSize))
    return E;
  if (Error E = readKernelMetaAsUInt32(
          Kernel, ".private_segment_fixed_size", Name,
          /*Required=*/true, Meta.PrivateSegmentFixedSize))
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
      if (Error E = readKernelMetaAsString(ArgMap, ".name", Name,
                                           /*Required=*/false, Arg.Name))
        return E;
      if (Error E = readKernelMetaAsUInt32(ArgMap, ".offset", Name,
                                           /*Required=*/true, Arg.Offset))
        return E;
      if (Error E = readKernelMetaAsUInt32(ArgMap, ".size", Name,
                                           /*Required=*/true, Arg.Size))
        return E;
      if (Error E = readKernelMetaAsString(ArgMap, ".value_kind", Name,
                                           /*Required=*/true, Arg.ValueKind))
        return E;
      if (Error E =
              readKernelMetaAsString(ArgMap, ".address_space", Name,
                                     /*Required=*/false, Arg.AddressSpace))
        return E;
      Meta.Args.push_back(std::move(Arg));
    }
  }

  Expected<DescriptorLoad> Loaded = readKernelDescriptor(Obj, Meta.Symbol);
  if (!Loaded)
    return Loaded.takeError();
  const amdhsa::kernel_descriptor_t &Descriptor = Loaded->Descriptor;
  if (Error E = checkDescriptorAgrees(Descriptor, Meta))
    return E;
  Meta.EntryAddress = Loaded->EntryAddress;
  Meta.ComputePgmRsrc1 = Descriptor.compute_pgm_rsrc1;
  Meta.ComputePgmRsrc2 = Descriptor.compute_pgm_rsrc2;
  Meta.KernelCodeProperties = Descriptor.kernel_code_properties;
  Meta.KernargPreload = Descriptor.kernarg_preload;

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

  // Bind the descriptor and metadata ABI layout to the declared code object
  // version before interpreting either, so an unmodelled version is refused
  // rather than parsed with the wrong field semantics.
  Expected<uint8_t> Cov = codeObjectVersion(Header.e_ident[ELF::EI_ABIVERSION]);
  if (!Cov)
    return Cov.takeError();

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

  // The metadata version pair and the code object version must agree: the
  // former selects the schema, the latter the descriptor layout.
  Expected<uint32_t> MinorVersion = readMetadataMinorVersion(Doc);
  if (!MinorVersion)
    return MinorVersion.takeError();
  uint32_t ExpectedMinor = (*Cov == 4) ? 1 : 2;
  if (*MinorVersion != ExpectedMinor)
    return makeHotswapError(
        formatv("code object V{0} declares metadata version 1.{1}, expected "
                "1.{2}",
                *Cov, *MinorVersion, ExpectedMinor));

  if (Error E = verifyMetadataSchema(Doc))
    return std::move(E);

  CodeObjectInfo Info;
  Info.Obj = std::move(*ObjOrErr);

  if (Error E =
          forEachKernelNode(Doc, [&](msgpack::MapDocNode &Kernel) -> Error {
            KernelMeta KM;
            if (Error E = parseKernel(*Info.Obj, *Cov, Kernel, KM))
              return E;
            if (Info.Kernels.count(KM.Name))
              return makeHotswapError(
                  formatv("duplicate kernel '{0}' in metadata", KM.Name));
            std::string KernelName = KM.Name;
            Info.Kernels.try_emplace(KernelName, std::move(KM));
            Info.KernelOrder.push_back(std::move(KernelName));
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
  // Sort by ascending address, breaking ties by descending size so the first
  // symbol at each address carries the largest recorded extent.
  llvm::sort(Functions, [](const FunctionSymbol &A, const FunctionSymbol &B) {
    if (A.Address != B.Address)
      return A.Address < B.Address;
    return A.Size > B.Size;
  });
  // Collapse aliases: several symbols may share one address, but they name a
  // single function. Keep the canonical (largest) one so the extent queries do
  // not emit duplicate or zero-sized ranges for it.
  Functions.erase(
      llvm::unique(Functions,
                   [](const FunctionSymbol &A, const FunctionSymbol &B) {
                     return A.Address == B.Address;
                   }),
      Functions.end());
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
  Expected<const KernelMeta *> Meta = kernel(KernelName);
  if (!Meta)
    return Meta.takeError();
  // The entry comes from the descriptor's code-entry offset, not a lookup of
  // `.name` in .text: `.symbol` may name a different symbol, so only the
  // descriptor authoritatively binds this kernel's ABI to its code.
  uint64_t Entry = (*Meta)->EntryAddress;

  Expected<object::SectionRef> Text = findSection(*Obj, ".text");
  if (!Text)
    return Text.takeError();
  uint64_t TextBase = Text->getAddress();
  uint64_t TextEnd = TextBase + Text->getSize();
  if (Entry < TextBase || Entry >= TextEnd)
    return makeHotswapError(
        formatv("kernelSymbolExtent: kernel '{0}' entry {1:x} is outside .text "
                "[{2:x}, {3:x})",
                KernelName, Entry, TextBase, TextEnd));

  // Bound the entry by the enclosing function symbol: use its recorded size
  // when present, otherwise the next distinct function address (symbol
  // placement does not establish ownership, so an intervening helper caps the
  // extent).
  Expected<SmallVector<FunctionSymbol>> Functions =
      collectTextFunctions(*Obj, *Text, TextBase, TextEnd);
  if (!Functions)
    return Functions.takeError();

  KernelSymbolExtent Extent;
  Extent.Offset = Entry - TextBase;

  uint64_t SymbolSize = 0;
  for (const FunctionSymbol &F : *Functions)
    if (F.Address == Entry) {
      SymbolSize = F.Size;
      break;
    }
  if (SymbolSize != 0) {
    if (SymbolSize > TextEnd - Entry)
      return makeHotswapError(
          formatv("kernelSymbolExtent: kernel '{0}' size extends past .text",
                  KernelName));
    Extent.Size = SymbolSize;
    return Extent;
  }
  Extent.Size =
      nextDistinctAddress(*Functions, /*Start=*/0, Entry, TextEnd) - Entry;
  return Extent;
}

Expected<SmallVector<KernelSymbolExtent>>
CodeObjectInfo::textFunctionExtents() const {
  Expected<object::SectionRef> Text = findSection(*Obj, ".text");
  if (!Text)
    return Text.takeError();
  uint64_t TextBase = Text->getAddress();
  uint64_t TextEnd = TextBase + Text->getSize();

  Expected<SmallVector<FunctionSymbol>> Functions =
      collectTextFunctions(*Obj, *Text, TextBase, TextEnd);
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
