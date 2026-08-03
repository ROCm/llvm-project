//===- code-object-utils.cpp - AMDGPU code-object metadata --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "code-object-utils.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Error.h"

namespace COMGR::hotswap {
namespace {

// Look up `Key` in `Map`. Returns null when the key is absent.
// `MapDocNode::find(StringRef)` allocates the lookup key on `Map`'s owning
// document, so callers need only pass the literal string.
llvm::msgpack::DocNode *findInMap(llvm::msgpack::MapDocNode &Map,
                                  llvm::StringRef Key) {
  auto It = Map.find(Key);
  return (It == Map.end()) ? nullptr : &It->second;
}

// Iterate the `amdhsa.kernels` array of a parsed AMDGPU MsgPack document and
// invoke `CB` on each kernel map node. Silently skips non-map children.
template <class Fn>
void forEachKernelNode(llvm::msgpack::Document &Doc, Fn &&CB) {
  llvm::msgpack::DocNode &Root = Doc.getRoot();
  if (!Root.isMap())
    return;
  llvm::msgpack::DocNode *Kernels = findInMap(Root.getMap(), "amdhsa.kernels");
  if (!Kernels || !Kernels->isArray())
    return;
  for (auto &K : Kernels->getArray()) {
    if (!K.isMap())
      continue;
    CB(K.getMap());
  }
}

} // namespace

llvm::Expected<llvm::SmallVector<std::string>>
listKernelNames(llvm::MemoryBufferRef ElfData) {
  using namespace llvm;

  Expected<object::ELF64LEFile> ElfOrErr =
      object::ELF64LEFile::create(ElfData.getBuffer());
  if (!ElfOrErr)
    return ElfOrErr.takeError();
  const object::ELF64LEFile &Elf = *ElfOrErr;

  Expected<ArrayRef<object::ELF64LE::Shdr>> SectionsOrErr = Elf.sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();

  SmallVector<std::string> Names;
  bool FoundNote = false;
  for (const object::ELF64LE::Shdr &Sec : *SectionsOrErr) {
    if (Sec.sh_type != ELF::SHT_NOTE)
      continue;
    Error Err = Error::success();
    for (const object::ELF64LE::Note &Note : Elf.notes(Sec, Err)) {
      if (Note.getType() != ELF::NT_AMDGPU_METADATA ||
          Note.getName() != "AMDGPU")
        continue;
      ArrayRef<uint8_t> Desc = Note.getDesc(Sec.sh_addralign);
      msgpack::Document Doc;
      if (!Doc.readFromBlob(
              StringRef(reinterpret_cast<const char *>(Desc.data()),
                        Desc.size()),
              /*Multi=*/false))
        return createStringError(
            "listKernelNames: failed to parse AMDGPU metadata note");
      forEachKernelNode(Doc, [&](msgpack::MapDocNode &KMap) {
        if (msgpack::DocNode *Name = findInMap(KMap, ".name"))
          Names.push_back(Name->toString());
      });
      FoundNote = true;
    }
    if (Err)
      return std::move(Err);
  }

  if (!FoundNote)
    return createStringError("listKernelNames: no AMDGPU metadata note");
  return Names;
}

} // namespace COMGR::hotswap
