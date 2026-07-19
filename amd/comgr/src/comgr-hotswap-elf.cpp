//===- comgr-hotswap-elf.cpp - ELF helpers and trampoline growth ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of hotswap::ElfView and the free-function ELF helpers.
/// Parses are delegated to llvm::object::ELFFile. ElfView caches immutable
/// symbol ranges and metadata-derived SGPR counts for the duration of one
/// rewrite so large code objects do not repeatedly parse the same ELF data.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

using Ehdr = ELF::Elf64_Ehdr;
using Shdr = ELF::Elf64_Shdr;
using Phdr = ELF::Elf64_Phdr;
using ELFT = ElfView::ELFT;
using ELFFileT = ElfView::ELFFileT;

// This file depends on the COMPUTE_PGM_RSRC1_GRANULATED_* field layout below.
// Assert it so the dependency is caught at compile time if it ever shifts.
static_assert(
    amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT == 0 &&
        amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH == 6,
    "GRANULATED_WORKITEM_VGPR_COUNT layout changed unexpectedly.");
static_assert(
    amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT == 6 &&
        amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH == 4,
    "GRANULATED_WAVEFRONT_SGPR_COUNT layout changed unexpectedly.");

static constexpr unsigned SgprEncodingGranule = 8;

// Page alignment for the appended trampoline pool's virtual address and file
// offset, so its PT_LOAD segment maps consistently.
static constexpr uint64_t TrampolinePoolAlign = 4096;

// Standard ELF note identifying one executable pool produced by HotSwap. The
// description is a fixed little-endian schema:
//   u32 version, u32 target state, u64 pool vaddr, u64 pool size.
// The section carrying the note is deliberately non-allocating; loaders need
// not map provenance that is consumed only by a later rewrite.
static constexpr StringLiteral HotswapPoolNoteName = "AMDGPU";
static constexpr uint32_t HotswapPoolNoteType = 0x48535750; // "HSWP"
static constexpr uint32_t HotswapPoolNoteVersion = 1;
static constexpr uint32_t HotswapPoolNoteDescSize = 24;

static std::optional<uint64_t> checkedAlignToUint64(uint64_t Value,
                                                    uint64_t Alignment,
                                                    StringRef Context) {
  if (Alignment == 0) {
    log() << "hotswap: error: " << Context << " has zero alignment.\n";
    return std::nullopt;
  }
  const uint64_t Remainder = Value % Alignment;
  if (Remainder == 0)
    return Value;
  return checkedAddUint64(Value, Alignment - Remainder, Context);
}

static SmallVector<uint8_t, 48>
buildHotswapPoolNote(uint64_t PoolVAddr, uint64_t PoolSize,
                     ExecutablePoolTargetState TargetState) {
  const uint32_t NameSize = HotswapPoolNoteName.size() + 1;
  const uint32_t NameStorageSize = alignTo(NameSize, uint32_t{4});
  SmallVector<uint8_t, 48> Bytes(12 + NameStorageSize +
                                    HotswapPoolNoteDescSize,
                                0);
  support::endian::write32le(Bytes.data(), NameSize);
  support::endian::write32le(Bytes.data() + 4, HotswapPoolNoteDescSize);
  support::endian::write32le(Bytes.data() + 8, HotswapPoolNoteType);
  std::memcpy(Bytes.data() + 12, HotswapPoolNoteName.data(),
              HotswapPoolNoteName.size());
  uint8_t *Desc = Bytes.data() + 12 + NameStorageSize;
  support::endian::write32le(Desc, HotswapPoolNoteVersion);
  support::endian::write32le(Desc + 4,
                             static_cast<uint32_t>(TargetState));
  support::endian::write64le(Desc + 8, PoolVAddr);
  support::endian::write64le(Desc + 16, PoolSize);
  return Bytes;
}

enum class MetadataSgprUpdateStatus {
  NotFound,
  Found,
  Error,
};

enum class MetadataCountUpdateStatus {
  NotFound,
  Found,
  Error,
};

static std::optional<uint64_t> checkedSectionFileOffset(const ELFT::Shdr &Sec,
                                                        uint64_t VAddr,
                                                        uint64_t AccessSize,
                                                        uint64_t FileSize,
                                                        StringRef Context) {
  if (VAddr < Sec.sh_addr) {
    log() << "hotswap: error: " << Context << " has vaddr 0x"
          << utohexstr(VAddr) << " before containing section vaddr 0x"
          << utohexstr(Sec.sh_addr) << ".\n";
    return std::nullopt;
  }

  uint64_t Delta = VAddr - Sec.sh_addr;
  std::optional<uint64_t> FileOffset = checkedAddUint64(
      Sec.sh_offset, Delta, (Twine(Context) + " file offset").str());
  if (!FileOffset)
    return std::nullopt;

  if (AccessSize > FileSize || *FileOffset > FileSize - AccessSize) {
    log() << "hotswap: error: " << Context
          << " extends past end of ELF at file offset 0x"
          << utohexstr(*FileOffset) << ".\n";
    return std::nullopt;
  }
  return FileOffset;
}

static std::optional<unsigned>
readUnsignedMetadataNode(const msgpack::DocNode &Node, StringRef KernelName,
                         StringRef Key, StringRef Context) {
  if (Node.getKind() == msgpack::Type::UInt) {
    uint64_t Value = Node.getUInt();
    if (Value > std::numeric_limits<unsigned>::max()) {
      log() << "hotswap: error: " << Context << ": " << Key << " for '"
            << KernelName << "' exceeds unsigned.\n";
      return std::nullopt;
    }
    return static_cast<unsigned>(Value);
  }

  if (Node.getKind() == msgpack::Type::Int) {
    int64_t Value = Node.getInt();
    if (Value < 0 ||
        static_cast<uint64_t>(Value) > std::numeric_limits<unsigned>::max()) {
      log() << "hotswap: error: " << Context << ": " << Key << " for '"
            << KernelName << "' is outside unsigned range.\n";
      return std::nullopt;
    }
    return static_cast<unsigned>(Value);
  }

  log() << "hotswap: error: " << Context << ": " << Key << " for '"
        << KernelName << "' is not an integer.\n";
  return std::nullopt;
}

static std::optional<unsigned>
readSgprCountMetadataNode(const msgpack::DocNode &SgprNode,
                          StringRef KernelName, StringRef Context) {
  return readUnsignedMetadataNode(SgprNode, KernelName, ".sgpr_count", Context);
}

static MetadataSgprUpdateStatus
rewriteKernelMetadataSgprCounts(uint8_t *Elf, const ELFFileT &File,
                                const StringMap<unsigned> &RequiredSgprs) {
  if (RequiredSgprs.empty())
    return MetadataSgprUpdateStatus::Found;

  struct PendingWrite {
    size_t Offset;
    std::string Blob;
  };

  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: updateKernelMetadataSgprCounts: failed to read "
          << "program headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return MetadataSgprUpdateStatus::Error;
  }

  bool SawMetadataNote = false;
  StringMap<bool> Found;
  std::vector<PendingWrite> PendingWrites;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;
      SawMetadataNote = true;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: updateKernelMetadataSgprCounts: AMDGPU "
              << "metadata note has an empty descriptor.\n";
        return MetadataSgprUpdateStatus::Error;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: updateKernelMetadataSgprCounts: failed to "
              << "parse AMDGPU metadata note.\n";
        return MetadataSgprUpdateStatus::Error;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: updateKernelMetadataSgprCounts: AMDGPU "
              << "metadata root is not a map.\n";
        return MetadataSgprUpdateStatus::Error;
      }

      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator KernelsIt =
          RootMap.find("amdhsa.kernels");
      if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
        continue;

      bool Modified = false;
      msgpack::ArrayDocNode &KernelArray = KernelsIt->second.getArray();
      for (msgpack::DocNode &KNode : KernelArray) {
        if (!KNode.isMap())
          continue;

        msgpack::MapDocNode &KMap = KNode.getMap();
        msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
        if (NameIt == KMap.end() || !NameIt->second.isString())
          continue;
        StringRef KernelName = NameIt->second.getString();
        StringMap<unsigned>::const_iterator Required =
            RequiredSgprs.find(KernelName);
        if (Required == RequiredSgprs.end() || Found.contains(KernelName))
          continue;

        msgpack::DocNode::MapTy::iterator SgprIt = KMap.find(".sgpr_count");
        if (SgprIt == KMap.end()) {
          log() << "hotswap: error: updateKernelMetadataSgprCounts: metadata "
                << "for kernel '" << KernelName << "' has no .sgpr_count.\n";
          return MetadataSgprUpdateStatus::Error;
        }

        std::optional<unsigned> CurrentSgprs = readSgprCountMetadataNode(
            SgprIt->second, KernelName, "updateKernelMetadataSgprCounts");
        if (!CurrentSgprs)
          return MetadataSgprUpdateStatus::Error;
        Found.try_emplace(KernelName, true);
        if (Required->second <= *CurrentSgprs)
          continue;

        SgprIt->second = static_cast<uint64_t>(Required->second);
        Modified = true;
      }

      if (Modified) {
        std::string NewBlob;
        Doc.writeToBlob(NewBlob);
        if (NewBlob.size() != Blob.size()) {
          log() << "hotswap: error: updateKernelMetadataSgprCounts: updating "
                   ".sgpr_count changes metadata note size from "
                << Blob.size() << " to " << NewBlob.size()
                << " bytes; in-place rewrite cannot preserve ELF layout.\n";
          return MetadataSgprUpdateStatus::Error;
        }

        const uint8_t *DescBegin = Desc.data();
        if (DescBegin < File.base() || DescBegin >= File.end()) {
          log() << "hotswap: error: updateKernelMetadataSgprCounts: metadata "
                << "descriptor pointer is outside the ELF buffer.\n";
          return MetadataSgprUpdateStatus::Error;
        }
        size_t DescOffset = DescBegin - File.base();
        if (Desc.size() > File.getBufSize() ||
            DescOffset > File.getBufSize() - Desc.size()) {
          log() << "hotswap: error: updateKernelMetadataSgprCounts: metadata "
                << "descriptor extends past the ELF buffer.\n";
          return MetadataSgprUpdateStatus::Error;
        }
        PendingWrites.push_back({DescOffset, std::move(NewBlob)});
      }

      if (Found.size() == RequiredSgprs.size())
        break;
    }

    if (Err) {
      log() << "hotswap: error: updateKernelMetadataSgprCounts: failed to "
            << "iterate AMDGPU notes: " << toString(std::move(Err)) << "\n";
      return MetadataSgprUpdateStatus::Error;
    }
    if (Found.size() == RequiredSgprs.size())
      break;
  }

  if (SawMetadataNote) {
    for (const StringMapEntry<unsigned> &Required : RequiredSgprs) {
      if (Found.contains(Required.first()))
        continue;
      log() << "hotswap: error: updateKernelMetadataSgprCounts: AMDGPU "
               "metadata has no entry for kernel '"
            << Required.first() << "'.\n";
      return MetadataSgprUpdateStatus::Error;
    }
    for (const PendingWrite &Write : PendingWrites)
      std::memcpy(Elf + Write.Offset, Write.Blob.data(), Write.Blob.size());
    return MetadataSgprUpdateStatus::Found;
  }
  return MetadataSgprUpdateStatus::NotFound;
}

bool ElfView::updateGfx1250RevisionMetadata(StringRef Revision) {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: updateGfx1250RevisionMetadata: failed to read "
          << "program headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return false;
  }

  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: updateGfx1250RevisionMetadata: AMDGPU "
              << "metadata note has an empty descriptor.\n";
        return false;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: updateGfx1250RevisionMetadata: failed to "
              << "parse AMDGPU metadata note.\n";
        return false;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: updateGfx1250RevisionMetadata: AMDGPU "
              << "metadata root is not a map.\n";
        return false;
      }

      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator KernelsIt =
          RootMap.find("amdhsa.kernels");
      if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
        continue;

      bool Changed = false;
      for (msgpack::DocNode &KNode : KernelsIt->second.getArray()) {
        if (!KNode.isMap())
          continue;
        msgpack::MapDocNode &KMap = KNode.getMap();
        msgpack::DocNode::MapTy::iterator RevisionIt =
            KMap.find(".gfx1250_revision");
        if (RevisionIt == KMap.end())
          continue;
        if (!RevisionIt->second.isString()) {
          log() << "hotswap: error: updateGfx1250RevisionMetadata: "
                << ".gfx1250_revision is not a string.\n";
          return false;
        }
        if (RevisionIt->second.getString() == Revision)
          continue;
        RevisionIt->second = Doc.getNode(Revision, /*Copy=*/true);
        Changed = true;
      }

      if (!Changed)
        continue;

      std::string NewBlob;
      Doc.writeToBlob(NewBlob);
      if (NewBlob.size() != Blob.size()) {
        log() << "hotswap: error: updateGfx1250RevisionMetadata: updating "
              << ".gfx1250_revision changes metadata note size from "
              << Blob.size() << " to " << NewBlob.size()
              << " bytes; in-place rewrite cannot preserve ELF layout.\n";
        return false;
      }

      const uint8_t *DescBegin = Desc.data();
      if (DescBegin < File.base() || DescBegin >= File.end()) {
        log() << "hotswap: error: updateGfx1250RevisionMetadata: metadata "
              << "descriptor pointer is outside the ELF buffer.\n";
        return false;
      }
      size_t DescOffset = DescBegin - File.base();
      if (Desc.size() > File.getBufSize() ||
          DescOffset > File.getBufSize() - Desc.size()) {
        log() << "hotswap: error: updateGfx1250RevisionMetadata: metadata "
              << "descriptor extends past the ELF buffer.\n";
        return false;
      }
      std::memcpy(data() + DescOffset, NewBlob.data(), NewBlob.size());
    }

    if (Err) {
      log() << "hotswap: error: updateGfx1250RevisionMetadata: failed to "
            << "iterate AMDGPU notes: " << toString(std::move(Err)) << "\n";
      return false;
    }
  }
  return true;
}

std::optional<bool>
ElfView::allKernelsHaveGfx1250Revision(StringRef Revision) const {
  if (!kernelDescriptorCacheIsComplete())
    return false;

  StringMap<bool> DescriptorSeen;
  for (const KernelDescriptorInfo &Descriptor : kernelDescriptors())
    DescriptorSeen.try_emplace(Descriptor.KernelName, false);
  if (DescriptorSeen.empty())
    return false;

  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: allKernelsHaveGfx1250Revision: failed to read "
             "program headers: "
          << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }

  bool FoundKernel = false;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: allKernelsHaveGfx1250Revision: AMDGPU "
                 "metadata note has an empty descriptor.\n";
        return std::nullopt;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: allKernelsHaveGfx1250Revision: failed to "
                 "parse AMDGPU metadata note.\n";
        return std::nullopt;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: allKernelsHaveGfx1250Revision: AMDGPU "
                 "metadata root is not a map.\n";
        return std::nullopt;
      }

      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator KernelsIt =
          RootMap.find("amdhsa.kernels");
      if (KernelsIt == RootMap.end())
        continue;
      if (!KernelsIt->second.isArray()) {
        log() << "hotswap: error: allKernelsHaveGfx1250Revision: "
                 "amdhsa.kernels is not an array.\n";
        return std::nullopt;
      }

      for (msgpack::DocNode &KNode : KernelsIt->second.getArray()) {
        if (!KNode.isMap()) {
          log() << "hotswap: error: allKernelsHaveGfx1250Revision: kernel "
                   "metadata entry is not a map.\n";
          return std::nullopt;
        }
        msgpack::MapDocNode &KMap = KNode.getMap();
        msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
        if (NameIt == KMap.end() || !NameIt->second.isString()) {
          log() << "hotswap: error: allKernelsHaveGfx1250Revision: kernel "
                   "metadata has no string .name.\n";
          return std::nullopt;
        }
        StringMap<bool>::iterator DescriptorIt =
            DescriptorSeen.find(NameIt->second.getString());
        if (DescriptorIt == DescriptorSeen.end())
          return false;
        DescriptorIt->second = true;

        msgpack::DocNode::MapTy::iterator RevisionIt =
            KMap.find(".gfx1250_revision");
        if (RevisionIt == KMap.end())
          return false;
        if (!RevisionIt->second.isString()) {
          log() << "hotswap: error: allKernelsHaveGfx1250Revision: "
                   ".gfx1250_revision is not a string.\n";
          return std::nullopt;
        }
        FoundKernel = true;
        if (RevisionIt->second.getString() != Revision)
          return false;
      }
    }
    if (Err) {
      log() << "hotswap: error: allKernelsHaveGfx1250Revision: failed to "
               "iterate AMDGPU notes: "
            << toString(std::move(Err)) << "\n";
      return std::nullopt;
    }
  }
  return FoundKernel &&
         llvm::all_of(DescriptorSeen,
                      [](const auto &Entry) { return Entry.getValue(); });
}

using MetadataNoteMutator = function_ref<std::optional<bool>(
    msgpack::Document &, msgpack::MapDocNode &)>;
using MetadataNoteValidator = function_ref<bool(bool)>;

struct PendingMetadataWrite {
  size_t Offset = 0;
  std::string Blob;
};

/// Parse each AMDGPU metadata note, invoke \p Mutator on its root map, and
/// defer changed writes until \p Validator accepts the complete traversal.
/// This keeps multi-note updates atomic while sharing the parsing, encoded-size
/// and destination validation for count mutations.
static bool rewriteMetadataNotes(uint8_t *Elf, const ELFFileT &File,
                                 StringRef Context, MetadataNoteMutator Mutator,
                                 MetadataNoteValidator Validator,
                                 bool &SawMetadataNote) {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: " << Context << ": failed to read program "
          << "headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return false;
  }
  std::vector<PendingMetadataWrite> PendingWrites;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;
      SawMetadataNote = true;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata note has an empty descriptor.\n";
        return false;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: " << Context
              << ": failed to parse AMDGPU metadata note.\n";
        return false;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata root is not a map.\n";
        return false;
      }

      std::optional<bool> Changed = Mutator(Doc, Root.getMap());
      if (!Changed)
        return false;
      if (!*Changed)
        continue;

      std::string NewBlob;
      Doc.writeToBlob(NewBlob);
      if (NewBlob.size() != Blob.size()) {
        log() << "hotswap: error: " << Context
              << ": updating AMDGPU metadata changes note size from "
              << Blob.size() << " to " << NewBlob.size()
              << " bytes; in-place rewrite cannot preserve ELF layout.\n";
        return false;
      }

      const uint8_t *DescBegin = Desc.data();
      if (DescBegin < File.base() || DescBegin >= File.end()) {
        log() << "hotswap: error: " << Context
              << ": metadata descriptor pointer is outside the ELF buffer.\n";
        return false;
      }
      size_t DescOffset = DescBegin - File.base();
      if (Desc.size() > File.getBufSize() ||
          DescOffset > File.getBufSize() - Desc.size()) {
        log() << "hotswap: error: " << Context
              << ": metadata descriptor extends past the ELF buffer.\n";
        return false;
      }
      PendingWrites.push_back({DescOffset, std::move(NewBlob)});
    }

    if (Err) {
      log() << "hotswap: error: " << Context
            << ": failed to iterate AMDGPU notes: " << toString(std::move(Err))
            << "\n";
      return false;
    }
  }

  if (!Validator(SawMetadataNote))
    return false;
  for (const PendingMetadataWrite &Write : PendingWrites)
    std::memcpy(Elf + Write.Offset, Write.Blob.data(), Write.Blob.size());
  return true;
}

static MetadataCountUpdateStatus
rewriteKernelMetadataCounts(uint8_t *Elf, const ELFFileT &File,
                            const StringMap<unsigned> &RequiredCounts,
                            StringRef MetadataKey, StringRef Context) {
  if (RequiredCounts.empty())
    return MetadataCountUpdateStatus::Found;

  bool SawMetadataNote = false;
  StringMap<bool> Found;
  bool Rewritten = rewriteMetadataNotes(
      Elf, File, Context,
      [&](msgpack::Document &,
          msgpack::MapDocNode &RootMap) -> std::optional<bool> {
        msgpack::DocNode::MapTy::iterator KernelsIt =
            RootMap.find("amdhsa.kernels");
        if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
          return false;

        bool Changed = false;
        for (msgpack::DocNode &KNode : KernelsIt->second.getArray()) {
          if (!KNode.isMap())
            continue;
          msgpack::MapDocNode &KMap = KNode.getMap();
          msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
          if (NameIt == KMap.end() || !NameIt->second.isString())
            continue;

          StringRef KernelName = NameIt->second.getString();
          StringMap<unsigned>::const_iterator Required =
              RequiredCounts.find(KernelName);
          if (Required == RequiredCounts.end() || Found.contains(KernelName))
            continue;

          msgpack::DocNode::MapTy::iterator CountIt = KMap.find(MetadataKey);
          if (CountIt == KMap.end()) {
            log() << "hotswap: error: " << Context << ": metadata for kernel '"
                  << KernelName << "' has no " << MetadataKey << ".\n";
            return std::nullopt;
          }

          std::optional<unsigned> CurrentCount = readUnsignedMetadataNode(
              CountIt->second, KernelName, MetadataKey, Context);
          if (!CurrentCount)
            return std::nullopt;
          Found.try_emplace(KernelName, true);
          if (Required->second <= *CurrentCount)
            continue;

          CountIt->second = static_cast<uint64_t>(Required->second);
          Changed = true;
        }
        return Changed;
      },
      [&](bool SawMetadata) {
        if (!SawMetadata)
          return true;
        for (const StringMapEntry<unsigned> &Required : RequiredCounts) {
          if (Found.contains(Required.first()))
            continue;
          log() << "hotswap: error: " << Context
                << ": AMDGPU metadata has no entry for kernel '"
                << Required.first() << "'.\n";
          return false;
        }
        return true;
      },
      SawMetadataNote);
  if (!Rewritten)
    return MetadataCountUpdateStatus::Error;
  return SawMetadataNote ? MetadataCountUpdateStatus::Found
                         : MetadataCountUpdateStatus::NotFound;
}

// -- applyByteReplace ---------------------------------------------------------

bool applyByteReplace(const RewriteRule &Rule, uint64_t InstOffset,
                      uint32_t InstSize, uint8_t *Text, uint64_t TextSize,
                      const LLVMState &S) {
  if (InstOffset > TextSize || InstSize > TextSize - InstOffset) {
    log() << "hotswap: error: applyByteReplace: instruction range [0x"
          << utohexstr(InstOffset) << ", 0x"
          << utohexstr(InstOffset + static_cast<uint64_t>(InstSize))
          << ") extends past .text size 0x" << utohexstr(TextSize) << ".\n";
    return false;
  }
  const size_t ReplaceSize = Rule.ReplaceBytes.size();
  if (ReplaceSize > InstSize) {
    log() << "hotswap: error: applyByteReplace: replacement size "
          << ReplaceSize << " exceeds original instruction size " << InstSize
          << " at .text offset 0x" << utohexstr(InstOffset) << ".\n";
    return false;
  }
  if (S.SNopBytes.size() != MinInstSize) {
    log() << "hotswap: error: applyByteReplace: cached s_nop size "
          << S.SNopBytes.size() << " does not match expected size "
          << MinInstSize << ".\n";
    return false;
  }
  std::memcpy(Text + InstOffset, Rule.ReplaceBytes.data(), ReplaceSize);
  uint64_t PadOffset = InstOffset + ReplaceSize;
  uint64_t Remaining = InstSize - ReplaceSize;
  while (Remaining >= MinInstSize) {
    std::memcpy(Text + PadOffset, S.SNopBytes.data(), MinInstSize);
    PadOffset += MinInstSize;
    Remaining -= MinInstSize;
  }
  return true;
}

// -- findNearestSled ----------------------------------------------------------

NopSled *findNearestSled(std::vector<NopSled> &Sleds, uint64_t Offset,
                         uint64_t Needed, NopSledUse Use) {
  NopSled *Best = nullptr;
  uint64_t BestDist = std::numeric_limits<uint64_t>::max();
  bool BestOwned = false;
  for (NopSled &Sled : Sleds) {
    if (Sled.GatewayOnly && Use != NopSledUse::Gateway)
      continue;
    const bool Eligible =
        Use == NopSledUse::Gateway
            ? Sled.canGatewayFrom(Offset)
            : Sled.canHoldBodyFrom(
                  Offset, Use == NopSledUse::RelocationBody);
    if (!Eligible)
      continue;
    if (Sled.WritePos > Sled.End || Needed > Sled.End - Sled.WritePos)
      continue;
    uint64_t Dist = Sled.WritePos > Offset ? Sled.WritePos - Offset
                                           : Offset - Sled.WritePos;
    const bool Owned = Sled.ownsSource(Offset);
    const bool PreferOwned = Use == NopSledUse::RelocationBody;
    const bool Better =
        !Best ||
        (PreferOwned ? (Owned && !BestOwned) ||
                           (Owned == BestOwned && Dist < BestDist)
                     : Dist < BestDist);
    if (Dist < MaxSledDistance && Better) {
      Best = &Sled;
      BestDist = Dist;
      BestOwned = Owned;
    }
  }
  return Best;
}

// -- ElfView::create ----------------------------------------------------------

Expected<ElfView> ElfView::create(uint8_t *Data, size_t Size) {
  // Data/Size are kept as factory parameters to document that the caller
  // must hand in a mutable buffer (hotswap mutates bytes through the
  // resulting ElfView). Once ELFFile is constructed, it owns the structural
  // view over these same bytes and we do not need to store Data/Size
  // separately -- ELFFile::base() / ELFFile::getBufSize() alias them.
  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Data), Size));
  if (!FileOrErr)
    return FileOrErr.takeError();

  const ELFFileT &File = *FileOrErr;
  Expected<ELFT::ShdrRange> SectionsOrErr = File.sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  ELFT::ShdrRange Sections = *SectionsOrErr;

  // HotSwap rewrites reason about symbol ownership by section index. Silently
  // treating SHN_XINDEX as a reserved/non-.text index would omit protected
  // entries and extents. Extended symbol indices are uncommon in code objects;
  // reject them until every symbol-table mutation preserves the associated
  // SHT_SYMTAB_SHNDX table.
  for (const ELFT::Shdr &Shdr : Sections) {
    if (Shdr.sh_type != ELF::SHT_SYMTAB && Shdr.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&Shdr);
    if (!SymsOrErr)
      return SymsOrErr.takeError();
    if (llvm::any_of(*SymsOrErr, [](const ELFT::Sym &Sym) {
          return Sym.st_shndx == ELF::SHN_XINDEX;
        }))
      return createStringError(
          object::object_error::parse_failed,
          "HotSwap does not support SHN_XINDEX symbol section indices");
  }

  const ELFT::Shdr *Text = nullptr;
  unsigned TextIdx = 0;
  unsigned Idx = 0;
  for (const ELFT::Shdr &Shdr : Sections) {
    Expected<StringRef> NameOrErr = File.getSectionName(Shdr);
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      ++Idx;
      continue;
    }
    if (*NameOrErr == ".text" && Shdr.sh_offset <= Size &&
        Shdr.sh_size <= Size - Shdr.sh_offset) {
      Text = &Shdr;
      TextIdx = Idx;
      break;
    }
    ++Idx;
  }
  if (!Text)
    return createStringError(object::object_error::parse_failed,
                             "no .text section found");
  return ElfView(std::move(*FileOrErr), Sections, Text, TextIdx);
}

// -- ElfView::functionTextRanges ---------------------------------------------

ArrayRef<ElfView::FunctionTextRange> ElfView::cachedFunctionTextRanges() const {
  if (FunctionRangeCache)
    return *FunctionRangeCache;

  std::vector<FunctionTextRange> Ranges;
  uint64_t TextBegin = textAddr();
  uint64_t TextSizeValue = textSize();
  if (TextSizeValue > std::numeric_limits<uint64_t>::max() - TextBegin) {
    log() << "hotswap: error: function text range scan: .text virtual "
          << "address range overflows uint64_t.\n";
    FunctionRangeCache.emplace();
    return *FunctionRangeCache;
  }
  uint64_t TextEnd = TextBegin + TextSizeValue;

  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      consumeError(SymsOrErr.takeError());
      continue;
    }

    std::vector<const ELFT::Sym *> FuncSyms;
    for (const ELFT::Sym &Sym : *SymsOrErr) {
      if (Sym.getType() != ELF::STT_FUNC && Sym.getType() != ELF::STT_GNU_IFUNC)
        continue;
      if (Sym.st_shndx != TextSectionIndex)
        continue;
      FuncSyms.push_back(&Sym);
    }
    llvm::sort(FuncSyms, [](const ELFT::Sym *A, const ELFT::Sym *B) {
      if (A->st_value != B->st_value)
        return A->st_value < B->st_value;
      return A->st_size > B->st_size;
    });

    for (size_t I = 0, E = FuncSyms.size(); I != E; ++I) {
      const ELFT::Sym &Sym = *FuncSyms[I];
      uint64_t Begin = Sym.st_value;
      if (Begin < TextBegin || Begin >= TextEnd)
        continue;
      uint64_t End = TextEnd;
      if (Sym.st_size != 0) {
        End = Sym.st_value + Sym.st_size;
        if (End < Begin)
          End = TextEnd;
        End = std::min(End, TextEnd);
      } else {
        for (size_t J = I + 1; J != E; ++J) {
          if (FuncSyms[J]->st_value > Begin) {
            End =
                std::min(static_cast<uint64_t>(FuncSyms[J]->st_value), TextEnd);
            break;
          }
        }
      }
      Ranges.push_back({Begin, End, &Sym, &SymShdr});
    }
  }

  llvm::stable_sort(
      Ranges, [](const FunctionTextRange &LHS, const FunctionTextRange &RHS) {
        return LHS.Begin < RHS.Begin;
      });
  FunctionRangeCache = std::move(Ranges);
  return *FunctionRangeCache;
}

std::vector<ElfView::FunctionTextRange> ElfView::functionTextRanges() const {
  ArrayRef<FunctionTextRange> Ranges = cachedFunctionTextRanges();
  return std::vector<FunctionTextRange>(Ranges.begin(), Ranges.end());
}

std::optional<std::vector<uint64_t>> ElfView::textSymbolOffsets() const {
  const uint64_t TextBegin = textAddr();
  if (textSize() > std::numeric_limits<uint64_t>::max() - TextBegin) {
    log() << "hotswap: error: .text symbol scan address range overflows\n";
    return std::nullopt;
  }
  const uint64_t TextEnd = TextBegin + textSize();
  std::vector<uint64_t> Offsets;
  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      log() << "hotswap: error: failed to enumerate .text symbols: "
            << toString(SymsOrErr.takeError()) << "\n";
      return std::nullopt;
    }
    for (const ELFT::Sym &Sym : *SymsOrErr)
      if (Sym.st_shndx == TextSectionIndex && Sym.st_value >= TextBegin &&
          Sym.st_value < TextEnd)
        Offsets.push_back(Sym.st_value - TextBegin);
  }
  llvm::sort(Offsets);
  Offsets.erase(std::unique(Offsets.begin(), Offsets.end()), Offsets.end());
  return Offsets;
}

std::optional<std::vector<ElfView::TextOffsetRange>>
ElfView::textSymbolExtents() const {
  const uint64_t TextBegin = textAddr();
  if (textSize() > std::numeric_limits<uint64_t>::max() - TextBegin) {
    log() << "hotswap: error: .text symbol extent scan address range "
             "overflows\n";
    return std::nullopt;
  }
  const uint64_t TextEnd = TextBegin + textSize();
  std::vector<TextOffsetRange> Extents;
  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;
    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      log() << "hotswap: error: failed to enumerate .text symbol extents: "
            << toString(SymsOrErr.takeError()) << "\n";
      return std::nullopt;
    }
    for (const ELFT::Sym &Sym : *SymsOrErr) {
      if (Sym.st_shndx != TextSectionIndex || Sym.st_size == 0 ||
          Sym.getType() == ELF::STT_FUNC || Sym.getType() == ELF::STT_GNU_IFUNC)
        continue;
      if (Sym.st_size > std::numeric_limits<uint64_t>::max() - Sym.st_value) {
        log() << "hotswap: error: .text symbol extent overflows\n";
        return std::nullopt;
      }
      const uint64_t SymbolEnd = Sym.st_value + Sym.st_size;
      const uint64_t Begin = std::max<uint64_t>(Sym.st_value, TextBegin);
      const uint64_t End = std::min<uint64_t>(SymbolEnd, TextEnd);
      if (Begin < End)
        Extents.push_back({Begin - TextBegin, End - TextBegin});
    }
  }
  llvm::sort(
      Extents, [](const TextOffsetRange &LHS, const TextOffsetRange &RHS) {
        return std::tie(LHS.Begin, LHS.End) < std::tie(RHS.Begin, RHS.End);
      });
  std::vector<TextOffsetRange> Coalesced;
  for (const TextOffsetRange &Extent : Extents) {
    if (Coalesced.empty() || Extent.Begin > Coalesced.back().End)
      Coalesced.push_back(Extent);
    else
      Coalesced.back().End = std::max(Coalesced.back().End, Extent.End);
  }
  return Coalesced;
}

// -- ElfView::findKernelAtAddress ---------------------------------------------

const ElfView::FunctionTextRange *
ElfView::findFunctionTextRangeAtAddress(uint64_t TextAddress) const {
  ArrayRef<FunctionTextRange> Ranges = cachedFunctionTextRanges();
  ArrayRef<FunctionTextRange>::const_iterator GroupEnd =
      std::upper_bound(Ranges.begin(), Ranges.end(), TextAddress,
                       [](uint64_t Address, const FunctionTextRange &Range) {
                         return Address < Range.Begin;
                       });

  // Prefer the covering range with the greatest start address, matching the
  // previous full scan. Preserve the stable symbol-table order for duplicate
  // starts so aliases resolve exactly as before.
  while (GroupEnd != Ranges.begin()) {
    ArrayRef<FunctionTextRange>::const_iterator GroupBegin = GroupEnd - 1;
    uint64_t Begin = GroupBegin->Begin;
    while (GroupBegin != Ranges.begin() && (GroupBegin - 1)->Begin == Begin)
      --GroupBegin;
    for (ArrayRef<FunctionTextRange>::const_iterator It = GroupBegin;
         It != GroupEnd; ++It)
      if (TextAddress < It->End)
        return &*It;
    GroupEnd = GroupBegin;
  }
  return nullptr;
}

std::string ElfView::findKernelAtAddress(uint64_t TextAddress) const {
  const FunctionTextRange *Range = findFunctionTextRangeAtAddress(TextAddress);
  if (Range) {
    const ELFT::Sym &Sym = *Range->Symbol;
    Expected<StringRef> StrTabOrErr =
        File.getStringTableForSymtab(*Range->Symtab, Sections);
    if (!StrTabOrErr) {
      consumeError(StrTabOrErr.takeError());
      return "";
    }
    Expected<StringRef> NameOrErr = Sym.getName(*StrTabOrErr);
    if (!NameOrErr) {
      log() << "hotswap: error: findKernelAtAddress: function symbol "
            << "covering address 0x" << utohexstr(TextAddress)
            << " has unreadable name: " << toString(NameOrErr.takeError())
            << "\n";
      return "";
    }
    std::string BestName = NameOrErr->str();
    // Confirm the selected symbol is actually a kernel: every kernel carries a
    // "<name>.kd" descriptor symbol, whereas a plain device function does not.
    // This is the same descriptor lookup getKernelVgprCount performs, so a real
    // kernel is never rejected; a non-kernel is reported as "not found" so the
    // caller declines instead of scratch-allocating against a wrong context.
    if (const_cast<ElfView *>(this)->findKernelDescriptor(BestName)) {
      return BestName;
    }
    log() << "hotswap: findKernelAtAddress: nearest function symbol '"
          << BestName << "' preceding address 0x" << utohexstr(TextAddress)
          << " has no .kd descriptor (not a kernel); treating as no match.\n";
    return "";
  }

  log() << "hotswap: findKernelAtAddress: no function symbol covers address 0x"
        << utohexstr(TextAddress) << " in .text.\n";
  return "";
}

std::optional<ElfView::FunctionTextRange>
ElfView::findFunctionTextRangeAtOffset(uint64_t TextOffset) const {
  if (TextOffset >= textSize() ||
      TextOffset > std::numeric_limits<uint64_t>::max() - textAddr())
    return std::nullopt;
  const FunctionTextRange *Range =
      findFunctionTextRangeAtAddress(textAddr() + TextOffset);
  if (!Range || Range->Begin < textAddr() || Range->End < textAddr())
    return std::nullopt;
  return FunctionTextRange{Range->Begin - textAddr(), Range->End - textAddr(),
                           Range->Symbol, Range->Symtab};
}

// -- ElfView::kernelDescriptors -----------------------------------------------

void ElfView::initializeKernelDescriptorCache() const {
  if (KernelDescriptorCache)
    return;

  namespace hsa = amdhsa;
  std::vector<KernelDescriptorInfo> Result;
  StringMap<uint64_t> FileOffsets;
  StringMap<std::pair<uint64_t, uint64_t>> SeenLocations;
  bool Complete = true;

  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      log() << "hotswap: error: kernelDescriptors: failed to read symbols: "
            << toString(SymsOrErr.takeError()) << "\n";
      Complete = false;
      continue;
    }
    Expected<StringRef> StrTabOrErr =
        File.getStringTableForSymtab(SymShdr, Sections);
    if (!StrTabOrErr) {
      log() << "hotswap: error: kernelDescriptors: failed to read symbol "
            << "string table: " << toString(StrTabOrErr.takeError()) << "\n";
      Complete = false;
      continue;
    }

    for (const ELFT::Sym &Sym : *SymsOrErr) {
      Expected<StringRef> NameOrErr = Sym.getName(*StrTabOrErr);
      if (!NameOrErr) {
        log() << "hotswap: error: kernelDescriptors: failed to read symbol "
              << "name: " << toString(NameOrErr.takeError()) << "\n";
        Complete = false;
        continue;
      }
      if (!NameOrErr->ends_with(".kd"))
        continue;

      Expected<const ELFT::Shdr *> HostShdrOrErr =
          File.getSection(Sym.st_shndx);
      if (!HostShdrOrErr) {
        log() << "hotswap: error: kernelDescriptors: descriptor symbol '"
              << *NameOrErr << "' has unreadable section index " << Sym.st_shndx
              << ": " << toString(HostShdrOrErr.takeError()) << "\n";
        Complete = false;
        continue;
      }
      const ELFT::Shdr &HostShdr = **HostShdrOrErr;
      std::optional<uint64_t> FileOffset = checkedSectionFileOffset(
          HostShdr, Sym.st_value, KdSize, size(),
          (Twine("kernelDescriptors: descriptor symbol '") + *NameOrErr + "'")
              .str());
      if (!FileOffset) {
        Complete = false;
        continue;
      }

      int64_t EntryOffset = 0;
      std::memcpy(
          &EntryOffset,
          data() + *FileOffset +
              offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
          sizeof(EntryOffset));

      StringRef KernelNameRef = NameOrErr->drop_back(3);
      auto Seen = SeenLocations.try_emplace(
          KernelNameRef, std::make_pair(Sym.st_value, *FileOffset));
      if (Seen.second) {
        Result.push_back(
            {KernelNameRef.str(), Sym.st_value, *FileOffset, EntryOffset});
        FileOffsets.try_emplace(KernelNameRef, *FileOffset);
      } else if (Seen.first->second.first != Sym.st_value ||
                 Seen.first->second.second != *FileOffset) {
        log() << "hotswap: error: kernelDescriptors: descriptor name '"
              << KernelNameRef << "' resolves to multiple locations\n";
        Complete = false;
      }
    }
  }

  KernelDescriptorFileOffsetCache = std::move(FileOffsets);
  KernelDescriptorCacheComplete = Complete;
  KernelDescriptorCache = std::move(Result);

  // Name -> vaddr map keeps per-fixup lookup O(1). Ambiguous duplicate names
  // are rejected above, so every retained name has exactly one location.
  KernelDescriptorVAddrCache.clear();
  for (const KernelDescriptorInfo &Info : *KernelDescriptorCache)
    KernelDescriptorVAddrCache.try_emplace(Info.KernelName, Info.VAddr);
}

uint8_t *ElfView::findKernelDescriptor(StringRef KernelName) {
  initializeKernelDescriptorCache();
  StringMap<uint64_t>::const_iterator It =
      KernelDescriptorFileOffsetCache.find(KernelName);
  if (It == KernelDescriptorFileOffsetCache.end())
    return nullptr;
  return data() + It->second;
}

ArrayRef<KernelDescriptorInfo> ElfView::kernelDescriptors() const {
  initializeKernelDescriptorCache();
  return *KernelDescriptorCache;
}

bool ElfView::kernelDescriptorCacheIsComplete() const {
  initializeKernelDescriptorCache();
  return KernelDescriptorCacheComplete;
}

std::optional<uint64_t>
ElfView::getKernelDescriptorVAddr(StringRef KernelName) const {
  initializeKernelDescriptorCache();
  StringMap<uint64_t>::const_iterator It =
      KernelDescriptorVAddrCache.find(KernelName);
  if (It == KernelDescriptorVAddrCache.end())
    return std::nullopt;
  return It->second;
}

bool ElfView::updateKernelDescriptorEntryOffset(StringRef KernelName,
                                                int64_t NewEntryOffset) {
  namespace hsa = amdhsa;
  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptorEntryOffset: kernel "
          << "descriptor symbol '" << KernelName << ".kd' not found.\n";
    return false;
  }
  std::memcpy(
      Kd + offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      &NewEntryOffset, sizeof(NewEntryOffset));
  for (KernelDescriptorInfo &Info : *KernelDescriptorCache) {
    if (Info.KernelName != KernelName)
      continue;
    Info.EntryOffset = NewEntryOffset;
    break;
  }
  return true;
}

bool ElfView::updateKernelDescriptorSgprCount(StringRef KernelName,
                                              unsigned RequiredSgprs,
                                              bool UpdateDescriptor) {
  namespace hsa = amdhsa;
  if (RequiredSgprs == 0)
    return true;

  uint8_t *Kd = nullptr;
  uint32_t Rsrc1 = 0;
  std::optional<uint32_t> RequiredGranulated;
  if (UpdateDescriptor) {
    Kd = findKernelDescriptor(KernelName);
    if (!Kd) {
      log() << "hotswap: error: updateKernelDescriptorSgprCount: kernel "
            << "descriptor symbol '" << KernelName << ".kd' not found.\n";
      return false;
    }

    std::memcpy(&Rsrc1,
                Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
                sizeof(Rsrc1));

    uint32_t CurrentGranulated = AMDHSA_BITS_GET(
        Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
    uint64_t CurrentSgprs =
        (static_cast<uint64_t>(CurrentGranulated) + 1) * SgprEncodingGranule;

    if (RequiredSgprs > CurrentSgprs) {
      uint64_t RequiredGranulated64 =
          (static_cast<uint64_t>(RequiredSgprs) + SgprEncodingGranule - 1) /
              SgprEncodingGranule -
          1;
      uint32_t MaxGranulated = static_cast<uint32_t>(
          hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT >>
          hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT);
      if (RequiredGranulated64 > MaxGranulated) {
        log() << "hotswap: error: updateKernelDescriptorSgprCount: kernel '"
              << KernelName << "' needs " << RequiredSgprs
              << " SGPRs, which exceeds the descriptor encoding limit.\n";
        return false;
      }
      RequiredGranulated = static_cast<uint32_t>(RequiredGranulated64);
    }
  }

  StringMap<unsigned> RequiredSgprCounts;
  RequiredSgprCounts.try_emplace(KernelName, RequiredSgprs);
  MetadataCountUpdateStatus MetadataStatus = rewriteKernelMetadataCounts(
      data(), File, RequiredSgprCounts, ".sgpr_count",
      "updateKernelDescriptorSgprCount");
  if (MetadataStatus == MetadataCountUpdateStatus::Error)
    return false;
  if (!UpdateDescriptor &&
      MetadataStatus == MetadataCountUpdateStatus::NotFound) {
    log() << "hotswap: error: updateKernelDescriptorSgprCount: kernel '"
          << KernelName << "' requires " << RequiredSgprs
          << " SGPRs, but gfx10+ code objects must carry .sgpr_count metadata "
             "because the descriptor SGPR-count field is reserved.\n";
    return false;
  }
  // On pre-gfx10 targets, NotFound is allowed for minimal code objects without
  // AMDGPU metadata because the descriptor remains the canonical count.

  if (SgprCacheState == KernelSgprCacheState::Metadata &&
      MetadataStatus == MetadataCountUpdateStatus::Found) {
    StringMap<std::optional<unsigned>>::iterator Cached =
        KernelSgprCountCache.find(KernelName);
    if (Cached == KernelSgprCountCache.end())
      KernelSgprCountCache.try_emplace(KernelName, RequiredSgprs);
    else if (!Cached->second || RequiredSgprs > *Cached->second)
      Cached->second = RequiredSgprs;
  }

  if (!RequiredGranulated)
    return true;

  AMDHSA_BITS_SET(Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  *RequiredGranulated);
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Rsrc1, sizeof(Rsrc1));
  if (SgprCacheState == KernelSgprCacheState::NoMetadata)
    KernelSgprCountCache.erase(KernelName);
  return true;
}

bool ElfView::updateKernelMetadataSgprCounts(
    const StringMap<unsigned> &RequiredSgprs) {
  MetadataCountUpdateStatus MetadataStatus =
      rewriteKernelMetadataCounts(data(), File, RequiredSgprs, ".sgpr_count",
                                  "updateKernelMetadataSgprCounts");
  if (MetadataStatus == MetadataCountUpdateStatus::Error)
    return false;
  if (MetadataStatus == MetadataCountUpdateStatus::NotFound) {
    log() << "hotswap: error: updateKernelMetadataSgprCounts: code object "
             "has no AMDGPU metadata note.\n";
    return false;
  }

  if (SgprCacheState == KernelSgprCacheState::Metadata) {
    for (const StringMapEntry<unsigned> &Required : RequiredSgprs) {
      StringMap<std::optional<unsigned>>::iterator Cached =
          KernelSgprCountCache.find(Required.first());
      if (Cached == KernelSgprCountCache.end())
        KernelSgprCountCache.try_emplace(Required.first(), Required.second);
      else if (!Cached->second || Required.second > *Cached->second)
        Cached->second = Required.second;
    }
  }
  return true;
}

bool ElfView::updateKernelMetadataVgprCounts(
    const StringMap<unsigned> &RequiredVgprs) {
  MetadataCountUpdateStatus MetadataStatus =
      rewriteKernelMetadataCounts(data(), File, RequiredVgprs, ".vgpr_count",
                                  "updateKernelMetadataVgprCounts");
  if (MetadataStatus == MetadataCountUpdateStatus::Error)
    return false;
  if (MetadataStatus == MetadataCountUpdateStatus::NotFound) {
    log() << "hotswap: error: updateKernelMetadataVgprCounts: code object "
             "has no AMDGPU metadata note.\n";
    return false;
  }
  return true;
}

std::optional<uint32_t>
ElfView::getKernelDescriptorInstPrefSize(StringRef KernelName,
                                         StringRef TargetCpu) const {
  namespace hsa = amdhsa;
  uint8_t *Kd = const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: getKernelDescriptorInstPrefSize: kernel "
          << "descriptor symbol '" << KernelName << ".kd' not found.\n";
    return std::nullopt;
  }

  uint32_t Rsrc3 = 0;
  std::memcpy(&Rsrc3,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              sizeof(Rsrc3));

  if (TargetCpu.starts_with("gfx12")) {
    return AMDHSA_BITS_GET(Rsrc3,
                           hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE);
  }

  log() << "hotswap: error: getKernelDescriptorInstPrefSize: unsupported "
        << "target CPU '" << TargetCpu << "' for kernel '" << KernelName
        << "'.\n";
  return std::nullopt;
}

bool ElfView::updateKernelDescriptorInstPrefSize(StringRef KernelName,
                                                 StringRef TargetCpu,
                                                 uint32_t InstPrefLines) {
  namespace hsa = amdhsa;
  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptorInstPrefSize: kernel "
          << "descriptor symbol '" << KernelName << ".kd' not found.\n";
    return false;
  }

  if (!TargetCpu.starts_with("gfx12")) {
    log() << "hotswap: error: updateKernelDescriptorInstPrefSize: unsupported "
          << "target CPU '" << TargetCpu << "' for kernel '" << KernelName
          << "'.\n";
    return false;
  }

  uint32_t MaxInstPrefLines = static_cast<uint32_t>(
      hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE >>
      hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE_SHIFT);
  if (InstPrefLines > MaxInstPrefLines) {
    log() << "hotswap: error: updateKernelDescriptorInstPrefSize: value "
          << InstPrefLines << " exceeds the gfx12 descriptor encoding limit.\n";
    return false;
  }

  uint32_t Rsrc3 = 0;
  std::memcpy(&Rsrc3,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              sizeof(Rsrc3));
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE,
                  InstPrefLines);
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              &Rsrc3, sizeof(Rsrc3));
  return true;
}

// -- ElfView::getKernelVgprCount ----------------------------------------------

std::optional<unsigned>
ElfView::getKernelVgprCount(StringRef KernelName,
                            unsigned VgprGranuleSize) const {
  if (VgprGranuleSize == 0) {
    log() << "hotswap: error: getKernelVgprCount: VgprGranuleSize is 0 for "
          << "kernel '" << KernelName << "'.\n";
    return std::nullopt;
  }
  namespace hsa = amdhsa;
  // findKernelDescriptor never writes through the returned pointer in this
  // call path but is shared (non-const) with descriptor update helpers. The
  // const_cast on `this` keeps the read-only accessor const-correct without
  // duplicating the lookup helper.
  uint8_t *Kd = const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: getKernelVgprCount: kernel descriptor symbol '"
          << KernelName << ".kd' not found.\n";
    return std::nullopt;
  }
  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  uint32_t Granulated = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint64_t VgprCount =
      (static_cast<uint64_t>(Granulated) + 1) * VgprGranuleSize;
  if (VgprCount > std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: getKernelVgprCount: descriptor VGPR count for '"
          << KernelName << "' exceeds unsigned.\n";
    return std::nullopt;
  }
  return static_cast<unsigned>(VgprCount);
}

static std::optional<unsigned> getKernelUnsignedMetadata(const ELFFileT &File,
                                                         StringRef KernelName,
                                                         StringRef Key,
                                                         StringRef Context) {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: " << Context
          << ": failed to read program headers: "
          << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }

  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata note has an empty descriptor.\n";
        return std::nullopt;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: " << Context
              << ": failed to parse AMDGPU metadata note.\n";
        return std::nullopt;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata root is not a map.\n";
        return std::nullopt;
      }

      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator KernelsIt =
          RootMap.find("amdhsa.kernels");
      if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
        continue;

      msgpack::ArrayDocNode &KernelArray = KernelsIt->second.getArray();
      for (msgpack::DocNode &KernelNode : KernelArray) {
        if (!KernelNode.isMap())
          continue;
        msgpack::MapDocNode &KernelMap = KernelNode.getMap();
        msgpack::DocNode::MapTy::iterator NameIt = KernelMap.find(".name");
        if (NameIt == KernelMap.end() || !NameIt->second.isString() ||
            NameIt->second.getString() != KernelName)
          continue;

        msgpack::DocNode::MapTy::iterator ValueIt = KernelMap.find(Key);
        if (ValueIt == KernelMap.end())
          return std::nullopt;
        return readUnsignedMetadataNode(ValueIt->second, KernelName, Key,
                                        Context);
      }
    }

    if (Err) {
      log() << "hotswap: error: " << Context
            << ": failed to iterate AMDGPU notes: " << toString(std::move(Err))
            << "\n";
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<unsigned>
ElfView::getKernelMaxFlatWorkgroupSize(StringRef KernelName) const {
  return getKernelUnsignedMetadata(File, KernelName, ".max_flat_workgroup_size",
                                   "getKernelMaxFlatWorkgroupSize");
}

std::optional<unsigned>
ElfView::getKernelMetadataVgprCount(StringRef KernelName) const {
  return getKernelUnsignedMetadata(File, KernelName, ".vgpr_count",
                                   "getKernelMetadataVgprCount");
}

std::optional<unsigned>
ElfView::getKernelWavefrontSize(StringRef KernelName) const {
  return getKernelUnsignedMetadata(File, KernelName, ".wavefront_size",
                                   "getKernelWavefrontSize");
}

// Reads the static (compile-time-fixed) LDS allocation from the kernel
// descriptor's group_segment_fixed_size field. Dynamic LDS is added by the
// host at dispatch time and is not visible here -- see the declaration's
// doc comment for the full lower-bound caveat.

std::optional<uint32_t>
ElfView::getKernelStaticLdsSize(StringRef KernelName) const {
  namespace hsa = amdhsa;
  // findKernelDescriptor never writes through the returned pointer in this
  // call path but is shared (non-const) with descriptor update helpers. The
  // const_cast on `this` keeps the read-only accessor const-correct without
  // duplicating the lookup helper.
  const uint8_t *Kd =
      const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: getKernelStaticLdsSize: kernel descriptor "
          << "symbol '" << KernelName << ".kd' not found.\n";
    return std::nullopt;
  }
  uint32_t LdsSize;
  std::memcpy(&LdsSize,
              Kd + offsetof(hsa::kernel_descriptor_t, group_segment_fixed_size),
              sizeof(LdsSize));
  return LdsSize;
}

// -- ElfView::getKernelSgprCount ----------------------------------------------
//
// Reads .sgpr_count from the amdhsa.kernels msgpack metadata note.
// On GFX10+ GRANULATED_WAVEFRONT_SGPR_COUNT in the kernel descriptor is
// architecturally reserved (must be zero), so the metadata note is the
// preferred source. Falls back to the KD field when no metadata note is
// present (e.g. minimal test ELFs assembled with -nostdlib).

void ElfView::initializeKernelSgprCountCache() const {
  if (SgprCacheState != KernelSgprCacheState::Uninitialized)
    return;

  // Default to Error so every malformed-note early return leaves an explicit
  // terminal cache state instead of reparsing the same large blob.
  SgprCacheState = KernelSgprCacheState::Error;
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  bool SawMetadataNote = false;
  if (PhdrsOrErr) {
    for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
      if (Phdr.p_type != ELF::PT_NOTE)
        continue;
      Error Err = Error::success();
      for (ELFT::Note Note : File.notes(Phdr, Err)) {
        if (Note.getName() != "AMDGPU" ||
            Note.getType() != ELF::NT_AMDGPU_METADATA)
          continue;
        SawMetadataNote = true;

        ArrayRef<uint8_t> Desc = Note.getDesc(4);
        if (Desc.empty()) {
          log() << "hotswap: error: SGPR cache: AMDGPU metadata note "
                << "has an empty descriptor.\n";
          return;
        }

        StringRef Blob(reinterpret_cast<const char *>(Desc.data()),
                       Desc.size());
        msgpack::Document Doc;
        if (!Doc.readFromBlob(Blob, false)) {
          log() << "hotswap: error: SGPR cache: failed to parse "
                << "AMDGPU metadata note.\n";
          return;
        }

        msgpack::DocNode Root = Doc.getRoot();
        if (!Root.isMap()) {
          log() << "hotswap: error: SGPR cache: AMDGPU metadata root "
                << "is not a map.\n";
          return;
        }
        msgpack::MapDocNode &RootMap = Root.getMap();
        msgpack::DocNode::MapTy::iterator KernelsIt =
            RootMap.find("amdhsa.kernels");
        if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
          continue;

        msgpack::ArrayDocNode &KernelArray = KernelsIt->second.getArray();
        for (msgpack::DocNode &KNode : KernelArray) {
          if (!KNode.isMap())
            continue;
          msgpack::MapDocNode &KMap = KNode.getMap();
          msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
          if (NameIt == KMap.end() || !NameIt->second.isString() ||
              KernelSgprCountCache.find(NameIt->second.getString()) !=
                  KernelSgprCountCache.end())
            continue;

          msgpack::DocNode::MapTy::iterator SgprIt = KMap.find(".sgpr_count");
          if (SgprIt == KMap.end()) {
            KernelSgprCountCache.try_emplace(NameIt->second.getString(),
                                             std::nullopt);
            continue;
          }
          StringRef Name = NameIt->second.getString();
          KernelSgprCountCache.try_emplace(
              Name, readUnsignedMetadataNode(SgprIt->second, Name,
                                             ".sgpr_count", "SGPR cache"));
        }
      }
      if (Err) {
        log() << "hotswap: error: SGPR cache: failed to iterate "
              << "AMDGPU notes: " << toString(std::move(Err)) << "\n";
        return;
      }
    }
  } else {
    log() << "hotswap: error: SGPR cache: failed to read program "
          << "headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return;
  }

  SgprCacheState = SawMetadataNote ? KernelSgprCacheState::Metadata
                                   : KernelSgprCacheState::NoMetadata;
}

std::optional<unsigned>
ElfView::getKernelSgprCount(StringRef KernelName) const {
  initializeKernelSgprCountCache();
  if (SgprCacheState == KernelSgprCacheState::Error)
    return std::nullopt;

  StringMap<std::optional<unsigned>>::const_iterator Cached =
      KernelSgprCountCache.find(KernelName);
  if (SgprCacheState == KernelSgprCacheState::Metadata) {
    if (Cached != KernelSgprCountCache.end()) {
      if (!Cached->second)
        log() << "hotswap: error: getKernelSgprCount: metadata for kernel '"
              << KernelName << "' has no valid .sgpr_count.\n";
      return Cached->second;
    }
    log() << "hotswap: error: getKernelSgprCount: AMDGPU metadata has no "
          << ".sgpr_count entry for kernel '" << KernelName << "'.\n";
    return std::nullopt;
  }

  if (Cached != KernelSgprCountCache.end())
    return Cached->second;

  // --- Fallback: read the KD field. ---
  // The LLVM assembler populates GRANULATED_WAVEFRONT_SGPR_COUNT even on
  // GFX10+ where the hardware ignores it, so this is still usable for
  // ROCm-compiled code objects that lack a metadata note.
  namespace hsa = amdhsa;
  uint8_t *Kd = const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    KernelSgprCountCache.try_emplace(KernelName, std::nullopt);
    return std::nullopt;
  }
  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  uint32_t Granulated = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  uint64_t SgprCount =
      (static_cast<uint64_t>(Granulated) + 1) * SgprEncodingGranule;
  if (SgprCount > std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: getKernelSgprCount: descriptor SGPR count for '"
          << KernelName << "' exceeds unsigned.\n";
    return std::nullopt;
  }
  std::optional<unsigned> Result = static_cast<unsigned>(SgprCount);
  KernelSgprCountCache.try_emplace(KernelName, Result);
  return Result;
}

// -- ElfView::getKernelClusterDims --------------------------------------------
//
// Reads optional fixed .cluster_dims metadata from the amdhsa.kernels msgpack
// note. Absence is expected for kernels with variable dispatch-time cluster
// dimensions, so callers use std::nullopt as the dynamic fallback signal.

std::optional<KernelClusterDims>
ElfView::getKernelClusterDims(StringRef KernelName) const {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: getKernelClusterDims: failed to read program "
          << "headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }

  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: getKernelClusterDims: AMDGPU metadata note "
              << "has an empty descriptor.\n";
        return std::nullopt;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: getKernelClusterDims: failed to parse "
              << "AMDGPU metadata note.\n";
        return std::nullopt;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: getKernelClusterDims: AMDGPU metadata root "
              << "is not a map.\n";
        return std::nullopt;
      }

      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator KernelsIt =
          RootMap.find("amdhsa.kernels");
      if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
        continue;

      msgpack::ArrayDocNode &KernelArray = KernelsIt->second.getArray();
      for (msgpack::DocNode &KNode : KernelArray) {
        if (!KNode.isMap())
          continue;

        msgpack::MapDocNode &KMap = KNode.getMap();
        msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
        if (NameIt == KMap.end() || !NameIt->second.isString() ||
            NameIt->second.getString() != KernelName)
          continue;

        msgpack::DocNode::MapTy::iterator DimsIt = KMap.find(".cluster_dims");
        if (DimsIt == KMap.end())
          return std::nullopt;
        if (!DimsIt->second.isArray()) {
          log() << "hotswap: error: getKernelClusterDims: .cluster_dims for '"
                << KernelName << "' is not an array.\n";
          return std::nullopt;
        }

        msgpack::ArrayDocNode &Dims = DimsIt->second.getArray();
        if (Dims.size() != 3) {
          log() << "hotswap: error: getKernelClusterDims: .cluster_dims for '"
                << KernelName << "' has " << Dims.size()
                << " entries, expected 3.\n";
          return std::nullopt;
        }

        std::optional<unsigned> X = readUnsignedMetadataNode(
            Dims[0], KernelName, ".cluster_dims[0]", "getKernelClusterDims");
        std::optional<unsigned> Y = readUnsignedMetadataNode(
            Dims[1], KernelName, ".cluster_dims[1]", "getKernelClusterDims");
        std::optional<unsigned> Z = readUnsignedMetadataNode(
            Dims[2], KernelName, ".cluster_dims[2]", "getKernelClusterDims");
        if (!X || !Y || !Z)
          return std::nullopt;
        return KernelClusterDims{*X, *Y, *Z};
      }
    }

    if (Err) {
      log() << "hotswap: error: getKernelClusterDims: failed to iterate "
            << "AMDGPU notes: " << toString(std::move(Err)) << "\n";
      return std::nullopt;
    }
  }

  return std::nullopt;
}

// -- ElfView::updateKernelDescriptorVgprCount ---------------------------------

bool ElfView::updateKernelDescriptorVgprCount(StringRef KernelName,
                                              unsigned RequiredVgprs,
                                              unsigned VgprGranuleSize) {
  namespace hsa = amdhsa;
  if (VgprGranuleSize == 0) {
    log() << "hotswap: error: updateKernelDescriptorVgprCount: VGPR granule "
             "is zero for kernel '"
          << KernelName << "'.\n";
    return false;
  }

  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptorVgprCount: kernel "
             "descriptor symbol '"
          << KernelName << ".kd' not found.\n";
    return false;
  }

  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  uint32_t CurrentGranulated = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint64_t CurrentVgprs =
      (static_cast<uint64_t>(CurrentGranulated) + 1) * VgprGranuleSize;
  if (RequiredVgprs <= CurrentVgprs)
    return true;

  uint32_t MaxGran = static_cast<uint32_t>(
      hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT >>
      hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT);
  uint64_t NewGranulated =
      (static_cast<uint64_t>(RequiredVgprs) + VgprGranuleSize - 1) /
          VgprGranuleSize -
      1;
  if (NewGranulated > MaxGran) {
    log() << "hotswap: error: updateKernelDescriptorVgprCount: kernel '"
          << KernelName << "' needs " << RequiredVgprs
          << " VGPRs, which exceeds the descriptor encoding limit.\n";
    return false;
  }
  AMDHSA_BITS_SET(Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  static_cast<uint32_t>(NewGranulated));
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Rsrc1, sizeof(Rsrc1));
  return true;
}

// -- ElfView::dataAtVAddr -----------------------------------------------------

const uint8_t *ElfView::dataAtVAddr(uint64_t VAddr, uint64_t Len) const {
  for (const ELFT::Shdr &Shdr : Sections) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) || Shdr.sh_type == ELF::SHT_NOBITS)
      continue;
    if (VAddr < Shdr.sh_addr)
      continue;
    uint64_t Off = VAddr - Shdr.sh_addr;
    if (Off > Shdr.sh_size || Len > Shdr.sh_size - Off)
      continue;
    if (Shdr.sh_offset > size() || Off > size() - Shdr.sh_offset ||
        Len > size() - Shdr.sh_offset - Off)
      continue;
    return data() + Shdr.sh_offset + Off;
  }
  return nullptr;
}

bool ElfView::isExecutableVAddrRange(uint64_t VAddr, uint64_t Len) const {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    consumeError(PhdrsOrErr.takeError());
    return false;
  }

  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_LOAD || !(Phdr.p_flags & ELF::PF_X) ||
        VAddr < Phdr.p_vaddr)
      continue;
    uint64_t Offset = VAddr - Phdr.p_vaddr;
    if (Offset <= Phdr.p_filesz && Len <= Phdr.p_filesz - Offset &&
        Phdr.p_offset <= size() && Offset <= size() - Phdr.p_offset &&
        Len <= size() - Phdr.p_offset - Offset)
      return true;
  }
  return false;
}

std::optional<bool> ElfView::executableCodeOutsideTextIsCompatibleWith(
    ExecutablePoolTargetState TargetState) const {
  struct ExecutableRegion {
    uint64_t VAddr;
    uint64_t Size;
    uint64_t FileOffset;
  };
  struct PoolProvenance {
    uint64_t VAddr;
    uint64_t Size;
    ExecutablePoolTargetState State;
  };

  if (TargetState != ExecutablePoolTargetState::A0 &&
      TargetState != ExecutablePoolTargetState::B0) {
    log() << "hotswap: error: executable pool compatibility requires an "
             "explicit A0 or B0 target state.\n";
    return std::nullopt;
  }

  const uint64_t FileSize = size();
  if (textOffset() > FileSize || textSize() > FileSize - textOffset()) {
    log() << "hotswap: error: .text extends past the end of the ELF buffer.\n";
    return std::nullopt;
  }
  if (textSize() > std::numeric_limits<uint64_t>::max() - textAddr()) {
    log() << "hotswap: error: .text virtual-address range overflows.\n";
    return std::nullopt;
  }
  const uint64_t TextAddrEnd = textAddr() + textSize();
  const uint64_t TextFileEnd = textOffset() + textSize();
  if (TextSectionIndex >= Sections.size()) {
    log() << "hotswap: error: cached .text section index is out of range.\n";
    return std::nullopt;
  }
  const ELFT::Shdr &TextShdr = Sections[TextSectionIndex];
  if (TextShdr.sh_type != ELF::SHT_PROGBITS ||
      !(TextShdr.sh_flags & ELF::SHF_ALLOC) ||
      !(TextShdr.sh_flags & ELF::SHF_EXECINSTR) ||
      (TextShdr.sh_flags & ELF::SHF_WRITE)) {
    log() << "hotswap: error: .text is not a non-writable allocatable "
             "SHT_PROGBITS executable section.\n";
    return std::nullopt;
  }

  SmallVector<ExecutableRegion, 4> ExternalSections;
  SmallVector<ExecutableRegion, 4> ExternalSegments;

  unsigned SectionIndex = 0;
  for (const ELFT::Shdr &Shdr : Sections) {
    const bool IsFileBackedExecutable =
        (Shdr.sh_flags & ELF::SHF_EXECINSTR) &&
        Shdr.sh_type != ELF::SHT_NOBITS && Shdr.sh_size != 0;
    if (!IsFileBackedExecutable) {
      ++SectionIndex;
      continue;
    }
    if (Shdr.sh_offset > FileSize ||
        Shdr.sh_size > FileSize - Shdr.sh_offset) {
      log() << "hotswap: error: executable section " << SectionIndex
            << " extends past the end of the ELF buffer.\n";
      return std::nullopt;
    }
    if (Shdr.sh_size > std::numeric_limits<uint64_t>::max() - Shdr.sh_addr) {
      log() << "hotswap: error: executable section " << SectionIndex
            << " virtual-address range overflows.\n";
      return std::nullopt;
    }
    if (SectionIndex != TextSectionIndex) {
      if (Shdr.sh_type != ELF::SHT_PROGBITS ||
          !(Shdr.sh_flags & ELF::SHF_ALLOC) ||
          (Shdr.sh_flags & ELF::SHF_WRITE)) {
        log() << "hotswap: error: external executable section " << SectionIndex
              << " is not a non-writable allocatable SHT_PROGBITS pool.\n";
        return false;
      }
      ExternalSections.push_back(
          {Shdr.sh_addr, Shdr.sh_size, Shdr.sh_offset});
    }
    ++SectionIndex;
  }

  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: failed to enumerate program headers while "
             "checking executable coverage: "
          << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }
  unsigned TextSegmentCount = 0;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_LOAD || !(Phdr.p_flags & ELF::PF_X))
      continue;
    if (Phdr.p_offset > FileSize ||
        Phdr.p_filesz > FileSize - Phdr.p_offset) {
      log() << "hotswap: error: executable PT_LOAD extends past the end of "
               "the ELF buffer.\n";
      return std::nullopt;
    }
    if (Phdr.p_filesz == 0)
      continue;
    if (Phdr.p_filesz != Phdr.p_memsz ||
        Phdr.p_filesz > std::numeric_limits<uint64_t>::max() - Phdr.p_vaddr) {
      log() << "hotswap: error: executable PT_LOAD has unequal file/memory "
               "sizes or an invalid virtual-address range.\n";
      return std::nullopt;
    }
    const uint64_t SegmentAddrEnd = Phdr.p_vaddr + Phdr.p_filesz;
    const uint64_t SegmentFileEnd = Phdr.p_offset + Phdr.p_filesz;
    const bool ContainsText =
        Phdr.p_vaddr <= textAddr() && TextAddrEnd <= SegmentAddrEnd &&
        Phdr.p_offset <= textOffset() && TextFileEnd <= SegmentFileEnd &&
        textAddr() - Phdr.p_vaddr == textOffset() - Phdr.p_offset;
    if (!(Phdr.p_flags & ELF::PF_R) || (Phdr.p_flags & ELF::PF_W)) {
      log() << "hotswap: error: executable PT_LOAD is not read-only "
               "executable.\n";
      return false;
    }
    if (ContainsText) {
      if (Phdr.p_vaddr != textAddr() || Phdr.p_offset != textOffset() ||
          Phdr.p_filesz != textSize()) {
        log() << "hotswap: error: executable PT_LOAD containing .text also "
                 "covers bytes outside the rewritten section.\n";
        return false;
      }
      ++TextSegmentCount;
      continue;
    }
    ExternalSegments.push_back(
        {Phdr.p_vaddr, Phdr.p_filesz, Phdr.p_offset});
  }
  if (TextSegmentCount != 1) {
    log() << "hotswap: error: .text must have exactly one exact RX PT_LOAD "
             "mapping; found "
          << TextSegmentCount << ".\n";
    return false;
  }

  auto RangesOverlap = [](uint64_t AStart, uint64_t ASize,
                          uint64_t BStart, uint64_t BSize) {
    // All range ends were checked above before regions were collected.
    return ASize != 0 && BSize != 0 && AStart < BStart + BSize &&
           BStart < AStart + ASize;
  };
  auto ValidateDisjointExecutableRegions =
      [&](ArrayRef<ExecutableRegion> Regions, StringRef Kind) {
        for (size_t I = 0; I != Regions.size(); ++I) {
          const ExecutableRegion &Region = Regions[I];
          if (RangesOverlap(Region.VAddr, Region.Size, textAddr(), textSize()) ||
              RangesOverlap(Region.FileOffset, Region.Size, textOffset(),
                            textSize())) {
            log() << "hotswap: error: external executable " << Kind << " " << I
                  << " overlaps or aliases .text.\n";
            return false;
          }
          for (size_t J = 0; J != I; ++J) {
            const ExecutableRegion &Other = Regions[J];
            if (RangesOverlap(Region.VAddr, Region.Size, Other.VAddr,
                              Other.Size) ||
                RangesOverlap(Region.FileOffset, Region.Size,
                              Other.FileOffset, Other.Size)) {
              log() << "hotswap: error: external executable " << Kind << " "
                    << I << " overlaps or aliases " << Kind << " " << J
                    << ".\n";
              return false;
            }
          }
        }
        return true;
      };
  if (!ValidateDisjointExecutableRegions(ExternalSections, "section") ||
      !ValidateDisjointExecutableRegions(ExternalSegments, "PT_LOAD"))
    return false;

  SmallVector<PoolProvenance, 4> Provenance;
  for (const ELFT::Shdr &Shdr : Sections) {
    if (Shdr.sh_type != ELF::SHT_NOTE)
      continue;
    if (Shdr.sh_offset > FileSize ||
        Shdr.sh_size > FileSize - Shdr.sh_offset) {
      log() << "hotswap: error: SHT_NOTE section extends past the end of the "
               "ELF buffer.\n";
      return std::nullopt;
    }
    if (Shdr.sh_addralign != 0 && Shdr.sh_addralign != 1 &&
        Shdr.sh_addralign != 4 && Shdr.sh_addralign != 8) {
      log() << "hotswap: error: SHT_NOTE section has unsupported alignment "
            << Shdr.sh_addralign << ".\n";
      return std::nullopt;
    }
    const uint64_t RecordAlign = std::max<uint64_t>(Shdr.sh_addralign, 4);

    ArrayRef<uint8_t> NoteBytes(data() + Shdr.sh_offset,
                                static_cast<size_t>(Shdr.sh_size));
    uint64_t Cursor = 0;
    while (Cursor != NoteBytes.size()) {
      if (Cursor > NoteBytes.size() || NoteBytes.size() - Cursor < 12) {
        log() << "hotswap: error: malformed SHT_NOTE record header.\n";
        return std::nullopt;
      }
      const uint8_t *Header = NoteBytes.data() + Cursor;
      const uint32_t NameSize = support::endian::read32le(Header);
      const uint32_t DescSize = support::endian::read32le(Header + 4);
      const uint32_t Type = support::endian::read32le(Header + 8);
      const uint64_t NameStorageSize =
          alignTo(uint64_t{NameSize}, RecordAlign);
      const uint64_t DescStorageSize =
          alignTo(uint64_t{DescSize}, RecordAlign);
      const uint64_t BytesAfterHeader = NoteBytes.size() - Cursor - 12;
      if (NameStorageSize > BytesAfterHeader ||
          DescStorageSize > BytesAfterHeader - NameStorageSize) {
        log() << "hotswap: error: malformed SHT_NOTE record payload.\n";
        return std::nullopt;
      }
      const uint64_t NameOffset = Cursor + 12;
      const uint64_t DescOffset = NameOffset + NameStorageSize;
      const uint64_t RecordEnd = DescOffset + DescStorageSize;
      ArrayRef<uint8_t> NameBytes =
          NoteBytes.slice(static_cast<size_t>(NameOffset), NameSize);
      const bool HasHotswapVendorSpelling =
          NameBytes.size() >= HotswapPoolNoteName.size() &&
          std::memcmp(NameBytes.data(), HotswapPoolNoteName.data(),
                      HotswapPoolNoteName.size()) == 0;

      if (Type == HotswapPoolNoteType && HasHotswapVendorSpelling) {
        const bool HasExactTerminatedName =
            NameBytes.size() == HotswapPoolNoteName.size() + 1 &&
            NameBytes.back() == 0;
        if (!HasExactTerminatedName || Shdr.sh_addralign != 4 ||
            Shdr.sh_entsize != 0 ||
            (Shdr.sh_flags & (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR)) ||
            Cursor != 0 || RecordEnd != NoteBytes.size()) {
          log() << "hotswap: error: HotSwap pool provenance does not use the "
                   "canonical non-allocating ELF-note layout.\n";
          return std::nullopt;
        }
        if (DescSize != HotswapPoolNoteDescSize) {
          log() << "hotswap: error: HotSwap pool provenance has descriptor "
                   "size "
                << DescSize << ", expected " << HotswapPoolNoteDescSize
                << ".\n";
          return std::nullopt;
        }
        const uint8_t *Desc = NoteBytes.data() + DescOffset;
        const uint32_t Version = support::endian::read32le(Desc);
        const uint32_t RawState = support::endian::read32le(Desc + 4);
        const uint64_t VAddr = support::endian::read64le(Desc + 8);
        const uint64_t RegionSize = support::endian::read64le(Desc + 16);
        if (Version != HotswapPoolNoteVersion ||
            RawState >
                static_cast<uint32_t>(ExecutablePoolTargetState::B0) ||
            RegionSize == 0 ||
            RegionSize > std::numeric_limits<uint64_t>::max() - VAddr) {
          log() << "hotswap: error: malformed or unsupported HotSwap pool "
                   "provenance.\n";
          return std::nullopt;
        }
        if (llvm::any_of(Provenance, [&](const PoolProvenance &Existing) {
              return Existing.VAddr == VAddr && Existing.Size == RegionSize;
            })) {
          log() << "hotswap: error: duplicate HotSwap pool provenance for "
                   "vaddr 0x"
                << utohexstr(VAddr) << ".\n";
          return std::nullopt;
        }
        Provenance.push_back(
            {VAddr, RegionSize,
             static_cast<ExecutablePoolTargetState>(RawState)});
      }
      Cursor = RecordEnd;
    }
  }

  if (Provenance.size() != ExternalSections.size() ||
      Provenance.size() != ExternalSegments.size()) {
    log() << "hotswap: error: external executable pool section/segment/note "
             "counts do not match.\n";
    return false;
  }

  for (const PoolProvenance &Pool : Provenance) {
    const ExecutableRegion *Section = nullptr;
    const ExecutableRegion *Segment = nullptr;
    unsigned SectionMatches = 0;
    unsigned SegmentMatches = 0;
    for (const ExecutableRegion &Region : ExternalSections)
      if (Region.VAddr == Pool.VAddr && Region.Size == Pool.Size) {
        Section = &Region;
        ++SectionMatches;
      }
    for (const ExecutableRegion &Region : ExternalSegments)
      if (Region.VAddr == Pool.VAddr && Region.Size == Pool.Size) {
        Segment = &Region;
        ++SegmentMatches;
      }
    if (SectionMatches != 1 || SegmentMatches != 1) {
      log() << "hotswap: error: HotSwap pool provenance at vaddr 0x"
            << utohexstr(Pool.VAddr)
            << " does not have one exact RX section/PT_LOAD mapping.\n";
      return false;
    }
    if (Section->FileOffset != Segment->FileOffset) {
      log() << "hotswap: error: HotSwap pool section/PT_LOAD at vaddr 0x"
            << utohexstr(Pool.VAddr)
            << " refer to different file bytes.\n";
      return false;
    }
    if (Pool.State != ExecutablePoolTargetState::Neutral &&
        Pool.State != TargetState) {
      log() << "hotswap: error: executable HotSwap pool at vaddr 0x"
            << utohexstr(Pool.VAddr)
            << " was produced for an incompatible target stepping.\n";
      return false;
    }
  }
  return true;
}

// -- ElfView::trampolinePoolVAddr ---------------------------------------------

std::optional<uint64_t> ElfView::trampolinePoolVAddr() const {
  uint64_t MaxAllocEnd = 0;
  for (const ELFT::Shdr &Shdr : Sections) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC))
      continue;
    // Overflow would collapse MaxAllocEnd and overlap the pool with existing
    // sections.
    std::optional<uint64_t> End = checkedAddUint64(
        Shdr.sh_addr, Shdr.sh_size, "allocatable section end for pool vaddr");
    if (!End)
      return std::nullopt;
    MaxAllocEnd = std::max(MaxAllocEnd, *End);
  }

  // A PT_LOAD may reserve a zero-fill tail that is not represented by any
  // section (p_memsz > p_filesz), or may cover mapped bytes without a section
  // header at all. The appended pool must be above those ranges as well.
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: failed to enumerate load segments for pool "
             "vaddr: "
          << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_LOAD)
      continue;
    std::optional<uint64_t> End = checkedAddUint64(
        Phdr.p_vaddr, Phdr.p_memsz, "load segment end for pool vaddr");
    if (!End)
      return std::nullopt;
    MaxAllocEnd = std::max(MaxAllocEnd, *End);
  }
  return checkedAlignToUint64(MaxAllocEnd, TrampolinePoolAlign,
                              "trampoline pool virtual address");
}

// -- addKernelEntryTrampolineSymbols ------------------------------------------

std::unique_ptr<WritableMemoryBuffer> addKernelEntryTrampolineSymbols(
    WritableMemoryBuffer &In, uint64_t PoolVAddr,
    ArrayRef<KernelEntryTrampolineFixup> Fixups) {
  if (Fixups.empty())
    return nullptr;

  const uint8_t *Data = reinterpret_cast<const uint8_t *>(In.getBufferStart());
  const size_t Size = In.getBufferSize();

  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Data), Size));
  if (!FileOrErr) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: failed to parse "
             "grown ELF: "
          << toString(FileOrErr.takeError()) << "\n";
    return nullptr;
  }
  ELFFileT File = std::move(*FileOrErr);
  Expected<ELFT::ShdrRange> SecsOrErr = File.sections();
  if (!SecsOrErr) {
    consumeError(SecsOrErr.takeError());
    return nullptr;
  }
  ELFT::ShdrRange Secs = *SecsOrErr;

  // Locate .symtab and its linked string table. Scan from the end, since the
  // symbol table sits near the end of the section list in these code objects.
  const ELFT::Shdr *SymShdr = nullptr;
  unsigned SymIdx = 0;
  for (unsigned I = Secs.size(); I-- > 0;)
    if (Secs[I].sh_type == ELF::SHT_SYMTAB) {
      SymShdr = &Secs[I];
      SymIdx = I;
      break;
    }
  if (!SymShdr) {
    log() << "hotswap: addKernelEntryTrampolineSymbols: no .symtab present; "
             "skipping stub symbols.\n";
    return nullptr;
  }
  for (const ELFT::Shdr &Shdr : Secs) {
    if (Shdr.sh_type != ELF::SHT_SYMTAB_SHNDX || Shdr.sh_link != SymIdx)
      continue;
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: cannot append "
             "symbols while a linked SHT_SYMTAB_SHNDX table is present.\n";
    return nullptr;
  }
  const unsigned StrIdx = SymShdr->sh_link;
  if (StrIdx == 0 || StrIdx >= Secs.size()) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: .symtab has an "
             "invalid sh_link ("
          << StrIdx << ").\n";
    return nullptr;
  }
  if (SymShdr->sh_entsize != sizeof(ELF::Elf64_Sym)) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: unexpected "
             ".symtab entry size "
          << SymShdr->sh_entsize << ".\n";
    return nullptr;
  }
  const ELFT::Shdr &StrShdr = Secs[StrIdx];
  if (StrShdr.sh_type != ELF::SHT_STRTAB) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: .symtab links "
             "to a non-string-table section.\n";
    return nullptr;
  }
  if ((SymShdr->sh_flags | StrShdr.sh_flags) & ELF::SHF_ALLOC) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: refusing to "
             "relocate an allocatable symbol or string table.\n";
    return nullptr;
  }

  const uint64_t SymOff = SymShdr->sh_offset;
  const uint64_t StrOff = StrShdr.sh_offset;
  std::optional<uint64_t> SymEnd = checkedAddUint64(
      SymOff, SymShdr->sh_size, "kernel-entry stub input .symtab end");
  std::optional<uint64_t> StrEnd = checkedAddUint64(
      StrOff, StrShdr.sh_size, "kernel-entry stub input .strtab end");
  if (!SymEnd || !StrEnd || *SymEnd > Size || *StrEnd > Size) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: symbol/string "
             "table extends past the ELF buffer.\n";
    return nullptr;
  }

  SmallVector<uint8_t> StrBlob, SymBlob;
  for (const KernelEntryTrampolineFixup &F : Fixups) {
    std::optional<uint64_t> StubVAddr = checkedAddUint64(
        PoolVAddr, F.StubTextOffset, "kernel-entry stub symbol vaddr");
    if (!StubVAddr)
      return nullptr;

    std::optional<unsigned> StubSectionIndex;
    for (unsigned I = 0; I != Secs.size(); ++I) {
      const ELFT::Shdr &Section = Secs[I];
      if (Section.sh_type != ELF::SHT_PROGBITS ||
          (Section.sh_flags & (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR)) !=
              (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR) ||
          *StubVAddr < Section.sh_addr)
        continue;

      const uint64_t SectionOffset = *StubVAddr - Section.sh_addr;
      if (SectionOffset > Section.sh_size ||
          KernelEntryStubStride > Section.sh_size - SectionOffset)
        continue;
      std::optional<uint64_t> StubFileOffset = checkedAddUint64(
          Section.sh_offset, SectionOffset,
          "kernel-entry stub symbol file offset");
      if (!StubFileOffset || *StubFileOffset > Size ||
          KernelEntryStubStride > Size - *StubFileOffset)
        continue;

      if (StubSectionIndex) {
        log() << "hotswap: error: addKernelEntryTrampolineSymbols: stub for '"
              << F.KernelName
              << "' is covered by multiple executable sections.\n";
        return nullptr;
      }
      StubSectionIndex = I;
    }
    if (!StubSectionIndex) {
      log() << "hotswap: error: addKernelEntryTrampolineSymbols: stub for '"
            << F.KernelName
            << "' is not fully contained in a file-backed executable "
               "section.\n";
      return nullptr;
    }
    if (*StubSectionIndex >= ELF::SHN_LORESERVE) {
      log() << "hotswap: error: addKernelEntryTrampolineSymbols: executable "
               "stub section index requires unsupported SHN_XINDEX.\n";
      return nullptr;
    }

    std::string Name = F.KernelName + ".stub";
    std::optional<uint64_t> NameOff = checkedAddUint64(
        StrShdr.sh_size, StrBlob.size(), "kernel-entry stub symbol name offset");
    if (!NameOff || *NameOff > std::numeric_limits<uint32_t>::max()) {
      log() << "hotswap: error: addKernelEntryTrampolineSymbols: string-table "
               "offset exceeds Elf64_Sym::st_name.\n";
      return nullptr;
    }
    StrBlob.append(Name.begin(), Name.end());
    StrBlob.push_back(0);

    ELF::Elf64_Sym Sym{};
    Sym.st_name = static_cast<uint32_t>(*NameOff);
    Sym.st_info = (ELF::STB_GLOBAL << 4) | ELF::STT_FUNC;
    Sym.st_other = ELF::STV_DEFAULT;
    Sym.st_shndx = static_cast<uint16_t>(*StubSectionIndex);
    Sym.st_value = *StubVAddr;
    Sym.st_size = KernelEntryStubStride;
    const uint8_t *P = reinterpret_cast<const uint8_t *>(&Sym);
    SymBlob.append(P, P + sizeof(Sym));
  }

  std::optional<uint64_t> NewStrSize = checkedAddUint64(
      StrShdr.sh_size, StrBlob.size(), "relocated stub .strtab size");
  std::optional<uint64_t> NewSymSize = checkedAddUint64(
      SymShdr->sh_size, SymBlob.size(), "relocated stub .symtab size");
  if (!NewStrSize || !NewSymSize)
    return nullptr;

  auto SectionAlignment = [](uint64_t Alignment, uint64_t Minimum,
                             StringRef Name) -> std::optional<uint64_t> {
    Alignment = std::max(Alignment, Minimum);
    if (!isPowerOf2_64(Alignment)) {
      log() << "hotswap: error: addKernelEntryTrampolineSymbols: " << Name
            << " alignment " << Alignment << " is not a power of two.\n";
      return std::nullopt;
    }
    return Alignment;
  };
  std::optional<uint64_t> StrAlign =
      SectionAlignment(StrShdr.sh_addralign, 1, ".strtab");
  std::optional<uint64_t> SymAlign = SectionAlignment(
      SymShdr->sh_addralign, alignof(ELF::Elf64_Sym), ".symtab");
  if (!StrAlign || !SymAlign)
    return nullptr;

  // Relocate these non-allocating tables after the complete grown ELF. This
  // leaves the executable pool, e_phoff, and every PT_LOAD p_offset unchanged.
  std::optional<uint64_t> NewStrOff = checkedAlignToUint64(
      Size, *StrAlign, "relocated kernel-entry stub .strtab offset");
  if (!NewStrOff)
    return nullptr;
  std::optional<uint64_t> AfterStr = checkedAddUint64(
      *NewStrOff, *NewStrSize, "relocated kernel-entry stub .strtab end");
  if (!AfterStr)
    return nullptr;
  std::optional<uint64_t> NewSymOff = checkedAlignToUint64(
      *AfterStr, *SymAlign, "relocated kernel-entry stub .symtab offset");
  if (!NewSymOff)
    return nullptr;
  std::optional<uint64_t> NewSizeOr = checkedAddUint64(
      *NewSymOff, *NewSymSize, "kernel-entry stub symbol ELF size");
  if (!NewSizeOr || *NewSizeOr > std::numeric_limits<size_t>::max())
    return nullptr;
  const size_t NewSize = static_cast<size_t>(*NewSizeOr);

  std::unique_ptr<WritableMemoryBuffer> Out =
      WritableMemoryBuffer::getNewMemBuffer(NewSize);
  if (!Out) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: allocation of "
          << NewSize << " bytes failed.\n";
    return nullptr;
  }
  uint8_t *O = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  std::memcpy(O, Data, Size);
  std::memcpy(O + *NewStrOff, Data + StrOff, StrShdr.sh_size);
  std::memcpy(O + *NewStrOff + StrShdr.sh_size, StrBlob.data(), StrBlob.size());
  std::memcpy(O + *NewSymOff, Data + SymOff, SymShdr->sh_size);
  std::memcpy(O + *NewSymOff + SymShdr->sh_size, SymBlob.data(), SymBlob.size());

  uint64_t Shoff;
  uint16_t Shentsize, Shnum;
  std::memcpy(&Shoff, O + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, O + offsetof(Ehdr, e_shentsize), sizeof(Shentsize));
  std::memcpy(&Shnum, O + offsetof(Ehdr, e_shnum), sizeof(Shnum));
  if (Shentsize < sizeof(Shdr) || Shoff > Size ||
      static_cast<uint64_t>(Shnum) * Shentsize > Size - Shoff)
    return nullptr;

  auto RelocateSection = [&](unsigned Index, uint64_t Offset, uint64_t Size) {
    uint8_t *Sh = O + Shoff + static_cast<uint64_t>(Index) * Shentsize;
    std::memcpy(Sh + offsetof(Shdr, sh_offset), &Offset, sizeof(Offset));
    std::memcpy(Sh + offsetof(Shdr, sh_size), &Size, sizeof(Size));
  };
  RelocateSection(StrIdx, *NewStrOff, *NewStrSize);
  RelocateSection(SymIdx, *NewSymOff, *NewSymSize);

  log() << "hotswap: added " << Fixups.size()
        << " kernel-entry stub symbol(s) to .symtab\n";
  return Out;
}

// -- ElfView::growWithTrampolines ---------------------------------------------

std::unique_ptr<WritableMemoryBuffer>
ElfView::growWithTrampolines(ArrayRef<Trampoline> Trampolines,
                             ArrayRef<uint8_t> SNopBytes,
                             ExecutablePoolTargetState TargetState) const {
  // SNopBytes is unused in the append-at-end model: nothing between .text and
  // the following sections moves, so there is no in-image gap to pad. It is
  // retained in the signature for callers and for a future in-place variant.
  (void)SNopBytes;

  const size_t InputSize = size();
  const uint8_t *Input = data();

  if (static_cast<uint32_t>(TargetState) >
      static_cast<uint32_t>(ExecutablePoolTargetState::B0)) {
    log() << "hotswap: error: growWithTrampolines: invalid executable pool "
             "target state.\n";
    return nullptr;
  }

  if (InputSize < sizeof(Ehdr)) {
    log() << "hotswap: error: growWithTrampolines: input (" << InputSize
          << " bytes) is smaller than an ELF64 header.\n";
    return nullptr;
  }

  size_t TrampTotal = 0;
  for (const Trampoline &T : Trampolines) {
    if (T.Bytes.size() > std::numeric_limits<size_t>::max() - TrampTotal) {
      log() << "hotswap: error: growWithTrampolines: trampoline byte count "
            << "overflows size_t.\n";
      return nullptr;
    }
    TrampTotal += T.Bytes.size();
  }
  if (TrampTotal == 0) {
    log() << "hotswap: growWithTrampolines: no trampolines to insert; "
          << "returning empty result.\n";
    return nullptr;
  }

  // Append the pool at a fresh virtual address above every existing
  // allocatable section (trampolinePoolVAddr()). Because existing sections,
  // symbols, and program headers keep their addresses, the baked PC-relative
  // literals (and DWARF) that reference post-.text data stay valid. The
  // previous scheme grew .text in place and shifted everything after it,
  // silently corrupting those baked references (a fully-linked AMDGPU object
  // carries no relocations) -- see
  // ElfView.GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol.
  //
  // The vaddr and file offset are page-aligned (equal modulo the alignment) so
  // the appended PT_LOAD maps consistently.
  std::optional<uint64_t> PoolVAddrOr = trampolinePoolVAddr();
  if (!PoolVAddrOr) {
    log() << "hotswap: error: growWithTrampolines: could not compute a "
          << "trampoline pool virtual address.\n";
    return nullptr;
  }
  const uint64_t PoolVAddr = *PoolVAddrOr;
  std::optional<uint64_t> PoolFileOffOr = checkedAlignToUint64(
      static_cast<uint64_t>(InputSize), TrampolinePoolAlign,
      "trampoline pool file offset");
  if (!PoolFileOffOr)
    return nullptr;
  const uint64_t PoolFileOff = *PoolFileOffOr;

  // Copy the program-header and section-header tables to the end of the file.
  // Add a PT_LOAD and SHF_ALLOC|SHF_EXECINSTR section for the pool, plus a
  // non-allocating SHT_NOTE that records its exact range and target state for a
  // subsequent rewrite. Then repoint e_phoff / e_shoff. Those tables and the
  // note are metadata addressed through the ELF header, so relocating them
  // moves nothing a baked literal can reference.
  uint64_t Phoff, Shoff;
  uint16_t Phentsize, Phnum, Shentsize, Shnum;
  std::memcpy(&Phoff, Input + offsetof(Ehdr, e_phoff), sizeof(Phoff));
  std::memcpy(&Phentsize, Input + offsetof(Ehdr, e_phentsize),
              sizeof(Phentsize));
  std::memcpy(&Phnum, Input + offsetof(Ehdr, e_phnum), sizeof(Phnum));
  std::memcpy(&Shoff, Input + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, Input + offsetof(Ehdr, e_shentsize),
              sizeof(Shentsize));
  std::memcpy(&Shnum, Input + offsetof(Ehdr, e_shnum), sizeof(Shnum));

  const bool HasPhdrs =
      Phnum > 0 && Phoff != 0 && Phentsize >= sizeof(Phdr) &&
      Phoff <= InputSize &&
      static_cast<uint64_t>(Phnum) * Phentsize <= InputSize - Phoff;
  const bool HasShdrs =
      Shnum > 0 && Shoff != 0 && Shentsize >= sizeof(Shdr) &&
      Shoff <= InputSize &&
      static_cast<uint64_t>(Shnum) * Shentsize <= InputSize - Shoff;
  if (!HasPhdrs || !HasShdrs) {
    log() << "hotswap: error: growWithTrampolines: cannot record executable "
             "pool provenance without valid program- and section-header "
             "tables.\n";
    return nullptr;
  }

  std::optional<uint64_t> PoolEnd =
      checkedAddUint64(PoolFileOff, TrampTotal, "trampoline pool file end");
  std::optional<uint64_t> PoolVAddrEnd = checkedAddUint64(
      PoolVAddr, TrampTotal, "trampoline pool virtual-address end");
  if (!PoolEnd || !PoolVAddrEnd)
    return nullptr;
  SmallVector<uint8_t, 48> PoolNote =
      buildHotswapPoolNote(PoolVAddr, TrampTotal, TargetState);
  std::optional<uint64_t> PoolNoteFileOffOr = checkedAlignToUint64(
      *PoolEnd, uint64_t{4}, "trampoline pool provenance note offset");
  if (!PoolNoteFileOffOr)
    return nullptr;
  const uint64_t PoolNoteFileOff = *PoolNoteFileOffOr;
  std::optional<uint64_t> PoolNoteEnd = checkedAddUint64(
      PoolNoteFileOff, PoolNote.size(), "trampoline pool provenance note end");
  if (!PoolNoteEnd)
    return nullptr;

  // Lay out metadata after the pool: [pool][note][phdrs][shdrs].
  uint64_t Cursor = *PoolNoteEnd;
  const uint64_t NewPhnum = static_cast<uint64_t>(Phnum) + 1;
  const uint64_t NewShnum = static_cast<uint64_t>(Shnum) + 2;
  if (NewPhnum >= ELF::PN_XNUM) {
    log() << "hotswap: error: growWithTrampolines: program-header count "
          << NewPhnum
          << " after appending a PT_LOAD requires unsupported extended "
             "numbering.\n";
    return nullptr;
  }
  std::optional<uint64_t> NewPhoffOr = checkedAlignToUint64(
      Cursor, static_cast<uint64_t>(alignof(Phdr)),
      "relocated phdr table offset");
  if (!NewPhoffOr)
    return nullptr;
  const uint64_t NewPhoff = *NewPhoffOr;
  std::optional<uint64_t> PhdrEnd = checkedAddUint64(
      NewPhoff, NewPhnum * Phentsize, "relocated phdr table end");
  if (!PhdrEnd)
    return nullptr;
  Cursor = *PhdrEnd;

  if (NewShnum >= ELF::SHN_LORESERVE) {
    log() << "hotswap: error: growWithTrampolines: section-header count "
          << NewShnum
          << " after appending the pool and provenance sections requires "
             "unsupported extended numbering.\n";
    return nullptr;
  }
  std::optional<uint64_t> NewShoffOr = checkedAlignToUint64(
      Cursor, static_cast<uint64_t>(alignof(Shdr)),
      "relocated shdr table offset");
  if (!NewShoffOr)
    return nullptr;
  const uint64_t NewShoff = *NewShoffOr;
  std::optional<uint64_t> ShdrEnd = checkedAddUint64(
      NewShoff, NewShnum * Shentsize, "relocated shdr table end");
  if (!ShdrEnd)
    return nullptr;
  Cursor = *ShdrEnd;
  if (Cursor > std::numeric_limits<size_t>::max()) {
    log()
        << "hotswap: error: growWithTrampolines: grown size exceeds size_t.\n";
    return nullptr;
  }
  const size_t NewSize = static_cast<size_t>(Cursor);

  // getNewMemBuffer zero-initializes, so the alignment gaps between regions are
  // well-defined padding without extra memsets.
  std::unique_ptr<WritableMemoryBuffer> Buf =
      WritableMemoryBuffer::getNewMemBuffer(NewSize);
  if (!Buf) {
    log() << "hotswap: error: growWithTrampolines: "
          << "WritableMemoryBuffer::getNewMemBuffer(" << NewSize
          << ") failed (out of memory).\n";
    return nullptr;
  }

  uint8_t *Out = reinterpret_cast<uint8_t *>(Buf->getBufferStart());
  // 1. Original bytes verbatim -- nothing shifts.
  std::memcpy(Out, Input, InputSize);
  // 2. Trampoline pool at its fresh, page-aligned file offset / vaddr.
  size_t Pos = static_cast<size_t>(PoolFileOff);
  for (const Trampoline &T : Trampolines) {
    std::memcpy(Out + Pos, T.Bytes.data(), T.Bytes.size());
    Pos += T.Bytes.size();
  }
  // 3. Non-allocating standard ELF note describing the executable pool.
  std::memcpy(Out + PoolNoteFileOff, PoolNote.data(), PoolNote.size());
  // 4. Relocated program-header table + appended PT_LOAD for the pool.
  std::memcpy(Out + NewPhoff, Input + Phoff,
              static_cast<size_t>(Phnum) * Phentsize);
  // PT_PHDR promises that the program-header table is part of the process
  // image at the address recorded in that entry. The relocated table is
  // metadata after the pool and is deliberately not mapped by a PT_LOAD, so an
  // inherited PT_PHDR would describe the obsolete table. PT_PHDR is optional;
  // remove the stale promise while preserving the remaining table slots.
  for (uint16_t I = 0; I != Phnum; ++I) {
    uint8_t *Entry = Out + NewPhoff + static_cast<uint64_t>(I) * Phentsize;
    Phdr Existing{};
    std::memcpy(&Existing, Entry, sizeof(Existing));
    if (Existing.p_type != ELF::PT_PHDR)
      continue;
    Phdr Null{};
    std::memcpy(Entry, &Null, sizeof(Null));
  }
  Phdr PoolPhdr{};
  PoolPhdr.p_type = ELF::PT_LOAD;
  PoolPhdr.p_flags = ELF::PF_R | ELF::PF_X;
  PoolPhdr.p_offset = PoolFileOff;
  PoolPhdr.p_vaddr = PoolVAddr;
  PoolPhdr.p_paddr = PoolVAddr;
  PoolPhdr.p_filesz = TrampTotal;
  PoolPhdr.p_memsz = TrampTotal;
  PoolPhdr.p_align = TrampolinePoolAlign;
  std::memcpy(Out + NewPhoff + static_cast<uint64_t>(Phnum) * Phentsize,
              &PoolPhdr, sizeof(PoolPhdr));
  std::memcpy(Out + offsetof(Ehdr, e_phoff), &NewPhoff, sizeof(NewPhoff));
  uint16_t NewPhnum16 = static_cast<uint16_t>(NewPhnum);
  std::memcpy(Out + offsetof(Ehdr, e_phnum), &NewPhnum16, sizeof(NewPhnum16));
  // 5. Relocated section-header table + appended pool and provenance sections.
  // Both have empty names (sh_name == 0), avoiding .shstrtab surgery.
  std::memcpy(Out + NewShoff, Input + Shoff,
              static_cast<size_t>(Shnum) * Shentsize);
  Shdr PoolShdr{};
  PoolShdr.sh_name = 0;
  PoolShdr.sh_type = ELF::SHT_PROGBITS;
  PoolShdr.sh_flags = ELF::SHF_ALLOC | ELF::SHF_EXECINSTR;
  PoolShdr.sh_addr = PoolVAddr;
  PoolShdr.sh_offset = PoolFileOff;
  PoolShdr.sh_size = TrampTotal;
  PoolShdr.sh_addralign = TrampolinePoolAlign;
  std::memcpy(Out + NewShoff + static_cast<uint64_t>(Shnum) * Shentsize,
              &PoolShdr, sizeof(PoolShdr));
  Shdr NoteShdr{};
  NoteShdr.sh_name = 0;
  NoteShdr.sh_type = ELF::SHT_NOTE;
  NoteShdr.sh_offset = PoolNoteFileOff;
  NoteShdr.sh_size = PoolNote.size();
  NoteShdr.sh_addralign = 4;
  std::memcpy(Out + NewShoff +
                  (static_cast<uint64_t>(Shnum) + 1) * Shentsize,
              &NoteShdr, sizeof(NoteShdr));
  std::memcpy(Out + offsetof(Ehdr, e_shoff), &NewShoff, sizeof(NewShoff));
  uint16_t NewShnum16 = static_cast<uint16_t>(NewShnum);
  std::memcpy(Out + offsetof(Ehdr, e_shnum), &NewShnum16, sizeof(NewShnum16));

  log() << "hotswap: growWithTrampolines: appended " << Trampolines.size()
        << (Trampolines.size() == 1 ? " trampoline (" : " trampolines (")
        << TrampTotal << " bytes) at vaddr 0x" << utohexstr(PoolVAddr)
        << " (file 0x" << utohexstr(PoolFileOff) << "); grew ELF from "
        << InputSize << " to " << NewSize << " bytes with target-state "
        << static_cast<uint32_t>(TargetState) << " provenance.\n";
  return Buf;
}

} // namespace hotswap
} // namespace COMGR
