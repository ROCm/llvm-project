//===- code-object-utils.h - AMDGPU code-object metadata ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extracts the AMDGPU-specific metadata the hotswap raiser needs from an ELF
// code object: the per-kernel MsgPack-derived ABI surface and the kernel
// descriptor fields read from .rodata.
//
// `CodeObjectInfo::create` is the entry point. It validates the code-object
// invariants the raiser relies on (little-endian 64-bit AMDGPU HSA ELF with a
// symbol table and a supported metadata version), parses the ELF and the
// AMDGPU MsgPack notes exactly once, and reads and validates every kernel
// descriptor up front. Queries then read from the parsed model without
// re-parsing. Results are `llvm::Expected`; forwarded LLVM errors keep their
// original ErrorInfo type, hotswap-detected malformed input uses `HotswapError`
// from `hotswap-error.h`.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H
#define HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H

#include "hotswap/common/kernel-meta.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <memory>
#include <string>

namespace COMGR::hotswap {

/// The AMDGPU `.text` section, as returned by `CodeObjectInfo::textSection`.
/// `Bytes` references the section contents inside the ELF MemoryBuffer, so that
/// buffer must outlive the `TextSection`.
struct TextSection {
  /// Runtime address of `.text`; PC-relative instructions use this source
  /// code-object address domain.
  uint64_t Address = 0;
  /// `.text` bytes, indexed by text-relative decoded instruction offsets.
  llvm::ArrayRef<uint8_t> Bytes;

  /// An allocated source section whose bytes may be read through PC-relative
  /// SMEM. `Bytes` references the ELF MemoryBuffer, as for `TextSection`.
  struct ImageSection {
    uint64_t Address = 0;
    llvm::ArrayRef<uint8_t> Bytes;
  };
  /// Minimal source image used for literal-table materialisation.
  llvm::SmallVector<ImageSection> ImageSections;
};

/// Resolved text-section extent for a kernel symbol. `Offset` is relative to
/// `.text`; `Size` bounds decoding to the selected symbol's byte range.
struct KernelSymbolExtent {
  uint64_t Offset = 0;
  uint64_t Size = 0;
};

/// A parsed, validated AMDGPU code object. Owns the `ObjectFile` and the fully
/// resolved per-kernel metadata; `create` performs all parsing and validation,
/// so the query methods never re-parse and never fail on malformed metadata.
///
/// The referenced `MemoryBufferRef` must outlive the `CodeObjectInfo`, and the
/// `TextSection` byte ranges it hands back reference that buffer in turn.
class CodeObjectInfo {
public:
  /// Parse and validate `ElfData`. Rejects code objects that are not
  /// little-endian 64-bit AMDGPU HSA ELFs, are stripped of their symbol table,
  /// declare an unsupported metadata version, or carry malformed / internally
  /// inconsistent kernel metadata or descriptors.
  static llvm::Expected<CodeObjectInfo> create(llvm::MemoryBufferRef ElfData);

  CodeObjectInfo(CodeObjectInfo &&) = default;
  CodeObjectInfo &operator=(CodeObjectInfo &&) = default;
  CodeObjectInfo(const CodeObjectInfo &) = delete;
  CodeObjectInfo &operator=(const CodeObjectInfo &) = delete;

  llvm::object::ObjectFile &object() const { return *Obj; }

  /// Kernel names in metadata declaration order.
  llvm::ArrayRef<std::string> kernelNames() const { return KernelOrder; }

  /// Resolved metadata for `KernelName`, or a `HotswapError` when the code
  /// object declares no such kernel.
  llvm::Expected<const KernelMeta *> kernel(llvm::StringRef KernelName) const;

  /// The `.text` section plus the image sections readable through PC-relative
  /// SMEM. The byte ranges reference the ELF buffer.
  llvm::Expected<TextSection> textSection() const;

  /// The `.text`-relative offset and extent of `KernelName`. A zero-sized
  /// kernel symbol is bounded by the next distinct function-symbol address, so
  /// symbol placement alone never merges two functions.
  llvm::Expected<KernelSymbolExtent>
  kernelSymbolExtent(llvm::StringRef KernelName) const;

  /// The `.text`-relative extent of every function symbol, sorted by ascending
  /// offset. A zero-sized symbol is bounded by the next distinct address (or
  /// the end of `.text`); aliases at the same address are skipped.
  llvm::Expected<llvm::SmallVector<KernelSymbolExtent>>
  textFunctionExtents() const;

private:
  CodeObjectInfo() = default;

  std::unique_ptr<llvm::object::ObjectFile> Obj;
  /// Fully resolved per-kernel metadata, keyed by `.name`. The values own their
  /// strings, so the source MsgPack document need not outlive `create`.
  llvm::StringMap<KernelMeta> Kernels;
  /// Kernel `.name`s in declaration order, for stable iteration.
  llvm::SmallVector<std::string> KernelOrder;
};

} // namespace COMGR::hotswap

#endif
