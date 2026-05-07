//===- code-object-utils.h - AMDGPU code-object metadata ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API for extracting the AMDGPU-specific metadata the hotswap raiser
// pipeline needs from an ELF code object: the .text bytes the instructions
// live in, the per-kernel MsgPack-derived ABI surface, and the kernel
// descriptor register fields read directly from .rodata. Each entry point
// returns `llvm::Expected<...>` (or `llvm::Error`) on failure -- forwarded
// LLVM errors keep their original ErrorInfo type, hotswap-detected
// mismatches use `HotswapError` from `hotswap-error.h`.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H
#define HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstdint>
#include <string>

namespace COMGR::hotswap {

/// Raw bytes of the AMDGPU `.text` section, as returned by
/// `extractTextSection`. The byte buffer is owned by the `TextSection` --
/// the underlying ELF MemoryBuffer is not borrowed across the call.
/// `Offset` and `Size` mirror the section header's `sh_addr` / `sh_size`
/// for callers that need to relate raw `.text` byte positions back to
/// ELF virtual addresses (e.g. relocation handling or symbol-offset
/// arithmetic against the on-disk ELF).
struct TextSection {
  llvm::SmallVector<uint8_t> Bytes;
  uint64_t Offset = 0;
  uint64_t Size = 0;
};

/// One entry of the kernel argument table extracted from the AMDGPU MsgPack
/// notes. Mirrors the AMDHSA `.args` schema; absent fields stay at the
/// constructor defaults below.
struct KernelArgMeta {
  std::string Name;
  uint32_t Offset = 0;
  uint32_t Size = 0;
  /// AMDHSA `.value_kind` enum spelling (e.g. `by_value`, `global_buffer`,
  /// `hidden_global_offset_x`). Kept as a string because the AMDHSA spec
  /// adds new kinds without bumping the metadata version, so a hand-rolled
  /// enum here would silently lose round-trip fidelity for unrecognised
  /// kinds.
  std::string ValueKind;
  /// LLVM AMDGPU address space id for pointer-typed arguments. Defaults to
  /// 0 -- the LLVM-default flat address space -- which matches non-pointer
  /// arguments where AMDHSA omits the field entirely.
  unsigned AddressSpace = 0;
};

/// Per-kernel metadata extracted from the AMDGPU code object's MsgPack notes
/// + kernel descriptor (`<name>.kd`).
struct KernelMeta {
  std::string Name;
  uint32_t KernargSegmentSize = 0;
  uint32_t GroupSegmentFixedSize = 0;
  uint32_t PrivateSegmentFixedSize = 0;
  uint32_t MaxFlatWorkgroupSize = 256;
  llvm::SmallVector<KernelArgMeta> Args;

  // ---------------------------------------------------------------------
  // Kernel descriptor (KD) raw fields.
  //
  // Populated by extractKernelMeta from the 64-byte amd_kernel_code_t block
  // that lives at the symbol named `<kernelName>.kd` (always in the .rodata
  // section for amdhsa code objects). These fields are the entire
  // surface needed to derive the source-ISA SGPR ABI:
  //
  //   * privateSegmentFixedSize (KD bytes 4-7, mirrored from MsgPack): source
  //     private/scratch bytes per work-item. A non-zero value paired with
  //     `compute_pgm_rsrc2.ENABLE_PRIVATE_SEGMENT` is the launch-time ABI
  //     request that makes ROCR/SPI allocate scratch backing.
  //
  //   * kernelCodeProperties  (KD bytes 56-57): bit field selecting which
  //     `enable_sgpr_*` user SGPRs the loader / packet processor will pre-
  //     populate before kernel entry. See LLVM's AMDHSAKernelDescriptor.h
  //     KERNEL_CODE_PROPERTY_ENABLE_SGPR_* enum for the bit positions.
  //
  //   * kernargPreload        (KD bytes 58-59): packed
  //     {LENGTH[6:0], OFFSET[15:7]} per LLVM's KERNARG_PRELOAD_SPEC enum.
  //     LENGTH=N and OFFSET=K mean: the hardware copies N dwords of kernarg
  //     memory starting at byte (K*4) into user SGPRs immediately above the
  //     `enable_sgpr_*`-selected ones, before kernel entry. This is the
  //     gfx1250-specific kernarg preload mechanism that the user-SGPR
  //     layout consumer needs to know about.
  //
  //   * computePgmRsrc2       (KD bytes 52-55): contains
  //     ENABLE_SGPR_WORKGROUP_ID_{X,Y,Z} / WORKGROUP_INFO bits and the
  //     USER_SGPR_COUNT field (read for verification only -- we recompute
  //     it from kernelCodeProperties + kernargPreload.length and assert
  //     equality).
  //
  //   * computePgmRsrc1       (KD bytes 48-51): not strictly required for
  //     the user-SGPR layout, but useful for diagnostics and for future
  //     wave-size-aware decisions. Captured for completeness.
  //
  // `HasKernelDescriptor` is true iff the .rodata KD bytes parsed
  // cleanly. extractKernelMeta treats KD parse failure as a partial
  // success (logs the underlying Error, returns the MsgPack-derived
  // fields, leaves this flag false) so callers who only need the
  // MsgPack metadata are not blocked by a missing KD. raiseToIR
  // gates its non-empty-input lift on this flag -- empty-input
  // scaffolding mode skips the check.
  bool HasKernelDescriptor = false;
  uint32_t ComputePgmRsrc1 = 0;
  uint32_t ComputePgmRsrc2 = 0;
  uint16_t KernelCodeProperties = 0;
  uint16_t KernargPreload = 0;

  /// Byte offset (8-byte aligned) of the first hidden argument in the
  /// kernarg segment. Hidden arguments (`hidden_*` value kinds) are
  /// appended after every explicit argument.
  uint64_t implicitArgsBase() const {
    uint64_t MaxEnd = 0;
    for (const KernelArgMeta &Arg : Args) {
      if (llvm::StringRef(Arg.ValueKind).starts_with("hidden_")) {
        continue;
      }
      uint64_t End = static_cast<uint64_t>(Arg.Offset) + Arg.Size;
      if (End > MaxEnd) {
        MaxEnd = End;
      }
    }
    return llvm::alignTo(MaxEnd, 8);
  }
};

/// Extract the `.text` section bytes from `ElfData`. Returns a
/// `HotswapError` when the ELF parses but has no `.text` section;
/// forwards `llvm::object` parse errors unchanged.
llvm::Expected<TextSection> extractTextSection(llvm::MemoryBufferRef ElfData);

/// List the kernel names declared in the AMDGPU MsgPack notes embedded in
/// `ElfData`. Returns a `HotswapError` when no AMDGPU metadata note is
/// present.
llvm::Expected<llvm::SmallVector<std::string>>
listKernelNames(llvm::MemoryBufferRef ElfData);

/// Extract the per-kernel metadata for `KernelName` from the MsgPack notes
/// in `ElfData`, including the kernel-descriptor register fields read out
/// of `.rodata`. KD-bytes lookup is best-effort: a usable KernelMeta is
/// still returned for the MsgPack-derived fields when the .rodata KD blob
/// is unreachable, with `HasKernelDescriptor == false`.
llvm::Expected<KernelMeta> extractKernelMeta(llvm::MemoryBufferRef ElfData,
                                             llvm::StringRef KernelName);

/// Resolve the byte offset of the kernel symbol `KernelName` within the
/// `.text` section of `ElfData`.
llvm::Expected<uint64_t> findKernelSymbolOffset(llvm::MemoryBufferRef ElfData,
                                                llvm::StringRef KernelName);

} // namespace COMGR::hotswap

#endif
