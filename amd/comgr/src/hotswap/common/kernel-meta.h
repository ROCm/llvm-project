//===- kernel-meta.h - Hotswap kernel ABI model -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The translation data model shared by the code-object loader and the raiser.
// It carries only the extracted kernel ABI, so translation clients can depend
// on it without pulling in the LLVM object-layer that the loader needs.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_KERNEL_META_H
#define HOTSWAP_TRANSPILER_KERNEL_META_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace COMGR::hotswap {

/// One entry of the kernel argument table extracted from the AMDGPU MsgPack
/// notes. Mirrors the AMDHSA `.args` schema; absent optional fields keep the
/// defaults.
struct KernelArgMeta {
  std::string Name;
  uint32_t Offset = 0;
  uint32_t Size = 0;
  /// AMDHSA `.value_kind` spelling (e.g. `by_value`, `global_buffer`). Kept as
  /// a string because the spec adds kinds without bumping the metadata version,
  /// so a hand-rolled enum would lose fidelity for unrecognised kinds.
  std::string ValueKind;
  /// AMDHSA `.address_space` spelling (e.g. `global`, `local`) for pointer
  /// arguments; empty when the metadata omits it. Kept as a string because the
  /// schema specifies it as a string enum, not an integer id.
  std::string AddressSpace;
};

/// Per-kernel metadata extracted from the AMDGPU MsgPack notes and the kernel
/// descriptor (`<name>.kd`).
struct KernelMeta {
  std::string Name;
  /// AMDHSA `.symbol`: the descriptor object's symbol name. Not required to
  /// match `Name`, so it is read from the metadata rather than synthesized.
  std::string Symbol;

  /// AMDGPU code object version (4, 5, or 6), taken from e_ident[EI_ABIVERSION]
  /// at the code-object boundary. Retained for version-dependent ABI handling.
  uint8_t CodeObjectVersion = 0;

  /// Absolute source address of the kernel entry in `.text`, decoded from the
  /// descriptor's signed `kernel_code_entry_byte_offset`. Authoritative over
  /// any name-based symbol lookup: `.symbol` (hence the descriptor) may differ
  /// from
  /// `.name`, so the descriptor's own entry offset is the sole binding between
  /// a kernel's ABI and its code.
  uint64_t EntryAddress = 0;
  uint32_t KernargSegmentSize = 0;
  uint32_t GroupSegmentFixedSize = 0;
  uint32_t PrivateSegmentFixedSize = 0;
  uint32_t MaxFlatWorkgroupSize = 256;

  /// Code object v6 `.cluster_dims`, absent when the metadata omits it. A
  /// present but all-zero value means clusters are disabled; any non-zero value
  /// carries source cluster state the raiser does not reconstruct yet.
  std::optional<std::array<uint32_t, 3>> ClusterDims;

  llvm::SmallVector<KernelArgMeta> Args;

  bool hasNonDisabledClusterDims() const {
    return ClusterDims &&
           llvm::any_of(*ClusterDims, [](uint32_t Dim) { return Dim != 0; });
  }

  /// The following fields come from `<name>.kd` in .rodata and are always
  /// populated: the loader refuses a code object whose descriptor it cannot
  /// read and validate.
  /// compute_pgm_rsrc1; kept for diagnostics and wave-size-aware decisions.
  uint32_t ComputePgmRsrc1 = 0;
  /// compute_pgm_rsrc2; carries the ENABLE_SGPR_WORKGROUP_ID_* and
  /// USER_SGPR_COUNT fields the user-SGPR layout consumes.
  uint32_t ComputePgmRsrc2 = 0;
  /// Selects which enable_sgpr_* user SGPRs the loader pre-populates before
  /// entry (see AMDHSAKernelDescriptor.h KERNEL_CODE_PROPERTY_ENABLE_SGPR_*).
  uint16_t KernelCodeProperties = 0;
  /// Packed {LENGTH[6:0], OFFSET[15:7]} (see KERNARG_PRELOAD_SPEC): the gfx1250
  /// kernarg preload the user-SGPR layout needs; absent from the MsgPack notes,
  /// which is why the descriptor is read directly.
  uint16_t KernargPreload = 0;
};

} // namespace COMGR::hotswap

#endif
