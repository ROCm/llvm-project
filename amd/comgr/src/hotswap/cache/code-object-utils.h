//===- code-object-utils.h - AMDGPU code-object metadata ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The single code-object metadata query the translation cache needs: the
// list of kernel names declared in an ELF's AMDGPU MsgPack notes. Depends
// only on LLVM (ELF object + MsgPack) so the cache module has no comgr
// metadata-layer coupling.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H
#define HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <string>

namespace COMGR::hotswap {

/// List the kernel names declared in the AMDGPU MsgPack notes embedded in
/// `ElfData`. Returns an error when no AMDGPU metadata note is present.
llvm::Expected<llvm::SmallVector<std::string>>
listKernelNames(llvm::MemoryBufferRef ElfData);

} // namespace COMGR::hotswap

#endif
