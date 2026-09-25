//===- flat-addr.h - Transpiler -------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_FLAT_ADDR_H
#define TRANSPILER_FLAT_ADDR_H

#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/raise-context.h"

#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

namespace COMGR::transpiler {

// Emit the address a GLOBAL memory instruction accesses, as a pointer into the
// global address space with the instruction's immediate byte offset folded in.
// Both addressing forms are recognized: a per-lane 64-bit address in `vaddr`,
// and an SGPR-pair base in `saddr` that a per-lane 32-bit offset in `vaddr` is
// added to. For gfx1250 SADDR forms, scale_offset multiplies the signed lane
// offset by AccessSizeInBytes. AccessAlign is the modeled alignment, which the
// immediate offset must preserve. Returns a structured refusal for unsupported
// addressing forms, offsets, or cache policies.
llvm::Expected<llvm::Value *> emitGlobalAddress(RaiseContext &Ctx,
                                                const DecodedInst &Di,
                                                unsigned AccessSizeInBytes,
                                                llvm::Align AccessAlign);

} // namespace COMGR::transpiler

#endif
