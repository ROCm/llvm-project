//===- flat-addr.h - Hotswap transpiler -----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_FLAT_ADDR_H
#define HOTSWAP_TRANSPILER_FLAT_ADDR_H

#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

// Emit the address a GLOBAL memory instruction accesses, as a pointer into the
// global address space with the instruction's immediate byte offset folded in.
// Both addressing forms are recognized: a per-lane 64-bit address in `vaddr`,
// and an SGPR-pair base in `saddr` that a per-lane 32-bit offset in `vaddr` is
// added to. AccessAlign is the alignment the access is modeled as having, which
// the immediate offset has to preserve. Returns a structured refusal for that
// offset, for the address-scaling modifier, and for a cache policy the raiser
// does not model.
llvm::Expected<llvm::Value *> emitGlobalAddress(RaiseContext &Ctx,
                                                const DecodedInst &Di,
                                                llvm::Align AccessAlign);

} // namespace COMGR::hotswap

#endif
