//===- flat-addr.h - Hotswap transpiler -----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_FLAT_ADDR_H
#define HOTSWAP_TRANSPILER_FLAT_ADDR_H

#include "decoded-inst.h"
#include "parsed-reg.h"
#include "raise-context.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

// Decoded address-shape of a FLAT / GLOBAL load or store.
//
// `ptr` is the final address-space-1 pointer with any byte memOffset
// already folded in via GEP, ready to feed to `CreateLoad` / `CreateStore`.
// `memOffset` is retained for debugging / diagnostics only -- callers
// should not re-GEP it onto `ptr`.
struct FlatAddr {
  llvm::Value *Ptr = nullptr;
  int64_t MemOffset = 0;
  bool HasSaddr = false;
  // For stores: the VGPR source register carrying the store data. Unused
  // (OTHER kind) for loads.
  ParsedReg StData;
};

// Decode a GLOBAL_LOAD addressing operand shape. Recognised forms:
//
//   plain form: vaddr(VGPR64), [imms...]
//   SADDR form: saddr(SGPR64), vaddr(VGPR32), [imms...]
//
// `elemBytes` is the access element size (used only when
// `di.hasScaleOffset` is set -- then the per-lane VGPR vaddr is
// multiplied by `elemBytes` before being added to the SGPR base).
// `diagLabel` is used in the error message if neither form matches
// (e.g. `"GLOBAL_LOAD sub-dword"`).
//
// Returns an error on unrecognised shapes.
llvm::Expected<FlatAddr> decodeGlobalLoadAddr(RaiseContext &Ctx,
                                              const DecodedInst &Di,
                                              OpResolver &Op, int ElemBytes,
                                              llvm::StringRef DiagLabel);

// Decode a GLOBAL_STORE addressing operand shape. Recognised forms:
//
//   plain form: vaddr(VGPR64), vdata(VGPR*), [imms...]
//   SADDR form: vaddr(VGPR32), vdata(VGPR*), saddr(SGPR64), [imms...]
//
// On success, `.stData` is populated with the vdata register. Other
// behaviour matches `decodeGlobalLoadAddr`.
llvm::Expected<FlatAddr> decodeGlobalStoreAddr(RaiseContext &Ctx,
                                               const DecodedInst &Di,
                                               OpResolver &Op, int ElemBytes,
                                               llvm::StringRef DiagLabel);

// Decode the first immediate in a GLOBAL/FLAT memory operand tail as the signed
// byte offset. Later immediates are encoding flags (`cpol`, TH, scope) and are
// deliberately ignored. MC may surface the signed 24-bit field as an unsigned
// bit pattern, so the helper sign-extends before callers materialise a GEP.
llvm::Expected<int64_t> getGlobalFlatOffset(const DecodedInst &Di);

} // namespace COMGR::hotswap

#endif
