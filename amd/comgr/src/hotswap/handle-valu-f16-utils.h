//===- handle-valu-f16-utils.h - F16 VALU helpers -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared VOP3 F16 modifier helpers for VALU handlers.
//
// AMDGPU VOP3 F16 encodings carry source arithmetic modifiers and true16
// half-select bits in the decoded `srcN_modifiers` operands. For true16 forms,
// src0's modifier word also carries the destination-half selector; handlers
// must merge the 16-bit result into that half while preserving the other half
// of the destination VGPR.
//
// The helpers below provide both strict VOP3-only reads (modifier operand must
// be present) and e32/e64-shared reads (missing modifier operand means the
// default low-half, unmodified form). Unsupported modifier bits are refused
// rather than silently ignored.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H
#define HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H

#include "handle-valu-internal.h"

#include "llvm/ADT/StringRef.h"

namespace COMGR::hotswap {

bool readRequiredVOP3F16SrcMods(const DecodedInst &Di, HandlerResult &Hr,
                                unsigned SrcIndex, llvm::StringRef OpName,
                                unsigned &Mods);

// Like `readRequiredVOP3F16SrcMods`, but accepts an absent modifier operand as
// the default VOP1/e32 shape (`Mods = 0`). This is for CanonicalOps that share a
// handler across e32 and e64 encodings.
bool readOptionalVOP3F16SrcMods(const DecodedInst &Di, HandlerResult &Hr,
                                unsigned SrcIndex, llvm::StringRef OpName,
                                unsigned &Mods);

// Read an F16 source selected by the required VOP3 modifier operand.
llvm::Value *readOpSelF16(RaiseContext &Ctx, const DecodedInst &Di,
                          OpResolver &Op, HandlerResult &Hr,
                          unsigned SrcIndex, llvm::StringRef OpName);

// Read an F16 source selected by an optional modifier operand. Missing modifier
// operands select the low half with no abs/neg.
llvm::Value *readOptionalOpSelF16(RaiseContext &Ctx, const DecodedInst &Di,
                                  OpResolver &Op, HandlerResult &Hr,
                                  unsigned SrcIndex, llvm::StringRef OpName);

// Decode src0's true16 destination-half selector from the required VOP3
// modifier operand.
bool readVOP3F16DstHigh(const DecodedInst &Di, HandlerResult &Hr,
                        llvm::StringRef OpName, bool &DstHigh);

// Merge a raw i16 result into the selected destination half using the same IR
// name for either half. The unselected half is preserved by reading the old
// destination VGPR value.
void writeSelectedI16Bits(RaiseContext &Ctx, ParsedReg Dst,
                          llvm::Value *Result, bool DstHigh,
                          llvm::StringRef MergeName);
// Merge a raw i16 result into the selected destination half, using
// half-specific IR names for the low/high merge sites. The unselected half is
// preserved by reading the old destination VGPR value.
void writeSelectedI16Bits(RaiseContext &Ctx, ParsedReg Dst,
                          llvm::Value *Result, bool DstHigh,
                          llvm::StringRef MergeLoName,
                          llvm::StringRef MergeHiName);

// Merge the F16 result into the selected destination half. The unselected half
// is preserved by reading the old destination VGPR value.
void writeOpSelF16(RaiseContext &Ctx, OpResolver &Op, llvm::Value *Result,
                   bool DstHigh, llvm::StringRef MergeLoName = "f16_merge_lo",
                   llvm::StringRef MergeHiName = "f16_merge_hi");

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H
