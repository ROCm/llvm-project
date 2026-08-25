//===- op-resolver.h - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_OP_RESOLVER_H
#define HOTSWAP_TRANSPILER_OP_RESOLVER_H

#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/parsed-reg.h"
#include "hotswap/raiser/raise-context.h"

#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace COMGR::hotswap {

// Operand access for one decoded instruction, handed to a handler: source
// reads through the decoded srcMap at 32-bit, 64-bit or EXEC width, register
// names for sources and destinations, and immediates. Float source reads apply
// the neg/abs modifiers.
struct OpResolver {
  // Context the operands are read through.
  RaiseContext &Ctx;
  // Instruction whose operands are being read.
  const DecodedInst &Di;

  // MC operand index of the I-th source.
  unsigned srcIdx(unsigned I) const {
    assert(I < Di.SrcMap.size() && "source index out of range");
    return Di.SrcMap[I];
  }
  // Number of sources the instruction takes.
  unsigned nSrcs() const { return static_cast<unsigned>(Di.SrcMap.size()); }

  // Modifier bits attached to the I-th source, 0 when it carries none. Bit 0
  // negates and bit 1 takes the absolute value.
  unsigned srcMod(unsigned I) const;

  // Apply the I-th source's modifiers to V, which the caller has already read.
  // An integer-typed V round-trips through float, since the modifiers are
  // defined on the float interpretation of the bits.
  llvm::Value *applyMods(unsigned I, llvm::Value *V);

  // Read the I-th source as a 32-bit value.
  llvm::Expected<llvm::Value *> src(unsigned I) {
    return Ctx.registers().readOp32(Di, srcIdx(I));
  }
  // Read the I-th source as a 32-bit value with its modifiers applied.
  llvm::Expected<llvm::Value *> srcF(unsigned I);
  // Read the I-th source as a 64-bit value.
  llvm::Expected<llvm::Value *> src64(unsigned I) {
    return Ctx.registers().readOp64(Di, srcIdx(I));
  }
  // Read the I-th source as a wave mask at target EXEC width.
  llvm::Expected<llvm::Value *> srcExecWidth(unsigned I) {
    return Ctx.registers().readOpExecWidth(Di, srcIdx(I));
  }
  // Read the I-th source's per-lane wave-mask value, or null when unavailable.
  llvm::Expected<llvm::Value *> srcWaveMaskI1(unsigned I) {
    return Ctx.registers().readOpWaveMaskI1(Di, srcIdx(I));
  }
  // Value of the I-th source, which must be an immediate.
  int64_t srcImm(unsigned I) {
    unsigned Index = srcIdx(I);
    assert(Di.isImm(Index) && "source operand must be an immediate");
    return Di.getImm(Index);
  }

  // Register the I-th destination names.
  llvm::Expected<ParsedReg> dst(unsigned I = 0) {
    assert(Di.isReg(I) && "destination operand must be a register");
    return Ctx.registers().parseReg(Di, I);
  }
  // Whether the I-th source is a register rather than an immediate.
  bool isSrcReg(unsigned I) { return Di.isReg(srcIdx(I)); }
  // Register the I-th source names, or no value when it is an immediate.
  llvm::Expected<std::optional<ParsedReg>> srcReg(unsigned I);
};

} // namespace COMGR::hotswap

#endif
