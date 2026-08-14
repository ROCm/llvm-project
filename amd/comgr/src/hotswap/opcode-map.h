//===- opcode-map.h - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_OPCODE_MAP_H
#define HOTSWAP_TRANSPILER_OPCODE_MAP_H

#include "canonical-op.h"
#include "llvm/IR/InstrTypes.h"

#include <cstdint>

namespace COMGR::hotswap {

// Side-table metadata for vector compare instructions (V_CMP_* / V_CMPX_*).
//
// The raiser collapses every V_CMP_* and V_CMPX_* MC opcode onto two
// CanonicalOps (`V_CMP`, `V_CMPX`) to avoid enumerating ~100 near-identical
// cases in handlers. The actual predicate / element type / width carried in
// the pseudo name (`v_cmp_EQ_U32_e64` -> EQ, unsigned 32-bit) is lifted out
// by hotswap-tblgen and looked up by MC opcode at dispatch time.
struct VCmpMeta {
  // LLVM predicate to feed to `IRBuilder::CreateICmp` / `CreateFCmp`.
  // Ignored when `IsClass == true` (no predicate; the comparison is the
  // floating-point classification mask in src1).
  llvm::CmpInst::Predicate Pred;
  // Element width in bits: 16, 32, or 64.
  uint8_t Bits;
  // False -> integer compare (ICmp; `Pred` is `ICMP_*`).
  // True  -> float compare   (FCmp; `Pred` is `FCMP_*`).
  bool IsFloat;
  // True for the V_CMP_CLASS_F* / V_CMPX_CLASS_F* family. These are
  // NOT predicate compares -- src0 is a float operand and src1 is an
  // i32 mask of FP classes; the result lane bit is set iff src0
  // matches any class enabled in the mask. Lifts to
  // `llvm.amdgcn.class.f<bits>(src0, src1)` rather than CreateFCmp;
  // `Pred` is unused on class entries. See V_CMP_CLASS in the
  // gfx9+ AMDGPU ISA manual and the dispatch in handle-valu-vcmp.cpp.
  bool IsClass = false;
};

// Maps a raw LLVM MC opcode emitted by the AMDGPU disassembler to the
// architecture-neutral CanonicalOp the raiser dispatches on. Returns
// `CanonicalOp::Unknown` for opcodes the raiser does not model.
//
// This is a dense array index. All the work -- folding e32/e64, DPP, SDWA,
// SADDR and subtarget-specific real encodings onto a canonical pseudo, then
// matching that pseudo against the table in `CanonicalOpcodes.td` -- happens
// at build time in hotswap-tblgen. See CanonicalOpcodeEmitter.cpp.
CanonicalOp canonicalOpFor(unsigned Opcode);

// Vector-compare metadata lookup. Returns nullptr if `Opcode` is not a
// V_CMP_* or V_CMPX_* instruction. Callers should only query this when
// `canonicalOpFor(Opcode)` returned `CanonicalOp::V_CMP` or
// `CanonicalOp::V_CMPX`.
const VCmpMeta *vcmpMetaFor(unsigned Opcode);

} // namespace COMGR::hotswap

#endif
