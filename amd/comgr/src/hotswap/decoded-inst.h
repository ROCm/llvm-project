//===- decoded-inst.h - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_DECODED_INST_H
#define HOTSWAP_TRANSPILER_DECODED_INST_H

#include "amdgpu-formats.h"
#include "canonical-op.h"

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include <cstdint>
#include <optional>
#include <string>

namespace COMGR::hotswap {
struct VCmpMeta;
} // namespace COMGR::hotswap

namespace COMGR::hotswap {

inline std::optional<int64_t> evalOperandAsConst(const llvm::MCInst &Inst,
                                                 unsigned OpIdx) {
  if (OpIdx >= Inst.getNumOperands()) {
    return std::nullopt;
  }
  const llvm::MCOperand &Op = Inst.getOperand(OpIdx);
  if (Op.isImm()) {
    return Op.getImm();
  }
  if (Op.isExpr()) {
    int64_t Val = 0;
    if (Op.getExpr()->evaluateAsAbsolute(Val)) {
      return Val;
    }
  }
  return std::nullopt;
}

struct DecodedInst {
  std::string Mnemonic;
  std::string RawMnemonic;
  std::string FullText;
  llvm::MCInst Inst;
  CanonicalOp CanonOp = CanonicalOp::Unknown;
  unsigned NumDefs = 0;
  bool IsBranch = false;
  bool IsConditionalBranch = false;
  uint64_t Offset = 0;
  uint64_t Size = 0;

  uint64_t TsFlags = 0;
  // Non-null iff `canonOp == V_CMP || canonOp == V_CMPX`. Points into the
  // OpcodeMap side-table; stable for the lifetime of the map.
  const VCmpMeta *Vcmp = nullptr;
  bool DefsScc = false;
  bool DefsVcc = false;
  bool DefsExec = false;
  // `scale_offset` flag from the CPol operand (gfx12+ FLAT/GLOBAL). When
  // set on a global_load/store in SADDR form, the per-lane vaddr is
  // multiplied by the access element size before being added to the
  // SGPR base. Decoded from the MCInst's `cpol` operand bit
  // `AMDGPU::CPol::SCAL` at disassembly time so handlers can branch
  // on a decoded bit instead of scanning `fullText`.
  bool HasScaleOffset = false;

  // ── DPP modifier state (Class 2 DppCrossLane; see
  //    hotswap/docs/wave-size-translation.md §6) ──
  //
  // DPP is a src0-pathway cross-lane shuffle modifier. The original
  // `inst.getOpcode()` retains the `_dpp` suffix and its MCInstrDesc
  // advertises the DPP bit via `TSFlags & SIInstrFlags::DPP`; the
  // opcode_map canonicalises the CanonicalOp down to the base op, so handlers
  // dispatched by CanonicalOp do not see the DPP variant directly.
  //
  // `hasDpp` is set in decode.cpp only for the DPP16 forms whose
  // modifier operands can be represented by `llvm.amdgcn.update.dpp`;
  // DPP8 also carries `TSFlags & SIInstrFlags::DPP`, but deliberately
  // leaves `hasDpp` false so the obstruction classifier can refuse it
  // as pending rather than routing it through the DPP16 intrinsic. The
  // modifier-operand fields below are populated by looking up their
  // named-operand indices in the original (non-canonicalised)
  // MCInstrDesc.
  //
  // Handler contract: `OpResolver::src(0)` / `srcF(0)` / `src64(0)`
  // transparently wrap their result through `emitUpdateDpp` when
  // `hasDpp` is true, using these fields as the intrinsic's
  // `dpp_ctrl` / `row_mask` / `bank_mask` / `bound_ctrl` immediates.
  // Handlers therefore need no per-op DPP awareness.
  //
  // `fi` (fetch-inactive / fetch-invalid) is an encoding-level flag
  // present on DPP16 and DPP8. `llvm.amdgcn.update.dpp` exposes the
  // DPP16 operand set without FI, so DPP16 FI sites must refuse loudly
  // before or during DPP wrapping rather than being silently lowered
  // as ordinary bound_ctrl DPP. DPP8 remains pending as a separate
  // lift family.
  bool HasDpp = false;
  uint16_t DppCtrl = 0;
  uint8_t DppRowMask = 0xF;
  uint8_t DppBankMask = 0xF;
  bool DppBoundCtrl = false;
  bool DppFi = false;

  // ── ds_swizzle_b32 imm state (Class 2 DsSwizzle; see
  //    hotswap/docs/wave-size-translation.md §6) ──
  //
  // The 16-bit `OpName::offset` immediate of `ds_swizzle_b32` encodes
  // a swizzle-mode selector + per-mode parameters (SIDefines.h
  // `Swizzle::EncBits`). Both the Phase 1.4.5 obstruction classifier
  // and `handle-ds.cpp::DS_SWIZZLE_B32` need this value: the
  // classifier to gate cross-wave safety via `dsSwizzleSafeForModRep`,
  // the handler to materialise the `i32 immarg` for
  // `llvm.amdgcn.ds.swizzle`. We extract once at decode time so both
  // consumers share a single canonical value (mirrors the
  // `hasDpp` / `dppCtrl` block above; same `decodeDsSwizzleImm`
  // pattern as `decodeDppModifiers`).
  //
  // Sentinel: when `canonOp != DS_SWIZZLE_B32`, `hasDsSwizzleImm`
  // stays false and `dsSwizzleImm` is meaningless. The decoder
  // refuses to set the field if the operand is missing,
  // non-immediate, or outside the unsigned 16-bit range -- same
  // soundness contract as the classifier extractor (an out-of-range
  // value silently truncated to uint16_t could land in either the
  // QUAD_PERM or BITMASK_PERM safe envelope and cause a silent
  // miscompile, so we refuse to populate it instead).
  bool HasDsSwizzleImm = false;
  uint16_t DsSwizzleImm = 0;

  // ── VOPD structural decode ────────────────────────────────────────
  //
  // VOPD packets contain two VALU component instructions sharing one
  // MCInst. The disassembler prints them as
  //   v_dual_<x> ... :: v_dual_<y> ...
  // but the raiser must not recover semantics by tokenizing that text.
  // The decoder populates this sidecar from LLVM's VOPD component tables
  // and MC operand indices; handle-vopd.cpp consumes only this typed view.
  struct VopdSource {
    enum class Kind : uint8_t {
      None,
      VGPR,
      AGPR,
      SGPR,
      TTMP,
      VCC,
      EXEC,
      SCC,
      M0,
      Imm,
    };
    Kind SrcKind = Kind::None;
    // Original MC operand index. Kept for diagnostics / drift checks.
    unsigned OperandIndex = 0;
    // Original MC register id for register-like kinds.
    unsigned Reg = 0;
    // Logical register-file index for register-like kinds.
    int BaseIdx = -1;
    int Width = 1;
    // Raw immediate when kind == Imm. The component CanonicalOp determines
    // whether the bit pattern is interpreted as integer bits or f32 bits.
    int64_t Imm = 0;
    // VOPD3 source modifier bits (same low-bit neg / abs contract that
    // VOP3 source modifiers use). Zero for VOPD1/2 and unmodified sources.
    uint8_t Modifiers = 0;
  };

  struct VopdHalf {
    CanonicalOp CanonOp = CanonicalOp::Unknown;
    unsigned ComponentOpcode = 0;
    unsigned DstReg = 0;
    VopdSource Src[3] = {};
    unsigned NumSrcs = 0;
    bool HasSrc2Acc = false;
    bool IsVoP3 = false;
    bool HasBitOp3 = false;
    uint8_t BitOp3 = 0;
  };

  bool HasVopd = false;
  bool IsVopd3 = false;
  VopdHalf Vopd[2] = {};

  unsigned FirstSrcIdx = 0;

  // Upper bound on the logical-source count the raiser's walk can produce.
  // Actual value is conservatively sized so it never clips any AMDGPU opcode
  // LLVM ships today; the bound is checked at MCState init time against the
  // widest `NumOperands - NumDefs` in MCInstrInfo, so a future LLVM that adds
  // a wider encoding will fatal at startup rather than silently truncate. See
  // `initMCState` for the check. If you bump the bound here, keep it as a
  // safe upper limit, not a tight fit: the startup assertion already makes
  // drift visible.
  static constexpr unsigned KMaxSrcs = 24;
  unsigned SrcMap[KMaxSrcs] = {};
  unsigned ModMap[KMaxSrcs] = {};
  unsigned NumSrcs = 0;

  unsigned numOps() const { return Inst.getNumOperands(); }
  bool isReg(unsigned I) const {
    return I < numOps() && Inst.getOperand(I).isReg();
  }
  bool isImm(unsigned I) const {
    return I < numOps() && Inst.getOperand(I).isImm();
  }
  unsigned getReg(unsigned I) const { return Inst.getOperand(I).getReg(); }
  int64_t getImm(unsigned I) const { return Inst.getOperand(I).getImm(); }
};

} // namespace COMGR::hotswap

#endif
