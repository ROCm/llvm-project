//===- reg-file.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_REG_FILE_H
#define HOTSWAP_TRANSPILER_REG_FILE_H

#include "parsed-reg.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
class MCRegisterInfo;
} // namespace llvm

namespace COMGR::hotswap {

struct ISAProfile;
class WaveProjection;

// Per-register alloca-based representation of the AMDGPU register file.
//
// Every architectural register gets its own i32 alloca (or i1 for VCC/SCC,
// i32/i64 for EXEC depending on wave width). `PromoteMemToReg` in
// `raiser.cpp` Phase 6 lifts these allocas to proper SSA with phis after
// the handler pass finishes. Keeping per-register state in allocas lets
// handlers emit straight-line loads/stores during dispatch without
// worrying about SSA construction.
//
// Storage is sized at `init()` time:
//
//   * SGPR bank: `MRI.getRegClass(AMDGPU::SGPR_32RegClassID).getNumRegs()`
//     -- the authoritative count from TableGen for the live subtarget (106
//     on every AMDGPU subtarget today).
//   * TTMP bank: `MRI.getRegClass(AMDGPU::TTMP_32RegClassID).getNumRegs()`
//     -- 16 on every AMDGPU subtarget.
//   * VGPR / AGPR bank: `KVGPRCap` (below). NOT sourced from the register
//     class because gfx1250 `S_SET_VGPR_MSB` extends the runtime-
//     addressable VGPR index range beyond the TableGen class size (256):
//     the raiser needs storage for every index a kernel can reach under
//     MSB replay, not just the indices the assembler names directly.
//     Keeping the cap explicit documents the extra storage as
//     intentional.
//
//     Sized to match gfx1250's `1024-addressable-vgprs` subtarget
//     feature (LLVM AMDGPU.td:1120 `def 1024AddressableVGPRs`):
//     S_SET_VGPR_MSB encodes a 2-bit MSB pair per operand slot, each
//     contributing `value * 256` to that slot's VGPR index, so the
//     maximum reachable index is `255 (8-bit base) + 3*256 = 1023`.
//     A smaller cap (the previous 512) crashed legitimate gfx1250
//     tensilelite kernels that issue `s_set_vgpr_msb 0x80` followed
//     by `v_mov_b32 v72 /*v584*/` (vdst MSB=2 -> effective index
//     `72 + 512 = 584`); see `failOOB` in reg-file.cpp.
struct AllocaRegFile {
  // See class-level comment for rationale.
  static constexpr unsigned KVGPRCap = 1024;

  llvm::SmallVector<llvm::AllocaInst *> Sgpr;
  llvm::SmallVector<llvm::AllocaInst *> Vgpr;
  llvm::SmallVector<llvm::AllocaInst *> Agpr;
  llvm::SmallVector<llvm::AllocaInst *> Ttmp;
  llvm::AllocaInst *Vcc = nullptr;
  // Wave32-source scratch slot for the VCC_HI register (see ParsedReg::
  // VCC_HI_SCRATCH). Only used when the source ISA is wave32; on wave64
  // sources VCC_HI is a real half of the VCC mask and routes through Vcc.
  llvm::AllocaInst *VccHiScratch = nullptr;
  // Wave32-source scratch slot for the EXEC_HI register (see ParsedReg::
  // EXEC_HI_SCRATCH). Symmetric with VccHiScratch: only used when the source
  // ISA is wave32; on wave64 sources EXEC_HI is a real half of the EXEC mask
  // and routes through Exec.
  llvm::AllocaInst *ExecHiScratch = nullptr;
  llvm::AllocaInst *Scc = nullptr;
  llvm::AllocaInst *Exec = nullptr;
  llvm::AllocaInst *M0 = nullptr;
  llvm::AllocaInst *FlatScr[2] = {};

  // Width of the EXEC alloca -- tracks the *source* ISA wave width
  // (i32 on wave32 source, i64 on wave64 source). Distinct from the
  // target-hardware wave mask width owned by `WaveProjection`.
  llvm::Type *ExecTy = nullptr;

  // Non-owning pointer to the cross-wave projection policy. Used by
  // VCC read/write paths inside the reg file (ballot for per-lane-i1 ->
  // wave-mask, and the inverse for scalar -> per-lane stores). Left null
  // outside the raiser -- the reg file is used in contexts where no
  // projection exists (e.g. register-file-only unit tests or EXEC
  // initialisation at function entry) and any code path that would
  // need the projection (VCC wave-mask round-trip) is then unreachable.
  const WaveProjection *Projection = nullptr;

  // Invalidation hook fired on every EXEC-mutating store. The owning
  // RaiseContext installs this so its per-instruction `lane_active`
  // memo stays in sync regardless of which path hits `storeExec`.
  //
  // `unique_function` owns the callable (so the lambda's lifetime can
  // be decoupled from the stack frame that installs it) while still
  // being move-only -- a correctness nudge toward "installed exactly
  // once, not copied around."
  llvm::unique_function<void()> OnExecWritten;

  // Invalidation hook fired on every per-SGPR store, at the low-level
  // `storeSGPR32` / `storeSGPR64` boundary. The owning RaiseContext
  // installs this so its `lastSgprWaveMaskI1` shadow map (see raise_
  // context.h) stays in sync with every path that mutates an SGPR --
  // including paths that bypass `writeReg32 / writeReg64` to call
  // `storeSGPR32` directly (several handlers, e.g. handle-smem.cpp's
  // multi-dword SMEM load splitting, handle-valu.cpp's SCC-flag SGPR
  // writes).
  //
  // `storeSGPR64` fires the hook twice, once per half (idx, idx+1), so
  // pair writes invalidate both shadow entries. `storeSGPR32` fires it
  // once.
  //
  // The V_CMP wave-mask write path (`writeRegExecWidth` -> `storeSGPR32`/
  // `storeSGPR64`) ALSO fires this hook, which transiently empties the
  // shadow entry before the V_CMP handler immediately re-records the
  // per-lane `i1` via `ctx.recordSgprWaveMaskI1`. That two-step shape
  // is deliberate: the hook fires on every low-level store, and the
  // handler layer is where we distinguish "scalar write (leave map
  // empty)" from "wave-mask write (re-populate map with the i1)".
  //
  // See hotswap/docs/sgpr-wave-mask-translation.md section 3.1 for the
  // full invariants (I1 additive / I2 SSA-monotonic / I3 any
  // interference defeats the cache).
  llvm::unique_function<void(int)> OnSgprWritten;

  // Initialise storage.
  //
  // `MRI` is queried for the architectural SGPR_32 / TTMP_32 register-
  // class sizes. `ISAProfile::hasAGPR` selects whether to allocate
  // AGPR slots. `proj` is stored as a non-owning pointer for use by
  // VCC read/write paths.
  void init(llvm::IRBuilder<> &B, llvm::Type *I32Ty, llvm::Type *I1Ty,
            const ISAProfile &Isa, const llvm::MCRegisterInfo &MRI,
            const WaveProjection &Proj);

  // Direct per-class store/load helpers. All are predicated on `idx`
  // being in range for the corresponding class; bounds failures surface
  // via `errs()` + an undef fallback for reads, and are otherwise
  // caught at higher levels by the raiser's allocation pass.
  void storeSGPR32(llvm::IRBuilder<> &B, int Idx, llvm::Value *V);
  llvm::Value *loadSGPR32(llvm::IRBuilder<> &B, int Idx);
  void storeSGPR64(llvm::IRBuilder<> &B, int Idx, llvm::Value *V);
  llvm::Value *loadSGPR64(llvm::IRBuilder<> &B, int Idx);
  void storeVGPR32(llvm::IRBuilder<> &B, int Idx, llvm::Value *V);
  llvm::Value *loadVGPR32(llvm::IRBuilder<> &B, int Idx);
  void storeVGPR64(llvm::IRBuilder<> &B, int Idx, llvm::Value *V);
  llvm::Value *loadVGPR64(llvm::IRBuilder<> &B, int Idx);
  void storeAGPR32(llvm::IRBuilder<> &B, int Idx, llvm::Value *V);
  llvm::Value *loadAGPR32(llvm::IRBuilder<> &B, int Idx);

  void storeVCC(llvm::IRBuilder<> &B, llvm::Value *V);
  llvm::Value *loadVCC(llvm::IRBuilder<> &B);
  void storeSCC(llvm::IRBuilder<> &B, llvm::Value *V);
  llvm::Value *loadSCC(llvm::IRBuilder<> &B);
  llvm::Value *loadExec(llvm::IRBuilder<> &B);
  void storeExec(llvm::IRBuilder<> &B, llvm::Value *V);

  // Read VCC as a wave-level bit-mask of width `resultTy`. Routed
  // through `WaveProjection::ballotI1ToWidth`; see
  // `wave-projection.h` for the outer-EXEC-only invariant this
  // relies on.
  llvm::Value *readVCCAsWaveMask(llvm::IRBuilder<> &B, llvm::Type *ResultTy);

  // Generic read/write by ParsedReg.
  llvm::Value *readReg32(llvm::IRBuilder<> &B, ParsedReg Pr);
  llvm::Value *readReg64(llvm::IRBuilder<> &B, ParsedReg Pr);
  llvm::Value *readExecWidth(llvm::IRBuilder<> &B);
  void writeExecWidth(llvm::IRBuilder<> &B, llvm::Value *V);
  void writeReg32(llvm::IRBuilder<> &B, ParsedReg Pr, llvm::Value *V);
  void writeReg64(llvm::IRBuilder<> &B, ParsedReg Pr, llvm::Value *V);
  void writeRegExecWidth(llvm::IRBuilder<> &B, ParsedReg Pr, llvm::Value *V);

  // Read/write N dwords as a vector from contiguous VGPRs/AGPRs.
  llvm::Value *readRegVec(llvm::IRBuilder<> &B, ParsedReg Pr,
                          llvm::Type *VecTy);
  void writeRegVec(llvm::IRBuilder<> &B, ParsedReg Pr, llvm::Value *V);

  // Populate `out` with every alloca the raiser emitted, for feeding
  // into `PromoteMemToReg`.
  void collectAllocas(llvm::SmallVectorImpl<llvm::AllocaInst *> &Out);
};

} // namespace COMGR::hotswap

#endif
