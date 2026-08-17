//===- reg-file.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_REG_FILE_H
#define HOTSWAP_TRANSPILER_REG_FILE_H

#include "hotswap/decoder/parsed-reg.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <array>

namespace llvm {
class MCRegisterInfo;
} // namespace llvm

namespace COMGR::hotswap {

class ISAProfile;
class WaveProjection;

// Per-register alloca-based representation of the AMDGPU register file. Every
// architectural register gets its own alloca (i32, or i1 for VCC/SCC, i32/i64
// for EXEC depending on wave width). Callers emit straight-line loads/stores;
// the raiser later runs `PromoteMemToReg` to lift the allocas to SSA. Storage
// is sized at `init()` time from the live subtarget's register classes, except
// the VGPR/AGPR banks (see `KVGPRCap`).
struct AllocaRegFile {
  // VGPR/AGPR bank size. Not taken from the VGPR_32 register class: gfx1250
  // S_SET_VGPR_MSB extends the addressable VGPR index range to 1024 (an 8-bit
  // base plus a 2-bit MSB pair contributing value*256, so 255 + 3*256 = 1023),
  // and the raiser needs storage for every reachable index.
  static constexpr unsigned KVGPRCap = 1024;

  llvm::SmallVector<llvm::AllocaInst *> Sgpr;
  llvm::SmallVector<llvm::AllocaInst *> Vgpr;
  llvm::SmallVector<llvm::AllocaInst *> Agpr;
  llvm::SmallVector<llvm::AllocaInst *> Ttmp;
  llvm::AllocaInst *Vcc = nullptr;
  // Wave32-source scratch slot for VCC_HI. On a wave32 source VCC_HI is a free
  // scalar; on a wave64 source it is a real half of the VCC mask and routes
  // through Vcc.
  llvm::AllocaInst *VccHiScratch = nullptr;
  // Wave32-source scratch slot for EXEC_HI, symmetric with VccHiScratch.
  llvm::AllocaInst *ExecHiScratch = nullptr;
  llvm::AllocaInst *Scc = nullptr;
  llvm::AllocaInst *Exec = nullptr;
  llvm::AllocaInst *M0 = nullptr;
  std::array<llvm::AllocaInst *, 2> FlatScr = {};

  // Non-owning cross-wave projection policy, used by the VCC read/write paths
  // (ballot for per-lane-i1 -> wave-mask and the inverse). Null when the reg
  // file is used without a projection (e.g. unit tests); the VCC wave-mask
  // paths are then unreachable.
  const WaveProjection *Projection = nullptr;

  // Initialise storage. `MRI` supplies the architectural SGPR_32 / TTMP_32
  // register-class sizes; `Isa.hasAgpr()` selects whether to allocate AGPR
  // slots; `Proj` is retained as a non-owning pointer for the VCC read/write
  // paths.
  void init(llvm::IRBuilder<> &B, llvm::Type *I32Ty, llvm::Type *I1Ty,
            const ISAProfile &Isa, const llvm::MCRegisterInfo &MRI,
            const WaveProjection &Proj);

  // Direct per-class store/load helpers. `Idx` must be in range for the
  // corresponding class; an out-of-range index is a raiser bug and
  // fatal-errors.
  void storeSGPR32(llvm::IRBuilder<> &B, unsigned Idx, llvm::Value *V);
  llvm::Value *loadSGPR32(llvm::IRBuilder<> &B, unsigned Idx);
  void storeSGPR64(llvm::IRBuilder<> &B, unsigned Idx, llvm::Value *V);
  llvm::Value *loadSGPR64(llvm::IRBuilder<> &B, unsigned Idx);
  void storeVGPR32(llvm::IRBuilder<> &B, unsigned Idx, llvm::Value *V);
  llvm::Value *loadVGPR32(llvm::IRBuilder<> &B, unsigned Idx);
  void storeVGPR64(llvm::IRBuilder<> &B, unsigned Idx, llvm::Value *V);
  llvm::Value *loadVGPR64(llvm::IRBuilder<> &B, unsigned Idx);
  void storeAGPR32(llvm::IRBuilder<> &B, unsigned Idx, llvm::Value *V);
  llvm::Value *loadAGPR32(llvm::IRBuilder<> &B, unsigned Idx);

  void storeVCC(llvm::IRBuilder<> &B, llvm::Value *V);
  llvm::Value *loadVCC(llvm::IRBuilder<> &B);
  void storeSCC(llvm::IRBuilder<> &B, llvm::Value *V);
  llvm::Value *loadSCC(llvm::IRBuilder<> &B);
  llvm::Value *loadExec(llvm::IRBuilder<> &B);
  void storeExec(llvm::IRBuilder<> &B, llvm::Value *V);

  // Read VCC as a wave-level bit-mask of width `ResultTy` through the wave
  // projection. Requires `init()` to have supplied a projection.
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

  // Populate `Out` with every alloca the raiser emitted, for feeding
  // into `PromoteMemToReg`.
  void collectAllocas(llvm::SmallVectorImpl<llvm::AllocaInst *> &Out) const;
};

} // namespace COMGR::hotswap

#endif
