//===- reg-file.cpp - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "reg-file.h"

#include "isa-profile.h"
#include "wave-projection.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::SGPR_32RegClassID, TTMP_32RegClassID

#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Human-readable name for a ParsedReg::Kind. Used only in fatal-error
// messages, so an accidentally-missed case here surfaces as a slightly
// uglier diagnostic (`"?<9>"`) rather than a silent bug.
const char *kindName(ParsedReg::Kind K) {
  switch (K) {
  case ParsedReg::SGPR:       return "SGPR";
  case ParsedReg::VGPR:       return "VGPR";
  case ParsedReg::AGPR:       return "AGPR";
  case ParsedReg::VCC:        return "VCC";
  case ParsedReg::EXEC:       return "EXEC";
  case ParsedReg::SCC:        return "SCC";
  case ParsedReg::MODE:       return "MODE";
  case ParsedReg::M0:         return "M0";
  case ParsedReg::FLAT_SCR:   return "FLAT_SCR";
  case ParsedReg::TTMP:       return "TTMP";
  case ParsedReg::LDS_DIRECT: return "LDS_DIRECT";
  case ParsedReg::SRC_VCCZ:   return "SRC_VCCZ";
  case ParsedReg::SRC_EXECZ:  return "SRC_EXECZ";
  case ParsedReg::SRC_SCC:    return "SRC_SCC";
  case ParsedReg::VCC_HI_SCRATCH: return "VCC_HI_SCRATCH";
  case ParsedReg::EXEC_HI_SCRATCH: return "EXEC_HI_SCRATCH";
  case ParsedReg::NOREG:      return "NOREG";
  case ParsedReg::OTHER:      return "OTHER";
  }
  return "<invalid>";
}

[[noreturn]] void failUnhandledKind(const char *Fn, ParsedReg Pr) {
  report_fatal_error(Twine("transpiler: ") + Fn +
                     ": unhandled ParsedReg Kind " + kindName(Pr.RegKind) +
                     " (baseIdx=" + Twine(Pr.BaseIdx) +
                     ", width=" + Twine(Pr.WidthInDwords) +
                     "). Caller must "
                     "handle NOREG / OTHER / MODE before dispatching "
                     "through readReg32/readReg64.");
}

} // namespace

void AllocaRegFile::init(IRBuilder<> &B, Type *I32Ty, Type *I1Ty,
                          const ISAProfile &Isa, const MCRegisterInfo &MRI,
                          const WaveProjection &Proj) {
  Projection = &Proj;
  // EXEC storage width is chosen by the projection. Modulo-replication
  // keeps it at source wave width (the long-standing default); wave-
  // native cross-widening widens it to the target hardware mask so
  // data-dependent EXEC writes survive the wave32 -> wave64 lift
  // without truncation. See `WaveProjection::execStorageTy` for the
  // contract and `wave-projection.cpp:WaveNativeProjection` for the
  // widened policy. `isa` continues to drive VGPR/AGPR bank sizing
  // below -- it's the *source* profile and thus the right place to
  // read `hasAGPR` from.
  ExecTy = Proj.execStorageTy();

  const unsigned NSgpr = MRI.getRegClass(AMDGPU::SGPR_32RegClassID).getNumRegs();
  Sgpr.assign(NSgpr, nullptr);
  for (unsigned I = 0; I < NSgpr; ++I)
    Sgpr[I] = B.CreateAlloca(I32Ty, nullptr, "Sgpr" + std::to_string(I));

  // VGPR storage is explicitly oversized relative to TableGen's VGPR_32
  // class (see `KVGPRCap` docs in reg-file.h). AGPR storage mirrors
  // the VGPR size because AGPRs share the same index space under MFMA
  // encoding conventions.
  Vgpr.assign(KVGPRCap, nullptr);
  for (unsigned I = 0; I < KVGPRCap; ++I)
    Vgpr[I] = B.CreateAlloca(I32Ty, nullptr, "Vgpr" + std::to_string(I));

  if (Isa.HasAgpr) {
    Agpr.assign(KVGPRCap, nullptr);
    for (unsigned I = 0; I < KVGPRCap; ++I)
      Agpr[I] = B.CreateAlloca(I32Ty, nullptr, "Agpr" + std::to_string(I));
  }

  // Condition-carrying scalar registers are initialised to zero so that a
  // read-before-write (raiser bug or unhandled instruction) yields a
  // deterministic "false / inactive" value rather than `undef`/poison.
  // Poison here would silently destroy SPE predication (the entry block
  // already reads EXEC, and VCC/SCC feed downstream branches). EXEC is
  // the only register initialised to all-ones -- that is the architectural
  // boot state of a dispatched wave and is load-bearing for every
  // subsequent `emitLaneActiveBit` call.
  Vcc = B.CreateAlloca(I1Ty, nullptr, "Vcc");
  B.CreateStore(ConstantInt::getFalse(I1Ty), Vcc);
  // Wave32-source VCC_HI scratch scalar (see ParsedReg::VCC_HI_SCRATCH).
  // Zero-initialised like the other condition scalars.
  VccHiScratch = B.CreateAlloca(I32Ty, nullptr, "VccHiScratch");
  B.CreateStore(ConstantInt::get(I32Ty, 0), VccHiScratch);
  // Wave32-source EXEC_HI scratch scalar (see ParsedReg::EXEC_HI_SCRATCH).
  // Symmetric with VccHiScratch; zero-initialised.
  ExecHiScratch = B.CreateAlloca(I32Ty, nullptr, "ExecHiScratch");
  B.CreateStore(ConstantInt::get(I32Ty, 0), ExecHiScratch);
  Scc = B.CreateAlloca(I1Ty, nullptr, "Scc");
  B.CreateStore(ConstantInt::getFalse(I1Ty), Scc);
  Exec = B.CreateAlloca(ExecTy, nullptr, "exec");
  // The initial EXEC value is projection-dependent. Default (same-wave
  // / modulo-replication): all-ones. Wave-native Wave32 -> Wave64
  // cross-widening: `@llvm.amdgcn.init_whole_wave` captures the
  // original per-lane active mask and forces hardware EXEC = -1 so the
  // WMMA -> MFMA cross-lane pipeline can run across all 64 Wave64
  // lanes even on a partial-wave dispatch. See
  // `WaveProjection::emitInitialExec` and
  // `WaveNativeProjection::emitInitialExec` for the correctness
  // argument and the rationale for superseding the earlier
  // `@llvm.amdgcn.strict.wwm`-per-MFMA-output strategy.
  B.CreateStore(Proj.emitInitialExec(B), Exec);
  M0 = B.CreateAlloca(I32Ty, nullptr, "M0");
  B.CreateStore(ConstantInt::get(I32Ty, 0), M0);
  FlatScr[0] = B.CreateAlloca(I32Ty, nullptr, "flat_scr_lo");
  FlatScr[1] = B.CreateAlloca(I32Ty, nullptr, "flat_scr_hi");

  const unsigned NTtmp = MRI.getRegClass(AMDGPU::TTMP_32RegClassID).getNumRegs();
  Ttmp.assign(NTtmp, nullptr);
  for (unsigned I = 0; I < NTtmp; ++I)
    Ttmp[I] = B.CreateAlloca(I32Ty, nullptr, "ttmp" + std::to_string(I));
}

void AllocaRegFile::storeSGPR32(IRBuilder<> &B, int Idx, Value *V) {
  Type *I32Ty = B.getInt32Ty();
  if (V->getType() != I32Ty)
    V = B.CreateBitCast(V, I32Ty);
  B.CreateStore(V, Sgpr[Idx]);
  // Invalidate any `V_CMP -> per-lane-i1` shadow entry for this SGPR.
  // Callers that wrote a wave-mask (V_CMP path) re-populate immediately
  // after; callers that wrote a scalar leave the entry empty so the
  // next consumer falls back to the narrow-mask extract. See
  // RaiseContext::lastSgprWaveMaskI1 / onSgprWritten contract.
  if (OnSgprWritten) OnSgprWritten(Idx);
}

namespace {

// Shared OOB check for the per-class load/store helpers. Fatal-errors
// instead of returning undef / crashing inside LLVM -- a caller that
// hands us an out-of-range `idx` has already produced an invalid
// ParsedReg, which is always a raiser bug rather than a kernel bug.
[[noreturn]] void failOOB(const char *Fn, int Idx, size_t Cap) {
  report_fatal_error(Twine("transpiler: ") + Fn + " idx=" + Twine(Idx) +
                     " out of range [0.." + Twine(Cap) +
                     ") or null; raiser bug, not a kernel bug -- the "
                     "caller produced an invalid ParsedReg.");
}

template <typename BankT>
void assertInBank(const char *Fn, const BankT &Bank, int Idx) {
  if (Idx < 0 || static_cast<unsigned>(Idx) >= Bank.size() || !Bank[Idx])
    failOOB(Fn, Idx, Bank.size());
}

// 64-bit reads touch idx and idx+1; both must be in range.
template <typename BankT>
void assertPairInBank(const char *Fn, const BankT &Bank, int Idx) {
  if (Idx < 0 || static_cast<unsigned>(Idx + 1) >= Bank.size() ||
      !Bank[Idx] || !Bank[Idx + 1])
    failOOB(Fn, Idx, Bank.size());
}

} // namespace

Value *AllocaRegFile::loadSGPR32(IRBuilder<> &B, int Idx) {
  assertInBank("loadSGPR32", Sgpr, Idx);
  return B.CreateLoad(B.getInt32Ty(), Sgpr[Idx]);
}

void AllocaRegFile::storeSGPR64(IRBuilder<> &B, int Idx, Value *V) {
  assertPairInBank("storeSGPR64", Sgpr, Idx);
  Type *I32Ty = B.getInt32Ty();
  Type *I64Ty = B.getInt64Ty();
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, I64Ty);
  if (V->getType() != I64Ty)
    V = B.CreateBitCast(V, I64Ty);
  Value *Lo = B.CreateTrunc(V, I32Ty);
  Value *Hi = B.CreateTrunc(B.CreateLShr(V, 32), I32Ty);
  B.CreateStore(Lo, Sgpr[Idx]);
  B.CreateStore(Hi, Sgpr[Idx + 1]);
  // Pair writes invalidate both halves of the shadow. The V_CMP
  // wave-mask write path keys its `recordSgprWaveMaskI1` on the low
  // index of the destination pair (on wave64 source this is the
  // authoritative key; on wave32 source `storeSGPR64` is not reachable
  // from V_CMP because the source SGPR is physically 32 bits). See
  // the `onSgprWritten` contract in reg-file.h.
  if (OnSgprWritten) {
    OnSgprWritten(Idx);
    OnSgprWritten(Idx + 1);
  }
}

Value *AllocaRegFile::loadSGPR64(IRBuilder<> &B, int Idx) {
  assertPairInBank("loadSGPR64", Sgpr, Idx);
  Type *I32Ty = B.getInt32Ty();
  Type *I64Ty = B.getInt64Ty();
  Value *Lo = B.CreateZExt(B.CreateLoad(I32Ty, Sgpr[Idx]), I64Ty);
  Value *Hi = B.CreateZExt(B.CreateLoad(I32Ty, Sgpr[Idx + 1]), I64Ty);
  return B.CreateOr(Lo, B.CreateShl(Hi, 32));
}

void AllocaRegFile::storeVGPR32(IRBuilder<> &B, int Idx, Value *V) {
  assertInBank("storeVGPR32", Vgpr, Idx);
  Type *I32Ty = B.getInt32Ty();
  if (V->getType() != I32Ty) {
    if (V->getType()->isPointerTy())
      V = B.CreatePtrToInt(V, B.getInt64Ty());
    if (V->getType() == B.getFloatTy())
      V = B.CreateBitCast(V, I32Ty);
    else if (V->getType() != I32Ty)
      V = B.CreateTrunc(V, I32Ty);
  }
  B.CreateStore(V, Vgpr[Idx]);
}

Value *AllocaRegFile::loadVGPR32(IRBuilder<> &B, int Idx) {
  assertInBank("loadVGPR32", Vgpr, Idx);
  return B.CreateLoad(B.getInt32Ty(), Vgpr[Idx]);
}

void AllocaRegFile::storeVGPR64(IRBuilder<> &B, int Idx, Value *V) {
  assertPairInBank("storeVGPR64", Vgpr, Idx);
  Type *I32Ty = B.getInt32Ty();
  Type *I64Ty = B.getInt64Ty();
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, I64Ty);
  if (V->getType() != I64Ty)
    V = B.CreateBitCast(V, I64Ty);
  Value *Lo = B.CreateTrunc(V, I32Ty);
  Value *Hi = B.CreateTrunc(B.CreateLShr(V, 32), I32Ty);
  B.CreateStore(Lo, Vgpr[Idx]);
  B.CreateStore(Hi, Vgpr[Idx + 1]);
}

Value *AllocaRegFile::loadVGPR64(IRBuilder<> &B, int Idx) {
  assertPairInBank("loadVGPR64", Vgpr, Idx);
  Type *I32Ty = B.getInt32Ty();
  Type *I64Ty = B.getInt64Ty();
  Value *Lo = B.CreateZExt(B.CreateLoad(I32Ty, Vgpr[Idx]), I64Ty);
  Value *Hi = B.CreateZExt(B.CreateLoad(I32Ty, Vgpr[Idx + 1]), I64Ty);
  return B.CreateOr(Lo, B.CreateShl(Hi, 32));
}

void AllocaRegFile::storeAGPR32(IRBuilder<> &B, int Idx, Value *V) {
  assertInBank("storeAGPR32", Agpr, Idx);
  Type *I32Ty = B.getInt32Ty();
  if (V->getType() != I32Ty)
    V = B.CreateBitCast(V, I32Ty);
  B.CreateStore(V, Agpr[Idx]);
}

Value *AllocaRegFile::loadAGPR32(IRBuilder<> &B, int Idx) {
  assertInBank("loadAGPR32", Agpr, Idx);
  return B.CreateLoad(B.getInt32Ty(), Agpr[Idx]);
}

void AllocaRegFile::storeVCC(IRBuilder<> &B, Value *V) {
  if (V->getType() != B.getInt1Ty())
    V = B.CreateICmpNE(V, Constant::getNullValue(V->getType()));
  B.CreateStore(V, Vcc);
}

Value *AllocaRegFile::loadVCC(IRBuilder<> &B) {
  return B.CreateLoad(B.getInt1Ty(), Vcc);
}

void AllocaRegFile::storeSCC(IRBuilder<> &B, Value *V) {
  if (V->getType() != B.getInt1Ty())
    V = B.CreateICmpNE(V, Constant::getNullValue(V->getType()));
  B.CreateStore(V, Scc);
}

Value *AllocaRegFile::loadSCC(IRBuilder<> &B) {
  return B.CreateLoad(B.getInt1Ty(), Scc);
}

Value *AllocaRegFile::loadExec(IRBuilder<> &B) {
  return B.CreateLoad(ExecTy, Exec, "exec_val");
}

void AllocaRegFile::storeExec(IRBuilder<> &B, Value *V) {
  if (V->getType() != ExecTy)
    V = B.CreateBitOrPointerCast(V, ExecTy);
  B.CreateStore(V, Exec);
  if (OnExecWritten)
    OnExecWritten();
}

Value *AllocaRegFile::readVCCAsWaveMask(IRBuilder<> &B, Type *ResultTy) {
  assert(Projection && "readVCCAsWaveMask requires a WaveProjection -- "
                        "call init() before using this reg-file");
  return Projection->ballotI1ToWidth(B, loadVCC(B), ResultTy, "vcc_ballot");
}

Value *AllocaRegFile::readReg32(IRBuilder<> &B, ParsedReg Pr) {
  if (Pr.RegKind == ParsedReg::SGPR) return loadSGPR32(B, Pr.BaseIdx);
  if (Pr.RegKind == ParsedReg::VGPR) return loadVGPR32(B, Pr.BaseIdx);
  if (Pr.RegKind == ParsedReg::AGPR) return loadAGPR32(B, Pr.BaseIdx);
  if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH)
    return B.CreateLoad(B.getInt32Ty(), VccHiScratch, "vcc_hi_scratch");
  if (Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH)
    return B.CreateLoad(B.getInt32Ty(), ExecHiScratch, "exec_hi_scratch");
  // VCC as a 32-bit scalar read: must go through the wave-mask ballot,
  // NOT a sign-extension of the local i1. Callers that want a per-lane
  // i1 (e.g. predicating a compute op) should call `loadVCC` directly.
  // This path is hit by handlers that bypass `RaiseContext::readOp32`
  // (which performs the same routing); keeping the routing centralised
  // here prevents silent miscompiles when a new handler adds a direct
  // `regs.readReg32(VCC)` call. See `readVCCAsWaveMask` for the ballot
  // invariant (must be emitted at wave-level / full EXEC).
  if (Pr.RegKind == ParsedReg::VCC)
    return readVCCAsWaveMask(B, B.getInt32Ty());
  if (Pr.RegKind == ParsedReg::EXEC) {
    Value *V = loadExec(B);
    Type *I32Ty = B.getInt32Ty();
    if (V->getType() == I32Ty)
      return V;
    // wave64 EXEC is i64; pick the correct half when reading a 32-bit
    // slice. width==2 reads are handled by readReg64 / readExecWidth;
    // this path is the width==1 case where baseIdx selects LO/HI.
    if (Pr.WidthInDwords >= 2)
      return B.CreateTrunc(V, I32Ty, "exec_lo");
    if (Pr.BaseIdx == 1)
      V = B.CreateLShr(V, 32, "exec_hi_shr");
    return B.CreateTrunc(V, I32Ty, Pr.BaseIdx == 1 ? "exec_hi" : "exec_lo");
  }
  if (Pr.RegKind == ParsedReg::SCC)
    return B.CreateZExt(loadSCC(B), B.getInt32Ty());
  if (Pr.RegKind == ParsedReg::M0)
    return B.CreateLoad(B.getInt32Ty(), M0, "m0_val");
  if (Pr.RegKind == ParsedReg::FLAT_SCR)
    return B.CreateLoad(B.getInt32Ty(), FlatScr[0], "fscr_val");
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx >= 0 &&
      static_cast<unsigned>(Pr.BaseIdx) < Ttmp.size())
    return B.CreateLoad(B.getInt32Ty(), Ttmp[Pr.BaseIdx], "ttmp_val");
  // GFX9 src_lds_direct (encoding 254): reads one dword from LDS at the
  // byte address in M0. There is NO auto-increment of M0 on GFX9 -- the
  // kernel manages M0 explicitly between reads. (GFX11+ DSDIR
  // `lds_direct_load` does auto-increment; if we ever raise GFX11+
  // kernels that use DSDIR, the increment must be modeled separately.)
  if (Pr.RegKind == ParsedReg::LDS_DIRECT) {
    auto *I32Ty = B.getInt32Ty();
    Value *Addr = B.CreateLoad(I32Ty, M0, "m0_lds_off");
    auto *LdsPtr = PointerType::get(I32Ty->getContext(), 3);
    Value *Ptr = B.CreateIntToPtr(Addr, LdsPtr, "lds_direct_ptr");
    return B.CreateLoad(I32Ty, Ptr, "lds_direct_val");
  }
  failUnhandledKind("readReg32", Pr);
}

Value *AllocaRegFile::readReg64(IRBuilder<> &B, ParsedReg Pr) {
  if (Pr.RegKind == ParsedReg::SGPR) return loadSGPR64(B, Pr.BaseIdx);
  if (Pr.RegKind == ParsedReg::VGPR) return loadVGPR64(B, Pr.BaseIdx);
  if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH)
    return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), VccHiScratch), B.getInt64Ty());
  if (Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH)
    return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), ExecHiScratch), B.getInt64Ty());
  // VCC as a 64-bit scalar read: route through the wave-mask ballot.
  // Previous implementations used `SExt(i1 -> i64)`, which replicates
  // the CURRENT LANE's VCC bit across all 64 bits -- a silent lie when
  // the consumer expects a wave-level mask (e.g. `s_and_b64 vcc, exec,
  // vcc`). All direct VCC reads must materialise the full per-lane
  // collection via `amdgcn.ballot`.
  if (Pr.RegKind == ParsedReg::VCC)
    return readVCCAsWaveMask(B, B.getInt64Ty());
  if (Pr.RegKind == ParsedReg::EXEC) {
    Value *V = loadExec(B);
    if (V->getType() != B.getInt64Ty())
      V = B.CreateZExt(V, B.getInt64Ty(), "exec_ext");
    return V;
  }
  if (Pr.RegKind == ParsedReg::M0)
    return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), M0, "m0_val"),
                        B.getInt64Ty());
  if (Pr.RegKind == ParsedReg::FLAT_SCR) {
    Type *I32Ty = B.getInt32Ty();
    Type *I64Ty = B.getInt64Ty();
    Value *Lo = B.CreateZExt(B.CreateLoad(I32Ty, FlatScr[0]), I64Ty);
    Value *Hi = B.CreateZExt(B.CreateLoad(I32Ty, FlatScr[1]), I64Ty);
    return B.CreateOr(Lo, B.CreateShl(Hi, 32), "fscr64");
  }
  // TTMP[baseIdx:baseIdx+1] read as i64 -- combine the two adjacent
  // i32 lanes the same way SGPR pairs are combined. The 32-bit
  // single-lane read above lives in readReg32; this is the
  // S_LOAD_DWORDX2 / DWORDX4 path where the kernel addresses a TTMP
  // pair (e.g. `s_load_dwordx2 ..., ttmp[12:13], 0x0`). Without this
  // case the dispatcher fires `failUnhandledKind`, which the corpus
  // exercises in 49 gfx1250 kernels.
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx >= 0 &&
      static_cast<unsigned>(Pr.BaseIdx + 1) < Ttmp.size()) {
    Type *I32Ty = B.getInt32Ty();
    Type *I64Ty = B.getInt64Ty();
    Value *Lo = B.CreateZExt(B.CreateLoad(I32Ty, Ttmp[Pr.BaseIdx]), I64Ty);
    Value *Hi = B.CreateZExt(B.CreateLoad(I32Ty, Ttmp[Pr.BaseIdx + 1]),
                              I64Ty);
    return B.CreateOr(Lo, B.CreateShl(Hi, 32), "ttmp64");
  }
  failUnhandledKind("readReg64", Pr);
}

Value *AllocaRegFile::readExecWidth(IRBuilder<> &B) { return loadExec(B); }

void AllocaRegFile::writeExecWidth(IRBuilder<> &B, Value *V) {
  storeExec(B, V);
}

void AllocaRegFile::writeReg32(IRBuilder<> &B, ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::SGPR) { storeSGPR32(B, Pr.BaseIdx, V); return; }
  if (Pr.RegKind == ParsedReg::VGPR) { storeVGPR32(B, Pr.BaseIdx, V); return; }
  if (Pr.RegKind == ParsedReg::AGPR) { storeAGPR32(B, Pr.BaseIdx, V); return; }
  if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH ||
      Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH) {
    // Wave32-source VCC_HI / EXEC_HI are plain 32-bit data scalars (like an
    // SGPR), so they never receive a pointer -- those only arise on
    // address/control-flow registers (EXEC, SGPR pairs). Coerce any non-i32
    // scalar width to i32.
    assert(!V->getType()->isPointerTy() &&
           "VCC_HI_SCRATCH/EXEC_HI_SCRATCH is a data scalar; pointer writes are a raiser bug");
    assert(V->getType()->getPrimitiveSizeInBits() == 32 &&
           "VCC_HI_SCRATCH/EXEC_HI_SCRATCH is a 32-bit data scalar");
    if (!V->getType()->isIntegerTy(32))
      V = B.CreateBitCast(V, B.getInt32Ty());
    B.CreateStore(V, Pr.RegKind == ParsedReg::VCC_HI_SCRATCH ? VccHiScratch
      : ExecHiScratch);
    return;
  }
  if (Pr.RegKind == ParsedReg::EXEC) {
    Type *I32Ty = B.getInt32Ty();
    // Coerce incoming value to i32. storeExec handles width matching to
    // execTy; for wave64-native EXEC (execTy == i64) a 32-bit write
    // addresses only a half of the mask and is reconciled below.
    if (V->getType() != I32Ty) {
      if (V->getType()->isPointerTy())
        V = B.CreatePtrToInt(V, B.getInt64Ty());
      if (V->getType() != I32Ty) {
        unsigned Bits = V->getType()->getPrimitiveSizeInBits();
        V = Bits > 32 ? B.CreateTrunc(V, I32Ty)
                       : B.CreateBitCast(V, I32Ty);
      }
    }
    if (ExecTy == I32Ty || Pr.WidthInDwords >= 2) {
      storeExec(B, V);
      return;
    }
    // execTy is i64 and the write is a single-half EXEC_LO or EXEC_HI.
    // Two reconciliation policies apply depending on the projection:
    //
    //   (a) Architectural half-write (default for wave64 source): the
    //       source author named one 32-bit half, so preserve the other
    //       half's live value and merge. This is the shape wave64
    //       source kernels rely on (`s_mov_b32 exec_hi, sN` after an
    //       `s_mov_b32 exec_lo, sM`).
    //
    //   (b) Wave-native cross-widening broadcast (wave32 source ->
    //       wave64 target, per `projection->broadcastNarrowExecLoWrite()`):
    //       the source author's wave32 view treats `exec_lo` as the
    //       whole wave mask, so the 32-bit value represents the
    //       whole-wave intent. The wave-native projection models each
    //       target lane as an independent source-thread equivalent,
    //       which means a whole-wave write must fan out to every
    //       target lane; we replicate the 32-bit value into both
    //       halves of the widened EXEC. EXEC_HI writes cannot arise
    //       from a wave32 source (the source ISA has no EXEC_HI), and
    //       reaching one under wave-native is programmer error or a
    //       raiser bug, so we fall back to (a) in that case for
    //       robustness.
    Value *V64 = B.CreateZExt(V, ExecTy);
    if (Projection && Projection->broadcastNarrowExecLoWrite() &&
        Pr.BaseIdx == 0) {
      // Replicate: EXEC = (v << 32) | v. Equivalent to the
      // "broadcast wave32 whole-wave mask across both halves of the
      // widened EXEC" semantics.
      Value *Hi = B.CreateShl(V64, 32);
      Value *Merged = B.CreateOr(V64, Hi, "exec_lo_broadcast");
      storeExec(B, Merged);
      return;
    }
    Value *Cur = loadExec(B);
    Value *Merged;
    if (Pr.BaseIdx == 1) {
      Value *Mask = ConstantInt::get(ExecTy, 0xFFFFFFFFULL);
      Merged = B.CreateOr(B.CreateAnd(Cur, Mask),
                           B.CreateShl(V64, 32), "exec_hi_write");
    } else {
      Value *Mask = ConstantInt::get(ExecTy, 0xFFFFFFFF00000000ULL);
      Merged = B.CreateOr(B.CreateAnd(Cur, Mask), V64, "exec_lo_write");
    }
    storeExec(B, Merged);
    return;
  }
  if (Pr.RegKind == ParsedReg::VCC) {
    assert(Projection && "writeReg32(VCC) requires a WaveProjection");
    storeVCC(B, Projection->extractLaneBitFromWaveMask(B, V));
    return;
  }
  if (Pr.RegKind == ParsedReg::M0) {
    if (V->getType() != B.getInt32Ty()) V = B.CreateBitCast(V, B.getInt32Ty());
    B.CreateStore(V, M0);
    return;
  }
  if (Pr.RegKind == ParsedReg::FLAT_SCR) {
    if (V->getType() != B.getInt32Ty()) V = B.CreateBitCast(V, B.getInt32Ty());
    B.CreateStore(V, FlatScr[0]);
    return;
  }
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx >= 0 &&
      static_cast<unsigned>(Pr.BaseIdx) < Ttmp.size()) {
    if (V->getType() != B.getInt32Ty()) V = B.CreateBitCast(V, B.getInt32Ty());
    B.CreateStore(V, Ttmp[Pr.BaseIdx]);
    return;
  }
}

void AllocaRegFile::writeReg64(IRBuilder<> &B, ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::SGPR) { storeSGPR64(B, Pr.BaseIdx, V); return; }
  if (Pr.RegKind == ParsedReg::VGPR) { storeVGPR64(B, Pr.BaseIdx, V); return; }
  if (Pr.RegKind == ParsedReg::VCC) {
    assert(Projection && "writeReg64(VCC) requires a WaveProjection");
    storeVCC(B, Projection->extractLaneBitFromWaveMask(B, V));
    return;
  }
  if (Pr.RegKind == ParsedReg::EXEC) {
    storeExec(B, V);
    return;
  }
  if (Pr.RegKind == ParsedReg::FLAT_SCR) {
    Type *I32Ty = B.getInt32Ty();
    Type *I64Ty = B.getInt64Ty();
    if (V->getType() != I64Ty) V = B.CreateBitOrPointerCast(V, I64Ty);
    B.CreateStore(B.CreateTrunc(V, I32Ty), FlatScr[0]);
    B.CreateStore(B.CreateTrunc(B.CreateLShr(V, 32), I32Ty), FlatScr[1]);
    return;
  }
  // 64-bit TTMP write -- split into two i32 stores at baseIdx and
  // baseIdx+1, matching the readReg64 TTMP shape above. Trap-handler
  // kernels routinely materialise a 64-bit address into a TTMP pair
  // before invoking the trap; we route them through the same alloca
  // bank as the 32-bit case so subsequent reads see the stored value.
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx >= 0 &&
      static_cast<unsigned>(Pr.BaseIdx + 1) < Ttmp.size()) {
    Type *I32Ty = B.getInt32Ty();
    Type *I64Ty = B.getInt64Ty();
    if (V->getType() != I64Ty) V = B.CreateBitOrPointerCast(V, I64Ty);
    B.CreateStore(B.CreateTrunc(V, I32Ty), Ttmp[Pr.BaseIdx]);
    B.CreateStore(B.CreateTrunc(B.CreateLShr(V, 32), I32Ty),
                   Ttmp[Pr.BaseIdx + 1]);
    return;
  }
}

void AllocaRegFile::writeRegExecWidth(IRBuilder<> &B, ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::SGPR) {
    // SGPR storage is sized by the *source* architecture: one 32-bit
    // SGPR on wave32 source (`s_and_saveexec_b32 s0, ...`), a 64-bit
    // SGPR pair on wave64 source. Under modulo-replication those
    // widths match `execTy` and no bridging is needed; under wave-
    // native cross-widening `execTy` is `WaveMaskTy` (i64 target
    // hardware) while the source-named SGPR is still 32-bit, so the
    // incoming EXEC-width value must be narrowed to the source width
    // before the store. See `WaveProjection::sourceWaveMaskTy` for the
    // width policy. This is the symmetric counterpart of the widen-by-
    // replication done when the value was first read via
    // `readOpExecWidth` or computed by the `V_CMP -> SGPR` ballot.
    Type *SourceWidthTy =
        (Projection && Projection->sourceWaveScopedLaneOps() &&
         Pr.WidthInDwords >= 2)
            ? B.getInt64Ty()
            : (Projection ? Projection->sourceWaveMaskTy() : ExecTy);
    if (V->getType() != SourceWidthTy) {
      unsigned Have = V->getType()->getPrimitiveSizeInBits();
      unsigned Want = SourceWidthTy->getPrimitiveSizeInBits();
      if (Have > Want)
        V = B.CreateTrunc(V, SourceWidthTy, "wn_exec_to_src_mask");
      else if (Have < Want)
        V = B.CreateZExt(V, SourceWidthTy, "wn_exec_to_src_mask");
    }
    if (SourceWidthTy == B.getInt32Ty())
      storeSGPR32(B, Pr.BaseIdx, V);
    else
      storeSGPR64(B, Pr.BaseIdx, V);
    return;
  }
  if (Pr.RegKind == ParsedReg::VCC) {
    assert(Projection && "writeRegExecWidth(VCC) requires a WaveProjection");
    storeVCC(B, Projection->extractLaneBitFromWaveMask(B, V));
    return;
  }
  if (Pr.RegKind == ParsedReg::EXEC) {
    storeExec(B, V);
    return;
  }
}

Value *AllocaRegFile::readRegVec(IRBuilder<> &B, ParsedReg Pr, Type *VecTy) {
  unsigned N = VecTy->isVectorTy()
      ? cast<FixedVectorType>(VecTy)->getNumElements()
      : 1;
  Type *ElemTy = VecTy->isVectorTy()
      ? cast<FixedVectorType>(VecTy)->getElementType()
      : VecTy;
  unsigned DwordsPerElem = ElemTy->getPrimitiveSizeInBits() / 32;
  if (DwordsPerElem == 0) DwordsPerElem = 1;

  if (N == 1 && !VecTy->isVectorTy() && VecTy->getPrimitiveSizeInBits() <= 32) {
    Value *V = readReg32(B, Pr);
    if (V->getType() != VecTy) V = B.CreateBitCast(V, VecTy);
    return V;
  }

  unsigned TotalDwords = 0;
  if (ElemTy->isFloatTy()) TotalDwords = N;
  else if (ElemTy->isIntegerTy(32)) TotalDwords = N;
  else if (ElemTy->isHalfTy()) TotalDwords = (N + 1) / 2;
  else TotalDwords = (N * ElemTy->getPrimitiveSizeInBits() + 31) / 32;

  SmallVector<Value *> Dwords;
  for (unsigned I = 0; I < TotalDwords; I++) {
    ParsedReg Sub = Pr;
    Sub.BaseIdx = Pr.BaseIdx + I;
    Sub.WidthInDwords = 1;
    Dwords.push_back(readReg32(B, Sub));
  }

  unsigned TotalBits = TotalDwords * 32;
  Type *IntTy = Type::getIntNTy(B.getContext(), TotalBits);

  Value *Packed = ConstantInt::get(IntTy, 0);
  for (unsigned I = 0; I < TotalDwords; I++) {
    Value *Ext = B.CreateZExt(Dwords[I], IntTy);
    if (I > 0) Ext = B.CreateShl(Ext, I * 32);
    Packed = B.CreateOr(Packed, Ext);
  }
  return B.CreateBitCast(Packed, VecTy);
}

void AllocaRegFile::writeRegVec(IRBuilder<> &B, ParsedReg Pr, Value *V) {
  Type *Ty = V->getType();
  unsigned TotalBits = Ty->getPrimitiveSizeInBits();
  unsigned TotalDwords = (TotalBits + 31) / 32;

  Type *IntTy = Type::getIntNTy(B.getContext(), TotalDwords * 32);
  Type *I32Ty = B.getInt32Ty();
  Value *Packed = B.CreateBitCast(V, IntTy);

  for (unsigned I = 0; I < TotalDwords; I++) {
    Value *Dw;
    if (I == 0)
      Dw = B.CreateTrunc(Packed, I32Ty);
    else
      Dw = B.CreateTrunc(B.CreateLShr(Packed, I * 32), I32Ty);
    ParsedReg Sub = Pr;
    Sub.BaseIdx = Pr.BaseIdx + I;
    Sub.WidthInDwords = 1;
    writeReg32(B, Sub, Dw);
  }
}

void AllocaRegFile::collectAllocas(SmallVectorImpl<AllocaInst *> &Out) {
  for (auto *A : Sgpr) if (A) Out.push_back(A);
  for (auto *A : Vgpr) if (A) Out.push_back(A);
  for (auto *A : Agpr) if (A) Out.push_back(A);
  if (Vcc) Out.push_back(Vcc);
  // VccHiScratch (the wave32-source VCC_HI scratch scalar) must be promoted
  // too. As with the ttmps below, a surviving private alloca is moved to local
  // data share (LDS) by the AMDGPU backend's AMDGPUPromoteAllocaToLDS, where it
  // can alias the kernel's own LDS and be clobbered. Its dominating entry store
  // (init() above) lets PromoteMemToReg lift it cleanly.
  if (VccHiScratch) Out.push_back(VccHiScratch);
  // ExecHiScratch is promoted for the same reason as VccHiScratch above.
  if (ExecHiScratch) Out.push_back(ExecHiScratch);
  if (Scc) Out.push_back(Scc);
  if (Exec) Out.push_back(Exec);
  if (M0) Out.push_back(M0);
  for (auto *A : FlatScr) if (A) Out.push_back(A);
  // ttmps must be promoted too, otherwise they survive into the AMDGPU
  // backend and trigger AMDGPUPromoteAllocaToLDS, which inserts an
  // `amdgcn.dispatch.ptr` intrinsic *and* removes the kernel's
  // `amdgpu-no-dispatch-ptr` attribute (see
  // AMDGPUPromoteAlloca.cpp::getLocalSizeYZ). Once dispatch_ptr is
  // re-enabled in the kernel descriptor, the regalloc treats s[0:1] as
  // a free preloaded SGPR and uses s1 as scratch -- corrupting buffer
  // pointer high words and producing the gfx1250 Triton SIGSEGV (R1).
  // Lifting these allocas requires a dominating defining store for
  // every read; raiser.cpp Phase 4 seeds every ttmp with `i32 0` in
  // the entry block so PromoteMemToReg can lift the rest cleanly.
  for (auto *A : Ttmp) if (A) Out.push_back(A);
}

} // namespace COMGR::hotswap
