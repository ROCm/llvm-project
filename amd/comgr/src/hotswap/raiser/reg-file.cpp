//===- reg-file.cpp - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/reg-file.h"

#include "hotswap/raiser/wave-projection.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Human-readable name for a ParsedReg::Kind, used only in fatal-error messages.
StringRef kindName(ParsedReg::Kind K) {
  switch (K) {
  case ParsedReg::SGPR:
    return "SGPR";
  case ParsedReg::VGPR:
    return "VGPR";
  case ParsedReg::AGPR:
    return "AGPR";
  case ParsedReg::VCC:
    return "VCC";
  case ParsedReg::EXEC:
    return "EXEC";
  case ParsedReg::SCC:
    return "SCC";
  case ParsedReg::MODE:
    return "MODE";
  case ParsedReg::M0:
    return "M0";
  case ParsedReg::FLAT_SCR:
    return "FLAT_SCR";
  case ParsedReg::TTMP:
    return "TTMP";
  case ParsedReg::LDS_DIRECT:
    return "LDS_DIRECT";
  case ParsedReg::SRC_VCCZ:
    return "SRC_VCCZ";
  case ParsedReg::SRC_EXECZ:
    return "SRC_EXECZ";
  case ParsedReg::SRC_SCC:
    return "SRC_SCC";
  case ParsedReg::VCC_HI_SCRATCH:
    return "VCC_HI_SCRATCH";
  case ParsedReg::EXEC_HI_SCRATCH:
    return "EXEC_HI_SCRATCH";
  case ParsedReg::NOREG:
    return "NOREG";
  case ParsedReg::OTHER:
    return "OTHER";
  }
  return "<invalid>";
}

// The base register index an indexed register kind requires. An absent index
// reaching an indexed kind is a raiser bug.
unsigned requireIndex(const ParsedReg &Pr) {
  assert(Pr.BaseIdx && "indexed register kind requires a base register index; "
                       "raiser produced an invalid ParsedReg");
  return *Pr.BaseIdx;
}

[[noreturn]] void failUnhandledKind(StringRef Fn, ParsedReg Pr) {
  report_fatal_error(
      Twine("transpiler: ") + Fn + ": unhandled ParsedReg Kind " +
      kindName(Pr.RegKind) +
      " (baseIdx=" + (Pr.BaseIdx ? Twine(*Pr.BaseIdx) : Twine("none")) +
      ", width=" + Twine(Pr.WidthInDwords) +
      "). Caller must handle NOREG / OTHER / MODE before "
      "dispatching through the reg-file read/write paths.");
}

// A reg-file index reaching a per-class helper out of range means the caller
// produced an invalid ParsedReg, which is always a raiser bug.
template <typename BankT>
void assertInBank([[maybe_unused]] const BankT &Bank,
                  [[maybe_unused]] unsigned Idx) {
  assert(Idx < Bank.size() && Bank[Idx] &&
         "reg-file access out of range; raiser produced an invalid ParsedReg");
}

// 64-bit accesses touch idx and idx+1; both must be in range.
template <typename BankT>
void assertPairInBank([[maybe_unused]] const BankT &Bank,
                      [[maybe_unused]] unsigned Idx) {
  assert(Idx + 1 < Bank.size() && Bank[Idx] && Bank[Idx + 1] &&
         "reg-file pair access out of range; raiser produced an invalid "
         "ParsedReg");
}

// Reinterpret a value as i32 for a 32-bit register slot. The value must
// already be 32 bits wide; only its type is changed.
Value *asI32(IRBuilder<> &B, Value *V) {
  Type *I32Ty = B.getInt32Ty();
  if (V->getType() == I32Ty)
    return V;
  assert(V->getType()->getPrimitiveSizeInBits() == 32 &&
         "32-bit register store expects a 32-bit value");
  return B.CreateBitCast(V, I32Ty);
}

// Coerce an arbitrary scalar or pointer to i32 for a VGPR slot: pointers are
// narrowed with ptrtoint, wider values truncated, equal-width values bitcast.
Value *coerceToI32(IRBuilder<> &B, Value *V) {
  Type *I32Ty = B.getInt32Ty();
  if (V->getType() == I32Ty)
    return V;
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, B.getInt64Ty());
  unsigned Bits = V->getType()->getPrimitiveSizeInBits();
  if (Bits == 32)
    return B.CreateBitCast(V, I32Ty);
  assert(Bits > 32 && "cannot widen a sub-32-bit value to a 32-bit slot");
  return B.CreateTrunc(V, I32Ty);
}

// Coerce a 64-bit value or pointer to i64 for splitting into a register pair.
Value *asI64(IRBuilder<> &B, Value *V) {
  Type *I64Ty = B.getInt64Ty();
  if (V->getType() == I64Ty)
    return V;
  if (V->getType()->isPointerTy())
    return B.CreatePtrToInt(V, I64Ty);
  assert(V->getType()->getPrimitiveSizeInBits() == 64 &&
         "64-bit register store expects a 64-bit value");
  return B.CreateBitCast(V, I64Ty);
}

// Store a 64-bit value as two i32 halves at Bank[Idx] and Bank[Idx + 1].
void storeLoHi(IRBuilder<> &B, ArrayRef<AllocaInst *> Bank, unsigned Idx,
               Value *V) {
  Type *I32Ty = B.getInt32Ty();
  V = asI64(B, V);
  B.CreateStore(B.CreateTrunc(V, I32Ty), Bank[Idx]);
  B.CreateStore(B.CreateTrunc(B.CreateLShr(V, 32), I32Ty), Bank[Idx + 1]);
}

// Combine two adjacent i32 halves at Bank[Idx] and Bank[Idx + 1] into an i64.
Value *loadLoHi(IRBuilder<> &B, ArrayRef<AllocaInst *> Bank, unsigned Idx) {
  Type *I32Ty = B.getInt32Ty();
  Type *I64Ty = B.getInt64Ty();
  Value *Lo = B.CreateZExt(B.CreateLoad(I32Ty, Bank[Idx]), I64Ty);
  Value *Hi = B.CreateZExt(B.CreateLoad(I32Ty, Bank[Idx + 1]), I64Ty);
  return B.CreateOr(Lo, B.CreateShl(Hi, 32));
}

} // namespace

void AllocaRegFile::init(IRBuilder<> &B, Type *I32Ty, Type *I1Ty,
                         const MCSubtargetInfo &STI, const MCRegisterInfo &MRI,
                         const WaveProjection &Proj) {
  Projection = &Proj;

  const unsigned NSgpr =
      MRI.getRegClass(AMDGPU::SGPR_32RegClassID).getNumRegs();
  Sgpr.assign(NSgpr, nullptr);
  for (unsigned I = 0; I < NSgpr; ++I)
    Sgpr[I] = B.CreateAlloca(I32Ty, nullptr, "Sgpr" + Twine(I));

  // VGPR storage is oversized relative to TableGen's VGPR_32 class (see
  // `KVGPRCap`). AGPR storage mirrors the VGPR size because AGPRs share
  // the same index space under MFMA encoding conventions.
  Vgpr.assign(KVGPRCap, nullptr);
  for (unsigned I = 0; I < KVGPRCap; ++I)
    Vgpr[I] = B.CreateAlloca(I32Ty, nullptr, "Vgpr" + Twine(I));

  if (STI.hasFeature(AMDGPU::FeatureMAIInsts)) {
    Agpr.assign(KVGPRCap, nullptr);
    for (unsigned I = 0; I < KVGPRCap; ++I)
      Agpr[I] = B.CreateAlloca(I32Ty, nullptr, "Agpr" + Twine(I));
  }

  // Condition-carrying scalar registers are zero-initialised so that a
  // read-before-write yields a deterministic "false / inactive" value rather
  // than poison, which would silently destroy predication.
  Vcc = B.CreateAlloca(I1Ty, nullptr, "Vcc");
  B.CreateStore(ConstantInt::getFalse(I1Ty), Vcc);
  VccHiScratch = B.CreateAlloca(I32Ty, nullptr, "VccHiScratch");
  B.CreateStore(ConstantInt::get(I32Ty, 0), VccHiScratch);
  ExecHiScratch = B.CreateAlloca(I32Ty, nullptr, "ExecHiScratch");
  B.CreateStore(ConstantInt::get(I32Ty, 0), ExecHiScratch);
  Scc = B.CreateAlloca(I1Ty, nullptr, "Scc");
  B.CreateStore(ConstantInt::getFalse(I1Ty), Scc);
  // EXEC storage width and initial value are both chosen by the projection.
  Exec = B.CreateAlloca(Proj.execStorageTy(), nullptr, "exec");
  B.CreateStore(Proj.emitInitialExec(B), Exec);
  M0 = B.CreateAlloca(I32Ty, nullptr, "M0");
  B.CreateStore(ConstantInt::get(I32Ty, 0), M0);
  FlatScr[0] = B.CreateAlloca(I32Ty, nullptr, "flat_scr_lo");
  FlatScr[1] = B.CreateAlloca(I32Ty, nullptr, "flat_scr_hi");

  const unsigned NTtmp =
      MRI.getRegClass(AMDGPU::TTMP_32RegClassID).getNumRegs();
  Ttmp.assign(NTtmp, nullptr);
  for (unsigned I = 0; I < NTtmp; ++I)
    Ttmp[I] = B.CreateAlloca(I32Ty, nullptr, "ttmp" + Twine(I));
}

void AllocaRegFile::storeSGPR32(IRBuilder<> &B, unsigned Idx, Value *V) {
  assertInBank(Sgpr, Idx);
  B.CreateStore(asI32(B, V), Sgpr[Idx]);
}

Value *AllocaRegFile::loadSGPR32(IRBuilder<> &B, unsigned Idx) {
  assertInBank(Sgpr, Idx);
  return B.CreateLoad(B.getInt32Ty(), Sgpr[Idx]);
}

void AllocaRegFile::storeSGPR64(IRBuilder<> &B, unsigned Idx, Value *V) {
  assertPairInBank(Sgpr, Idx);
  storeLoHi(B, Sgpr, Idx, V);
}

Value *AllocaRegFile::loadSGPR64(IRBuilder<> &B, unsigned Idx) {
  assertPairInBank(Sgpr, Idx);
  return loadLoHi(B, Sgpr, Idx);
}

void AllocaRegFile::storeVGPR32(IRBuilder<> &B, unsigned Idx, Value *V) {
  assertInBank(Vgpr, Idx);
  B.CreateStore(coerceToI32(B, V), Vgpr[Idx]);
}

Value *AllocaRegFile::loadVGPR32(IRBuilder<> &B, unsigned Idx) {
  assertInBank(Vgpr, Idx);
  return B.CreateLoad(B.getInt32Ty(), Vgpr[Idx]);
}

void AllocaRegFile::storeVGPR64(IRBuilder<> &B, unsigned Idx, Value *V) {
  assertPairInBank(Vgpr, Idx);
  storeLoHi(B, Vgpr, Idx, V);
}

Value *AllocaRegFile::loadVGPR64(IRBuilder<> &B, unsigned Idx) {
  assertPairInBank(Vgpr, Idx);
  return loadLoHi(B, Vgpr, Idx);
}

void AllocaRegFile::storeAGPR32(IRBuilder<> &B, unsigned Idx, Value *V) {
  assertInBank(Agpr, Idx);
  B.CreateStore(asI32(B, V), Agpr[Idx]);
}

Value *AllocaRegFile::loadAGPR32(IRBuilder<> &B, unsigned Idx) {
  assertInBank(Agpr, Idx);
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
  return B.CreateLoad(Exec->getAllocatedType(), Exec, "exec_val");
}

void AllocaRegFile::storeExec(IRBuilder<> &B, Value *V) {
  Type *ExecTy = Exec->getAllocatedType();
  if (V->getType() != ExecTy)
    V = B.CreateBitOrPointerCast(V, ExecTy);
  B.CreateStore(V, Exec);
}

Value *AllocaRegFile::readVCCAsWaveMask(IRBuilder<> &B, Type *ResultTy) {
  assert(Projection && "readVCCAsWaveMask requires a WaveProjection -- "
                       "call init() before using this reg-file");
  // The ballot only reconciles widths the source wave has lanes for. Asking
  // for more names dwords above the mask, which hold no lanes and so read as
  // zero.
  Type *MaskTy = Projection->sourceWaveMaskTy();
  if (ResultTy->getPrimitiveSizeInBits() <= MaskTy->getPrimitiveSizeInBits())
    return Projection->ballotI1ToWidth(B, loadVCC(B), ResultTy, "vcc_ballot");
  Value *Mask =
      Projection->ballotI1ToWidth(B, loadVCC(B), MaskTy, "vcc_ballot");
  return B.CreateZExt(Mask, ResultTy, "vcc_ext");
}

Value *AllocaRegFile::readReg32(IRBuilder<> &B, ParsedReg Pr) {
  if (Pr.RegKind == ParsedReg::SGPR)
    return loadSGPR32(B, requireIndex(Pr));
  if (Pr.RegKind == ParsedReg::VGPR)
    return loadVGPR32(B, requireIndex(Pr));
  if (Pr.RegKind == ParsedReg::AGPR)
    return loadAGPR32(B, requireIndex(Pr));
  if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH)
    return B.CreateLoad(B.getInt32Ty(), VccHiScratch, "vcc_hi_scratch");
  if (Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH)
    return B.CreateLoad(B.getInt32Ty(), ExecHiScratch, "exec_hi_scratch");
  // VCC read as a scalar goes through the wave-mask ballot, not a
  // sign-extension of the local i1; callers wanting a per-lane i1 call
  // `loadVCC` directly.
  if (Pr.RegKind == ParsedReg::VCC) {
    Value *V = readVCCAsWaveMask(B, Projection->sourceWaveMaskTy());
    if (Pr.WidthInDwords == 1 && Pr.BaseIdx == 1)
      V = B.CreateLShr(V, 32, "vcc_hi_shr");
    return B.CreateTruncOrBitCast(V, B.getInt32Ty(),
                                  Pr.BaseIdx == 1 ? "vcc_hi" : "vcc_lo");
  }
  if (Pr.RegKind == ParsedReg::EXEC) {
    Value *V = loadExec(B);
    Type *I32Ty = B.getInt32Ty();
    if (V->getType() == I32Ty)
      return V;
    // wave64 EXEC is i64; pick the correct half when reading a 32-bit slice.
    if (Pr.WidthInDwords >= 2)
      return B.CreateTrunc(V, I32Ty, "exec_lo");
    // A narrow read names EXEC_LO (index 0) or EXEC_HI (index 1); an absent
    // index reads the low half.
    unsigned Half = Pr.BaseIdx.value_or(0);
    if (Half == 1)
      V = B.CreateLShr(V, 32, "exec_hi_shr");
    return B.CreateTrunc(V, I32Ty, Half == 1 ? "exec_hi" : "exec_lo");
  }
  if (Pr.RegKind == ParsedReg::SCC)
    return B.CreateZExt(loadSCC(B), B.getInt32Ty());
  if (Pr.RegKind == ParsedReg::M0)
    return B.CreateLoad(B.getInt32Ty(), M0, "m0_val");
  if (Pr.RegKind == ParsedReg::FLAT_SCR) {
    unsigned Idx = requireIndex(Pr);
    assertInBank(FlatScr, Idx);
    return B.CreateLoad(B.getInt32Ty(), FlatScr[Idx], "fscr_val");
  }
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx && *Pr.BaseIdx < Ttmp.size())
    return B.CreateLoad(B.getInt32Ty(), Ttmp[*Pr.BaseIdx], "ttmp_val");
  // GFX9 src_lds_direct (encoding 254) reads one dword from LDS at the byte
  // address in M0; M0 is not auto-incremented, so the kernel manages it
  // explicitly between reads.
  if (Pr.RegKind == ParsedReg::LDS_DIRECT) {
    Type *I32Ty = B.getInt32Ty();
    Value *Addr = B.CreateLoad(I32Ty, M0, "m0_lds_off");
    PointerType *LdsPtr =
        PointerType::get(I32Ty->getContext(), AMDGPUAS::LOCAL_ADDRESS);
    Value *Ptr = B.CreateIntToPtr(Addr, LdsPtr, "lds_direct_ptr");
    return B.CreateLoad(I32Ty, Ptr, "lds_direct_val");
  }
  failUnhandledKind("readReg32", Pr);
}

Value *AllocaRegFile::readReg64(IRBuilder<> &B, ParsedReg Pr) {
  if (Pr.RegKind == ParsedReg::SGPR)
    return loadSGPR64(B, requireIndex(Pr));
  if (Pr.RegKind == ParsedReg::VGPR)
    return loadVGPR64(B, requireIndex(Pr));
  if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH)
    return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), VccHiScratch),
                        B.getInt64Ty());
  if (Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH)
    return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), ExecHiScratch),
                        B.getInt64Ty());
  // VCC read as a scalar routes through the wave-mask ballot: sign-extending
  // the local i1 would replicate this lane's VCC bit across all bits, a silent
  // lie when the consumer expects a wave-level mask (e.g. `s_and_b64 vcc, exec,
  // vcc`).
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
  if (Pr.RegKind == ParsedReg::FLAT_SCR)
    return loadLoHi(B, FlatScr, 0);
  // TTMP pair read as i64, combining the two adjacent i32 lanes.
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx &&
      *Pr.BaseIdx + 1 < Ttmp.size())
    return loadLoHi(B, Ttmp, *Pr.BaseIdx);
  failUnhandledKind("readReg64", Pr);
}

Value *AllocaRegFile::readExecWidth(IRBuilder<> &B) { return loadExec(B); }

void AllocaRegFile::writeExecWidth(IRBuilder<> &B, Value *V) {
  storeExec(B, V);
}

void AllocaRegFile::writeReg32(IRBuilder<> &B, ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::SGPR) {
    storeSGPR32(B, requireIndex(Pr), V);
    return;
  }
  if (Pr.RegKind == ParsedReg::VGPR) {
    storeVGPR32(B, requireIndex(Pr), V);
    return;
  }
  if (Pr.RegKind == ParsedReg::AGPR) {
    storeAGPR32(B, requireIndex(Pr), V);
    return;
  }
  if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH ||
      Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH) {
    // Wave32-source VCC_HI / EXEC_HI are plain 32-bit data scalars, never
    // pointers (only address/control registers carry those). Coerce any
    // non-i32 scalar width to i32.
    assert(!V->getType()->isPointerTy() &&
           "VCC_HI_SCRATCH/EXEC_HI_SCRATCH is a data scalar; pointer writes "
           "are a raiser bug");
    B.CreateStore(coerceToI32(B, V), Pr.RegKind == ParsedReg::VCC_HI_SCRATCH
                                         ? VccHiScratch
                                         : ExecHiScratch);
    return;
  }
  if (Pr.RegKind == ParsedReg::EXEC) {
    Type *I32Ty = B.getInt32Ty();
    Type *ExecTy = Exec->getAllocatedType();
    // Coerce the incoming value to i32; storeExec matches it to the EXEC
    // storage width. For wave64-native EXEC (ExecTy == i64) a 32-bit write
    // addresses a single half and is reconciled below.
    V = coerceToI32(B, V);
    if (ExecTy == I32Ty || Pr.WidthInDwords >= 2) {
      storeExec(B, V);
      return;
    }
    // ExecTy is i64 and the write is a single EXEC_LO (index 0) or EXEC_HI
    // (index 1) half. When the projection broadcasts a narrow EXEC_LO write,
    // the whole-wave mask is replicated into both halves; otherwise the other
    // half is preserved.
    unsigned Half = requireIndex(Pr);
    Value *V64 = B.CreateZExt(V, ExecTy);
    if (Projection && Projection->broadcastNarrowExecLoWrite() && Half == 0) {
      Value *Hi = B.CreateShl(V64, 32);
      Value *Merged = B.CreateOr(V64, Hi, "exec_lo_broadcast");
      storeExec(B, Merged);
      return;
    }
    Value *Cur = loadExec(B);
    Value *Merged;
    if (Half == 1) {
      Value *Mask = ConstantInt::get(ExecTy, 0xFFFFFFFFULL);
      Merged = B.CreateOr(B.CreateAnd(Cur, Mask), B.CreateShl(V64, 32),
                          "exec_hi_write");
    } else {
      Value *Mask = ConstantInt::get(ExecTy, 0xFFFFFFFF00000000ULL);
      Merged = B.CreateOr(B.CreateAnd(Cur, Mask), V64, "exec_lo_write");
    }
    storeExec(B, Merged);
    return;
  }
  if (Pr.RegKind == ParsedReg::VCC) {
    assert(Projection && "writeReg32(VCC) requires a WaveProjection");
    Value *NewBit = Projection->extractLaneBitFromWaveMask(B, V);
    if (Pr.WidthInDwords == 1 && Projection->sourceWaveSize() == 64) {
      unsigned Half = requireIndex(Pr);
      assert(Half < 2 && "VCC half index must be zero or one");
      Value *Lane = Projection->emitLaneIdx(B);
      Value *Boundary = ConstantInt::get(Lane->getType(), 32);
      Value *WritesLane = Half == 0
                              ? B.CreateICmpULT(Lane, Boundary, "vcc_write_lo")
                              : B.CreateICmpUGE(Lane, Boundary, "vcc_write_hi");
      NewBit =
          B.CreateSelect(WritesLane, NewBit, loadVCC(B), "vcc_partial_write");
    }
    storeVCC(B, NewBit);
    return;
  }
  if (Pr.RegKind == ParsedReg::M0) {
    V = asI32(B, V);
    B.CreateStore(V, M0);
    return;
  }
  if (Pr.RegKind == ParsedReg::FLAT_SCR) {
    unsigned Idx = requireIndex(Pr);
    assertInBank(FlatScr, Idx);
    B.CreateStore(asI32(B, V), FlatScr[Idx]);
    return;
  }
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx &&
      *Pr.BaseIdx < Ttmp.size()) {
    B.CreateStore(asI32(B, V), Ttmp[*Pr.BaseIdx]);
    return;
  }
  failUnhandledKind("writeReg32", Pr);
}

void AllocaRegFile::writeReg64(IRBuilder<> &B, ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::SGPR) {
    storeSGPR64(B, requireIndex(Pr), V);
    return;
  }
  if (Pr.RegKind == ParsedReg::VGPR) {
    storeVGPR64(B, requireIndex(Pr), V);
    return;
  }
  if (Pr.RegKind == ParsedReg::VCC) {
    assert(Projection && "writeReg64(VCC) requires a WaveProjection");
    storeVCC(B, Projection->extractLaneBitFromWaveMask(B, V));
    return;
  }
  if (Pr.RegKind == ParsedReg::EXEC) {
    assert(Projection && "writeReg64(EXEC) requires a WaveProjection");
    // Only the low half of a 64-bit write has lanes on a wave32 source, so it
    // lands the way a write to EXEC_LO does: narrowed to the EXEC storage
    // width the projection chose, and subject to its policy for a narrow
    // write. A wave64 source writes the mask whole.
    if (Projection->sourceWaveSize() == 32) {
      ParsedReg ExecLo = Pr;
      ExecLo.WidthInDwords = 1;
      ExecLo.BaseIdx = 0;
      writeReg32(B, ExecLo, V);
      return;
    }
    storeExec(B, V);
    return;
  }
  if (Pr.RegKind == ParsedReg::FLAT_SCR) {
    storeLoHi(B, FlatScr, 0, V);
    return;
  }
  // Trap-handler kernels materialise a 64-bit address into a TTMP pair before
  // invoking the trap; split it into two i32 stores at Idx and Idx+1.
  if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx &&
      *Pr.BaseIdx + 1 < Ttmp.size()) {
    storeLoHi(B, Ttmp, *Pr.BaseIdx, V);
    return;
  }
  failUnhandledKind("writeReg64", Pr);
}

void AllocaRegFile::writeRegExecWidth(IRBuilder<> &B, ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::SGPR) {
    unsigned Idx = requireIndex(Pr);
    // Under wave-native widening the EXEC storage is the wider target mask
    // while a source-named SGPR stays at the source width, so narrow the
    // incoming EXEC-width value to the source width before storing.
    Type *SourceWidthTy =
        (Projection && Projection->sourceWaveScopedLaneOps() &&
         Pr.WidthInDwords >= 2)
            ? B.getInt64Ty()
            : (Projection ? Projection->sourceWaveMaskTy()
                          : Exec->getAllocatedType());
    if (V->getType() != SourceWidthTy) {
      unsigned Have = V->getType()->getPrimitiveSizeInBits();
      unsigned Want = SourceWidthTy->getPrimitiveSizeInBits();
      if (Have > Want)
        V = B.CreateTrunc(V, SourceWidthTy, "wn_exec_to_src_mask");
      else if (Have < Want)
        V = B.CreateZExt(V, SourceWidthTy, "wn_exec_to_src_mask");
    }
    if (SourceWidthTy == B.getInt32Ty())
      storeSGPR32(B, Idx, V);
    else
      storeSGPR64(B, Idx, V);
    return;
  }
  if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH ||
      Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH) {
    // Wave32 vcc_hi / exec_hi are scratch scalars, not the wave mask.
    writeReg32(B, Pr, V);
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
  failUnhandledKind("writeRegExecWidth", Pr);
}

Value *AllocaRegFile::readRegVec(IRBuilder<> &B, ParsedReg Pr, Type *VecTy) {
  FixedVectorType *VecT = dyn_cast<FixedVectorType>(VecTy);
  unsigned N = VecT ? VecT->getNumElements() : 1;
  Type *ElemTy = VecT ? VecT->getElementType() : VecTy;

  if (N == 1 && !VecT && VecTy->getPrimitiveSizeInBits() <= 32) {
    Value *V = readReg32(B, Pr);
    if (V->getType() != VecTy)
      V = B.CreateBitCast(V, VecTy);
    return V;
  }

  unsigned TotalDwords = 0;
  if (ElemTy->isFloatTy())
    TotalDwords = N;
  else if (ElemTy->isIntegerTy(32))
    TotalDwords = N;
  else if (ElemTy->isHalfTy())
    TotalDwords = (N + 1) / 2;
  else
    TotalDwords = (N * ElemTy->getPrimitiveSizeInBits() + 31) / 32;

  const unsigned Base = requireIndex(Pr);
  SmallVector<Value *> Dwords;
  for (unsigned I = 0; I < TotalDwords; ++I) {
    ParsedReg Sub = Pr;
    Sub.BaseIdx = Base + I;
    Sub.WidthInDwords = 1;
    Dwords.push_back(readReg32(B, Sub));
  }

  unsigned TotalBits = TotalDwords * 32;
  Type *IntTy = Type::getIntNTy(B.getContext(), TotalBits);

  Value *Packed = ConstantInt::get(IntTy, 0);
  for (unsigned I = 0; I < TotalDwords; ++I) {
    Value *Ext = B.CreateZExt(Dwords[I], IntTy);
    if (I > 0)
      Ext = B.CreateShl(Ext, I * 32);
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

  const unsigned Base = requireIndex(Pr);
  for (unsigned I = 0; I < TotalDwords; ++I) {
    Value *Dw;
    if (I == 0)
      Dw = B.CreateTrunc(Packed, I32Ty);
    else
      Dw = B.CreateTrunc(B.CreateLShr(Packed, I * 32), I32Ty);
    ParsedReg Sub = Pr;
    Sub.BaseIdx = Base + I;
    Sub.WidthInDwords = 1;
    writeReg32(B, Sub, Dw);
  }
}

void AllocaRegFile::collectAllocas(SmallVectorImpl<AllocaInst *> &Out) const {
  for (AllocaInst *A : Sgpr)
    Out.push_back(A);
  for (AllocaInst *A : Vgpr)
    Out.push_back(A);
  for (AllocaInst *A : Agpr)
    Out.push_back(A);
  if (Vcc)
    Out.push_back(Vcc);
  // VccHiScratch must be promoted too: a surviving private alloca would be
  // moved to LDS by the AMDGPU backend, where it could alias the kernel's
  // own LDS and be clobbered.
  if (VccHiScratch)
    Out.push_back(VccHiScratch);
  // ExecHiScratch is promoted for the same reason as VccHiScratch.
  if (ExecHiScratch)
    Out.push_back(ExecHiScratch);
  if (Scc)
    Out.push_back(Scc);
  if (Exec)
    Out.push_back(Exec);
  if (M0)
    Out.push_back(M0);
  for (AllocaInst *A : FlatScr)
    if (A)
      Out.push_back(A);
  // ttmps must be promoted too: a surviving private alloca is moved to
  // LDS by the AMDGPU backend, which re-enables dispatch_ptr and lets the
  // register allocator reuse s[0:1], corrupting preloaded pointers.
  for (AllocaInst *A : Ttmp)
    Out.push_back(A);
}

} // namespace COMGR::hotswap
