//===- handle-mubuf.cpp - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"
#include "mubuf-addr.h"

#include "canonical-op.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace COMGR::hotswap {
HandlerResult handleMUBUF(RaiseContext &Ctx, const DecodedInst &Di,
                        OpResolver &Op) {
  HandlerResult Hr;
  StringRef Mn(Di.Mnemonic);
  CanonicalOp Sop = Di.CanonOp;

  // d16Half encodes D16 partial-write target: 0 = full-write (regular
  // load), 1 = low half (`_D16`), 2 = high half (`_D16_HI`). The
  // partial-write loads zero/sign-extend the loaded sub-dword to i16
  // and merge into the named half of the destination VGPR, preserving
  // the other half. See BUFInstructions.td:1155-1177 (predicate
  // `D16PreservesUnusedBits`).
  auto MubufClassify = [](CanonicalOp S)
      -> std::tuple<bool, bool, int, int, bool, bool, int> {
    // returns {isLoad, isStore, dwords, loadBits, isSubDword, isSigned, d16Half}
    switch (S) {
    case CanonicalOp::BUFFER_LOAD_DWORD:    return {true, false, 1, 32, false, false, 0};
    case CanonicalOp::BUFFER_LOAD_DWORDX2:  return {true, false, 2, 64, false, false, 0};
    case CanonicalOp::BUFFER_LOAD_DWORDX3:  return {true, false, 3, 96, false, false, 0};
    case CanonicalOp::BUFFER_LOAD_DWORDX4:  return {true, false, 4, 128, false, false, 0};
    case CanonicalOp::BUFFER_LOAD_UBYTE:    return {true, false, 1, 8, true, false, 0};
    case CanonicalOp::BUFFER_LOAD_SBYTE:    return {true, false, 1, 8, true, true, 0};
    case CanonicalOp::BUFFER_LOAD_USHORT:   return {true, false, 1, 16, true, false, 0};
    case CanonicalOp::BUFFER_LOAD_SSHORT:   return {true, false, 1, 16, true, true, 0};
    case CanonicalOp::BUFFER_LOAD_SHORT_D16:     return {true, false, 1, 16, true, false, 1};
    case CanonicalOp::BUFFER_LOAD_SHORT_D16_HI:  return {true, false, 1, 16, true, false, 2};
    case CanonicalOp::BUFFER_LOAD_UBYTE_D16:     return {true, false, 1, 8,  true, false, 1};
    case CanonicalOp::BUFFER_LOAD_UBYTE_D16_HI:  return {true, false, 1, 8,  true, false, 2};
    case CanonicalOp::BUFFER_LOAD_SBYTE_D16:     return {true, false, 1, 8,  true, true,  1};
    case CanonicalOp::BUFFER_LOAD_SBYTE_D16_HI:  return {true, false, 1, 8,  true, true,  2};
    case CanonicalOp::BUFFER_STORE_DWORD:   return {false, true, 1, 32, false, false, 0};
    case CanonicalOp::BUFFER_STORE_DWORDX2: return {false, true, 2, 64, false, false, 0};
    case CanonicalOp::BUFFER_STORE_DWORDX3: return {false, true, 3, 96, false, false, 0};
    case CanonicalOp::BUFFER_STORE_DWORDX4: return {false, true, 4, 128, false, false, 0};
    case CanonicalOp::BUFFER_STORE_BYTE:    return {false, true, 1, 8, true, false, 0};
    case CanonicalOp::BUFFER_STORE_SHORT:   return {false, true, 1, 16, true, false, 0};
    default: return {false, false, 0, 0, false, false, 0};
    }
  };
  auto [isLoad, isStore, dwords, loadBits, isSubDword, isBufSigned, d16Half] =
      MubufClassify(Sop);
  if (isLoad || isStore) {
    // Use gfx942 buffer intrinsics directly. The hardware handles OOB:
    // loads return 0, stores are silently dropped. This avoids the
    // flat-memory lowering that requires conditional branches (which
    // break under LLVM -O1+ SIMT optimizations).
    MubufAddr Mbuf = decodeMubufAddr(Ctx, Di, Op, isStore, "MUBUF");
    // For loads, vdata is the dst; for stores it's the first VGPR src
    // (captured into mbuf.stData by the decoder).
    ParsedReg Vdata = isStore ? Mbuf.StData : Op.dst(0);
    Value *Srd = Mbuf.Srd;
    Value *Voffset = Mbuf.Voffset;
    Value *Soffset = Mbuf.Soffset;
    Value *AuxFlags = Mbuf.AuxFlags;
    auto RawPtrBufferLoad = [&](Type *LoadTy) -> Value * {
      Function *BufLd = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_raw_ptr_buffer_load, {LoadTy});
      return Ctx.B.CreateCall(
          BufLd, {Mbuf.RawPtrRsrc, Voffset, Soffset, AuxFlags},
          "buf_ld_rawptr");
    };

    if (isLoad) {
      if (isSubDword) {
        // Load the sub-dword datum and zero/sign-extend to i32. For
        // plain ushort/sbyte/etc. (`d16Half == 0`) we then write the
        // whole VGPR; for D16 partial-write loads we merge with the
        // prior dst (see comment block above mubufClassify).
        Type *MemTy = (loadBits == 8) ? Type::getInt8Ty(Ctx.C)
                                      : Type::getInt16Ty(Ctx.C);
        Value *Loaded = nullptr;
        if (Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
          Loaded = RawPtrBufferLoad(MemTy);
        } else {
          Function *BufLd = Intrinsic::getOrInsertDeclaration(
              &Ctx.M, Intrinsic::amdgcn_raw_buffer_load, {MemTy});
          Loaded = Ctx.B.CreateCall(BufLd,
              {Srd, Voffset, Soffset, AuxFlags}, "buf_ld");
        }
        if (d16Half == 0) {
          Value *Ext = isBufSigned ? Ctx.B.CreateSExt(Loaded, Ctx.I32Ty)
                                   : Ctx.B.CreateZExt(Loaded, Ctx.I32Ty);
          Ctx.writeReg32(Vdata, Ext);
        } else {
          // Partial-write: extend to i16 (sign for `_SBYTE_D16*`,
          // zero for `_UBYTE_D16*` / `_SHORT_D16*`), zext to i32 so
          // the high half of the i32 is exactly zero before merging.
          Value *Ext16 = Loaded;
          if (loadBits == 8) {
            Ext16 = isBufSigned
                        ? Ctx.B.CreateSExt(Loaded, Type::getInt16Ty(Ctx.C))
                        : Ctx.B.CreateZExt(Loaded, Type::getInt16Ty(Ctx.C));
          }
          Value *Ext32 = Ctx.B.CreateZExt(Ext16, Ctx.I32Ty);
          Value *Prior = Ctx.Regs.readReg32(Ctx.B, Vdata);
          Value *Merged;
          if (d16Half == 1) {
            // _D16: place datum in lo 16, preserve hi 16 of prior.
            Value *PriorHi =
                Ctx.B.CreateAnd(Prior, ConstantInt::get(Ctx.I32Ty, 0xFFFF0000));
            Merged = Ctx.B.CreateOr(PriorHi, Ext32, "d16_lo_merge");
          } else {
            // _D16_HI: place datum in hi 16, preserve lo 16 of prior.
            Value *PriorLo =
                Ctx.B.CreateAnd(Prior, ConstantInt::get(Ctx.I32Ty, 0x0000FFFF));
            Value *Shifted =
                Ctx.B.CreateShl(Ext32, ConstantInt::get(Ctx.I32Ty, 16));
            Merged = Ctx.B.CreateOr(PriorLo, Shifted, "d16_hi_merge");
          }
          Ctx.writeReg32(Vdata, Merged);
        }
      } else if (dwords == 1) {
        Value *Loaded = nullptr;
        if (Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
          Loaded = RawPtrBufferLoad(Ctx.I32Ty);
        } else {
          Function *BufLd = Intrinsic::getOrInsertDeclaration(
              &Ctx.M,
              Intrinsic::amdgcn_raw_buffer_load,
              {Ctx.I32Ty});
          Loaded = Ctx.B.CreateCall(BufLd,
              {Srd, Voffset, Soffset, AuxFlags}, "buf_ld");
        }
        Ctx.writeReg32(Vdata, Loaded);
      } else {
        auto *VecTy = FixedVectorType::get(Ctx.I32Ty, dwords);
        Value *Loaded = nullptr;
        if (Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
          Loaded = RawPtrBufferLoad(VecTy);
        } else {
          Function *BufLd = Intrinsic::getOrInsertDeclaration(
              &Ctx.M,
              Intrinsic::amdgcn_raw_buffer_load,
              {VecTy});
          Loaded = Ctx.B.CreateCall(BufLd,
              {Srd, Voffset, Soffset, AuxFlags}, "buf_ld");
        }
        Ctx.writeRegVec(Vdata, Loaded);
      }
      Hr.Handled = true;
    return Hr;
    }
    if (isStore) {
      // Use the gfx942 buffer-store intrinsic directly, exactly
      // mirroring the load path above. The hardware's BUFFER unit
      // handles OOB silently (the write is dropped when the per-lane
      // offset is >= num_records), so no software OOB sink is needed.
      //
      // The earlier implementation lowered every BUFFER_STORE_* to a
      // generic `store` against an `addrspacecast(alloca i32, addrspace(5))`
      // OOB sink (selected via `select i1 oob, sink, real`). That was
      // wrong on three independent axes:
      //
      //   1. Size mismatch. The sink alloca was always `i32` (4 B), but
      //      `BUFFER_STORE_DWORDX4` writes 16 B. For OOB lanes the
      //      flat_store_dwordx4 walked 12 B past the sink and into
      //      either the next per-thread scratch slot or unmapped
      //      scratch -- root cause of the gfx1250 Triton vector-add
      //      SIGSEGV (R1).
      //
      //   2. Forced scratch enablement. Adding any `addrspace(5)`
      //      alloca that survives PromoteMemToReg makes the AMDGPU
      //      backend emit `.amdhsa_enable_private_segment 1` plus
      //      `.amdhsa_private_segment_fixed_size > 0`. Hotswap's KD
      //      doesn't request `flat_scratch_init` (we model only the
      //      source ABI's user-SGPR set), so on entry FLAT_SCRATCH is
      //      undefined; any flat instruction touching the scratch
      //      aperture (including the OOB sink path) is a fault waiting
      //      to happen. Native gfx942 Triton emits `buffer_store_*`
      //      directly and reports `private_segment_fixed_size 0` /
      //      `enable_private_segment 0` -- confirming hardware OOB
      //      handling is the right primitive.
      //
      //   3. Asymmetric with the load path. Loads already go through
      //      `amdgcn.raw.buffer.load` and rely on hardware OOB clamp.
      //      Routing stores through a software select+sink was an
      //      avoidable divergence whose only justification ("avoids
      //      flat-memory lowering with conditional branches breaking
      //      under -O1+ SIMT optimisations" -- comment block above)
      //      doesn't apply when we use the buffer intrinsic itself.
      //
      // EXEC gating:
      //
      // In the ordinary projection path we wrap the store in `emitUnderExec`,
      // matching the rest of the side-effecting handlers: source-inactive
      // lanes skip the intrinsic in IR.
      //
      // WaveNative cross-widening is different.  That projection deliberately
      // holds hardware EXEC at the full target wave for the kernel body
      // (`providesFullWaveExecInvariant`).  Triton masked MUBUF stores in this
      // class already encode the per-lane predicate in the vector offset:
      // inactive source lanes receive an OOB offset and the BUFFER unit drops
      // those writes.  Adding a second IR-level `emitUnderExec` diamond can
      // make the translated code stricter than the source packet and drop the
      // valid lane as well; `get_num_kv_splits_triton` is the observed case.
      // Emitting the raw buffer store directly preserves the source packet's
      // hardware contract: full-wave issue plus per-lane OOB suppression.
      Type *StoreTy;
      Value *Val;
      if (isSubDword) {
        StoreTy = Type::getIntNTy(Ctx.C, loadBits);
        Val = Ctx.B.CreateTrunc(Ctx.Regs.readReg32(Ctx.B, Vdata), StoreTy);
      } else if (dwords == 1) {
        StoreTy = Ctx.I32Ty;
        Val = Ctx.Regs.readReg32(Ctx.B, Vdata);
      } else {
        auto *VecTy = FixedVectorType::get(Ctx.I32Ty, dwords);
        StoreTy = VecTy;
        Val = Ctx.Regs.readRegVec(Ctx.B, Vdata, VecTy);
      }
      Function *BufSt = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_raw_buffer_store, {StoreTy});
      auto EmitStore = [&] {
        Ctx.B.CreateCall(BufSt, {Val, Srd, Voffset, Soffset, AuxFlags});
      };
      if (Ctx.Projection.providesFullWaveExecInvariant())
        EmitStore();
      else
        Ctx.emitUnderExec(EmitStore);
      Hr.Handled = true;
    return Hr;
    }
  }

  // ---- Buffer load to LDS (buffer_load_dword lds, ...) ----
  // Data goes directly to LDS at M0 + vaddr, not to a VGPR.
  // Model as: tmp = raw_buffer_load; ds_write(LDS[M0], tmp)
  if (Sop == CanonicalOp::BUFFER_LOAD_DWORD_LDS ||
      Sop == CanonicalOp::BUFFER_LOAD_DWORDX2_LDS ||
      Sop == CanonicalOp::BUFFER_LOAD_DWORDX4_LDS) {
    int Dwords = (Sop == CanonicalOp::BUFFER_LOAD_DWORDX4_LDS) ? 4
               : (Sop == CanonicalOp::BUFFER_LOAD_DWORDX2_LDS) ? 2 : 1;

    MubufAddr Mbuf = decodeMubufAddr(Ctx, Di, Op, /*isStore=*/false,
                                      "MUBUF_LDS");

    // Load from buffer into a temp value.
    Type *LdTy = (Dwords == 1)
                     ? Ctx.I32Ty
                     : FixedVectorType::get(Ctx.I32Ty, Dwords);
    Function *BufLd = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_raw_buffer_load, {LdTy});
    Value *Loaded = Ctx.B.CreateCall(
        BufLd, {Mbuf.Srd, Mbuf.Voffset, Mbuf.Soffset, Mbuf.AuxFlags},
        "lds_buf_ld");

    // Store to LDS at address from M0.
    ParsedReg M0Reg; M0Reg.RegKind = ParsedReg::M0; M0Reg.BaseIdx = 0;
    Value *LdsAddr = Ctx.Regs.readReg32(Ctx.B, M0Reg);
    auto *LdsPtrTy = PointerType::get(Ctx.C, 3);
    Value *LdsPtr = Ctx.B.CreateIntToPtr(LdsAddr, LdsPtrTy);
    Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Loaded, LdsPtr); });

    Hr.Handled = true;
    return Hr;
  }

  // ---- Buffer atomics ----
  //
  // RTN / non-RTN operand shape note.  MUBUF buffer atomics put the
  // vdata register at operand 0 in BOTH the RTN (glc=1 / tied-def)
  // and non-RTN (glc=0 / pure-source) forms.  The difference is
  // whether operand 0 is also tied to the destination (RTN) or
  // only a source (non-RTN).  `decodeMubufAddr(..., isStore=true)`
  // treats the first VGPR source as vdata for both forms, and the
  // RTN-only write-back below is gated by `di.numDefs > 0`
  // (consistent with the assert just below this comment), which
  // correctly SKIPS for non-RTN.  The
  // two per-form invariants are pinned by
  // `lit_tests/buffer_atomic_swap_b32/` (RTN) +
  // `lit_tests/buffer_atomic_swap_b32_nortn/` (non-RTN) and the
  // cmpswap twins.
  if (Sop >= CanonicalOp::BUFFER_ATOMIC_ADD && Sop <= CanonicalOp::BUFFER_ATOMIC_MAX_NUM_F64) {
    assert(((Di.TsFlags & SIInstrFlags::IsAtomicRet) != 0) == (Di.NumDefs > 0) &&
           "buffer atomic: IsAtomicRet disagrees with numDefs");
    MubufAddr Mbuf = decodeMubufAddr(Ctx, Di, Op, /*isStore=*/true,
                                     "buffer_atomic");

    // `BUFFER_ATOMIC_CMPSWAP` is the one buffer atomic whose vdata is
    // a register PAIR carrying `{cmp, new}` rather than a single data
    // word.  Split it out before the single-word raw-buffer atomic
    // dispatch below. The MUBUF data register is the first VGPR source,
    // and the second word is read with a synthetic `baseIdx + 1`.
    if (Sop == CanonicalOp::BUFFER_ATOMIC_CMPSWAP) {
      ParsedReg DataPair = Mbuf.StData;
      Value *CmpVal = Ctx.Regs.readReg32(Ctx.B, DataPair);
      ParsedReg NewReg = DataPair;
      NewReg.BaseIdx += 1;
      NewReg.Width = 1;
      Value *NewVal = Ctx.Regs.readReg32(Ctx.B, NewReg);
      Function *CasFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_raw_buffer_atomic_cmpswap, {Ctx.I32Ty});
      Ctx.emitUnderExec([&] {
        // Raw-buffer atomics preserve descriptor-relative addressing and
        // hardware OOB behavior. The intrinsic takes {new, cmp}, matching
        // LLVM's AMDGPU intrinsic contract for buffer cmpswap.
        Value *OldVal = Ctx.B.CreateCall(
            CasFn, {NewVal, CmpVal, Mbuf.Srd, Mbuf.Voffset, Mbuf.Soffset,
                    Mbuf.AuxFlags},
            "buf_atomic_cmpswap");
        if (Di.NumDefs > 0)
          Ctx.Regs.writeReg32(Ctx.B, Op.dst(), OldVal);
      });
      Hr.Handled = true;
      return Hr;
    }

    const bool IsF64 = Sop == CanonicalOp::BUFFER_ATOMIC_ADD_F64 ||
                       Sop == CanonicalOp::BUFFER_ATOMIC_MIN_F64 ||
                       Sop == CanonicalOp::BUFFER_ATOMIC_MAX_F64 ||
                       Sop == CanonicalOp::BUFFER_ATOMIC_MIN_NUM_F64 ||
                       Sop == CanonicalOp::BUFFER_ATOMIC_MAX_NUM_F64;
    Value *Data = IsF64 ? Ctx.Regs.readReg64(Ctx.B, Mbuf.StData)
                        : Ctx.Regs.readReg32(Ctx.B, Mbuf.StData);

    // BUFFER_ATOMIC_{MIN,MAX}{,_NUM}_F64: CAS loop. The canonical op
    // distinguishes gfx942 raw `<`/`>` from gfx12 IEEE 754-2019
    // minimumNumber/maximumNumber; pick the comparator accordingly.
    if (Sop == CanonicalOp::BUFFER_ATOMIC_MIN_F64 ||
        Sop == CanonicalOp::BUFFER_ATOMIC_MAX_F64 ||
        Sop == CanonicalOp::BUFFER_ATOMIC_MIN_NUM_F64 ||
        Sop == CanonicalOp::BUFFER_ATOMIC_MAX_NUM_F64) {
      const bool IsMax = Sop == CanonicalOp::BUFFER_ATOMIC_MAX_F64 ||
                         Sop == CanonicalOp::BUFFER_ATOMIC_MAX_NUM_F64;
      const bool IsIeeeNum =
          Sop == CanonicalOp::BUFFER_ATOMIC_MIN_NUM_F64 ||
          Sop == CanonicalOp::BUFFER_ATOMIC_MAX_NUM_F64;
      Value *SrcF64 = Ctx.B.CreateBitCast(Data, Ctx.F64Ty,
                                          "fp64_minmax_src");
      Function *BufLd = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_raw_buffer_load, {Ctx.I64Ty});
      Function *CasFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_raw_buffer_atomic_cmpswap,
          {Ctx.I64Ty});
      Ctx.emitUnderExec([&] {
        Value *InitI64 = Ctx.B.CreateCall(
            BufLd,
            {Mbuf.Srd, Mbuf.Voffset, Mbuf.Soffset, Mbuf.AuxFlags},
            "fp64_minmax_init");
        Function *F = Ctx.B.GetInsertBlock()->getParent();
        BasicBlock *PreBb = Ctx.B.GetInsertBlock();
        BasicBlock *LoopBb =
            BasicBlock::Create(Ctx.C, "fp64_minmax_loop", F);
        BasicBlock *ExitBb =
            BasicBlock::Create(Ctx.C, "fp64_minmax_exit", F);
        Ctx.B.CreateBr(LoopBb);
        Ctx.B.SetInsertPoint(LoopBb);
        PHINode *Expected =
            Ctx.B.CreatePHI(Ctx.I64Ty, 2, "fp64_minmax_expected");
        Expected->addIncoming(InitI64, PreBb);
        Value *OldF64 = Ctx.B.CreateBitCast(Expected, Ctx.F64Ty,
                                            "fp64_minmax_old");
        Value *NewF64;
        if (IsIeeeNum) {
          Intrinsic::ID NumIntr =
              IsMax ? Intrinsic::maximumnum : Intrinsic::minimumnum;
          Function *NumFn = Intrinsic::getOrInsertDeclaration(
              &Ctx.M, NumIntr, {Ctx.F64Ty});
          NewF64 = Ctx.B.CreateCall(NumFn, {OldF64, SrcF64},
                                    "fp64_minmax_new");
        } else {
          Value *Cmp =
              IsMax ? Ctx.B.CreateFCmpOGT(SrcF64, OldF64,
                                          "fp64_minmax_cmp")
                    : Ctx.B.CreateFCmpOLT(SrcF64, OldF64,
                                          "fp64_minmax_cmp");
          NewF64 = Ctx.B.CreateSelect(Cmp, SrcF64, OldF64,
                                      "fp64_minmax_new");
        }
        Value *NewI64 = Ctx.B.CreateBitCast(NewF64, Ctx.I64Ty,
                                            "fp64_minmax_new_bits");
        Value *Returned = Ctx.B.CreateCall(
            CasFn,
            {NewI64, Expected, Mbuf.Srd, Mbuf.Voffset, Mbuf.Soffset,
             Mbuf.AuxFlags},
            "fp64_minmax_cas");
        Value *Ok = Ctx.B.CreateICmpEQ(Returned, Expected,
                                       "fp64_minmax_ok");
        Expected->addIncoming(Returned, Ctx.B.GetInsertBlock());
        Ctx.B.CreateCondBr(Ok, ExitBb, LoopBb);
        Ctx.B.SetInsertPoint(ExitBb);
        if (Di.NumDefs > 0)
          Ctx.Regs.writeReg64(Ctx.B, Op.dst(), Returned);
      });
      Hr.Handled = true;
      return Hr;
    }

    Intrinsic::ID AtomicIntrinsic = Intrinsic::not_intrinsic;
    Type *AtomicTy = Ctx.I32Ty;
    bool IsFp = false;
    switch (Sop) {
    case CanonicalOp::BUFFER_ATOMIC_ADD:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_add;
      break;
    case CanonicalOp::BUFFER_ATOMIC_SUB:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_sub;
      break;
    case CanonicalOp::BUFFER_ATOMIC_AND:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_and;
      break;
    case CanonicalOp::BUFFER_ATOMIC_OR:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_or;
      break;
    case CanonicalOp::BUFFER_ATOMIC_XOR:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_xor;
      break;
    // `buffer_atomic_swap` -- pure exchange.  Added in the same
    // commit as the CMPSWAP branch above to close the handler gap
    // that used to refuse both atomics at the `default:` arm.  The
    // RTN-form write-back (see the shared emit below) is what makes
    // SWAP semantically meaningful; a dropped result would reduce
    // `buffer_atomic_swap` to a plain store and lose the caller's
    // "old value" read, quietly miscompiling any CAS-loop or
    // lock-free shape that relies on it.
    case CanonicalOp::BUFFER_ATOMIC_SWAP:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_swap;
      break;
    case CanonicalOp::BUFFER_ATOMIC_ADD_F32:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_fadd;
      AtomicTy = Ctx.F32Ty;
      IsFp = true;
      break;
    case CanonicalOp::BUFFER_ATOMIC_PK_ADD_BF16:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_fadd;
      AtomicTy = FixedVectorType::get(Type::getBFloatTy(Ctx.C), 2);
      IsFp = true;
      break;
    case CanonicalOp::BUFFER_ATOMIC_PK_ADD_F16:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_fadd;
      AtomicTy = FixedVectorType::get(Type::getHalfTy(Ctx.C), 2);
      IsFp = true;
      break;
    case CanonicalOp::BUFFER_ATOMIC_ADD_F64:
      AtomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_fadd;
      AtomicTy = Ctx.F64Ty;
      IsFp = true;
      break;
    // BUFFER_ATOMIC_{MIN,MAX}{,_NUM}_F64 handled by the CAS-loop block
    // above the switch; they never reach this default.
    default:
      llvm::errs() << "transpiler: Unsupported buffer atomic: " << Mn << "\n";
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(Di, "MUBUF",
                                                   "unsupported buffer atomic");
      return Hr;
    }
    if (IsFp) Data = Ctx.B.CreateBitCast(Data, AtomicTy);
    Function *AtomicFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, AtomicIntrinsic, {AtomicTy});
    Ctx.emitUnderExec([&] {
      Value *OldVal = Ctx.B.CreateCall(
          AtomicFn,
          {Data, Mbuf.Srd, Mbuf.Voffset, Mbuf.Soffset, Mbuf.AuxFlags},
          "buf_atomic");
      // RTN-form write-back. The raw-buffer intrinsic returns the old
      // memory value just like the target ISA RTN form; when the source
      // is non-RTN, leaving the result unused lets the backend select
      // the no-return encoding.
      if (Di.NumDefs > 0) {
        Value *RetVal = OldVal;
        if (IsF64) {
          if (IsFp) RetVal = Ctx.B.CreateBitCast(RetVal, Ctx.I64Ty);
          Ctx.Regs.writeReg64(Ctx.B, Op.dst(), RetVal);
        } else {
          if (IsFp) RetVal = Ctx.B.CreateBitCast(RetVal, Ctx.I32Ty);
          Ctx.Regs.writeReg32(Ctx.B, Op.dst(), RetVal);
        }
      }
    });
    Hr.Handled = true;
    return Hr;
  }
  return Hr;
}

} // namespace COMGR::hotswap
