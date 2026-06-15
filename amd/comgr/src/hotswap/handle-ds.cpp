//===- handle-ds.cpp - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handlers.h"

#include "canonical-op.h"
#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace COMGR::hotswap {
HandlerResult handleDS(RaiseContext &Ctx, const DecodedInst &Di,
                        OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  // Map CanonicalOp to {dwords, loadBits, isSigned} for SINGLE-OFFSET DS
  // read/write. The two-offset DS_READ2/WRITE2 family (including the
  // gfx11+ *ST64 variants) has fundamentally different memory
  // semantics -- two independent accesses at raw-field-scaled byte
  // offsets -- and is intercepted by the dedicated block below BEFORE
  // this classifier runs. Keeping the two-offset SemOps out of this
  // table is how we prevent the former silent-miscompile shape
  // (single contiguous `<N x i32>` load at raw offset0) from ever
  // being reachable again; if a handler ever falls through to here
  // with a DS_READ2* CanonicalOp the dsClassify default returns {-1, 0, …}
  // and the generic block below surfaces it as `unsupportedInstructionForm`
  // rather than silently emitting wrong IR.
  auto DsClassify = [](CanonicalOp S) -> std::tuple<int, int, bool> {
    switch (S) {
    case CanonicalOp::DS_READ_B128:  case CanonicalOp::DS_WRITE_B128:
      return {4, 128, false};
    // 96-bit (3 x i32) LDS load/store. gfx11+ asm spellings are
    // `ds_load_b96` / `ds_store_b96`; LLVM MC keeps the legacy
    // `DS_READ_B96` / `DS_WRITE_B96` pseudo names. The generic
    // `vecTy = <3 x i32>` path below handles the lift; the gfx942
    // backend lowers the resulting `load <3 x i32>` /
    // `store <3 x i32>` to either a native ds_read_b96/ds_write_b96
    // (gfx9 inherits the `_vi` Real form from DSInstructions.td) or
    // splits into 3x ds_read_b32/ds_write_b32 -- both correct.
    case CanonicalOp::DS_READ_B96:   case CanonicalOp::DS_WRITE_B96:
      return {3, 96, false};
    case CanonicalOp::DS_READ_B64:   case CanonicalOp::DS_WRITE_B64:
      return {2, 64, false};
    case CanonicalOp::DS_READ_B32:   case CanonicalOp::DS_WRITE_B32:
      return {1, 32, false};
    case CanonicalOp::DS_READ_U16:   case CanonicalOp::DS_WRITE_B16:
      return {0, 16, false};
    case CanonicalOp::DS_READ_I16:
      return {0, 16, true};
    case CanonicalOp::DS_READ_U8:    case CanonicalOp::DS_WRITE_B8:
      return {0, 8, false};
    case CanonicalOp::DS_READ_I8:
      return {0, 8, true};
    default: return {-1, 0, false};
    }
  };
  // ds_load_tr16_b128: LDS transpose load for Wave32 with 16-bit elements.
  //
  // Hardware behaviour (gfx1250, Wave32):
  //   Each lane provides a base address via VGPR + immediate offset.  The
  //   hardware reads 128 bits (8 × i16) from each lane's LDS address, then
  //   transposes the data across lanes within groups of 8.
  //
  //   Per the CDNA4 ISA doc §11.4 (gfx950 DS_READ_B64_TR_B16):
  //   "Read N bits of data per lane from data share. Interpret the data as
  //    a matrix with 16 bit elements and transpose the matrix."
  //   "Each lane (one VGPR) holds 4 consecutive M or N values."
  //
  //   The transpose converts M-contiguous LDS data into the WMMA/MFMA
  //   register layout where each thread holds K-contiguous elements.
  //
  // Software emulation (for gfx942 which lacks transpose loads):
  //   1. Contiguous 128-bit load (4 × i32) from each lane's address.
  //   2. 8×8 cross-lane transpose via ds_bpermute within groups of 8 lanes:
  //        result[lane][elem] = raw[group_base + elem][lane_in_group]
  //      This exchanges rows and columns so that each lane, which started
  //      with 8 values from consecutive M positions for one K column, now
  //      holds 8 values from different K columns for its M position.
  // gfx950 ds_read_b64_tr_b16: LDS transpose read, returns 64 bits as
  // v4i16 (4 × i16). Emit the LLVM intrinsic so the backend can lower it
  // to the correct instruction for the target ISA (native on gfx950,
  // software-emulated on targets that lack it).
  if (Sop == CanonicalOp::DS_READ_B64_TR_B16) {
    Value *Addr = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty, "ds_addr");
    for (unsigned K = 1; K < Op.nSrcs(); K++) {
      if (Di.isImm(Op.srcIdx(K))) {
        int64_t Imm = Di.getImm(Op.srcIdx(K));
        if (Imm != 0)
          Addr = Ctx.B.CreateAdd(Addr, ConstantInt::get(Ctx.I64Ty, Imm), "ds_off");
        break;
      }
    }
    auto *LdsPtrTy = PointerType::get(Ctx.C, 3);
    Value *Ptr = Ctx.B.CreateIntToPtr(Addr, LdsPtrTy, "tr64_ptr");
    auto *V4i16Ty = FixedVectorType::get(Type::getInt16Ty(Ctx.C), 4);
    Function *TrFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_ds_read_tr16_b64, {V4i16Ty});
    Value *TrResult = Ctx.B.CreateCall(TrFn, {Ptr}, "tr64_ld");
    Value *AsDwords = Ctx.B.CreateBitCast(
        TrResult, FixedVectorType::get(Ctx.I32Ty, 2), "tr64_dw");
    ParsedReg Dest = Op.dst();
    Ctx.writeRegVec(Dest, AsDwords);
    Hr.Handled = true;
    return Hr;
  }
  // 64-bit transposed LDS load with 8-bit elements (gfx950
  // `ds_read_b64_tr_b8` and its gfx1250 spelling `ds_load_tr8_b64`).
  //
  // Hardware behaviour (CDNA4 ISA §11.4 / RDNA4 gfx1250 spec):
  //   Each lane provides a base address via VGPR + immediate offset.
  //   The hardware reads 64 bits (8 x i8) from each lane's LDS
  //   address, then transposes across 8-lane groups so each lane
  //   post-transpose holds 8 i8 values that originally lived at the
  //   same intra-group element offset across 8 different source
  //   lanes:
  //     result[lane][k] = raw[group_base + k][l_in_group]   k = 0..7
  //   where group_base = lane & ~7 and l_in_group = lane & 7.
  //   The transpose converts M-contiguous LDS data into the
  //   WMMA/MFMA register layout where each thread holds K-contiguous
  //   elements.
  //
  // Software emulation (target gfx942 has no isel pattern for either
  // intrinsic -- neither HasGFX950Insts nor isGFX1250Plus):
  //   1. Compute lane_id, group_base, l_in_group via
  //      mbcnt_lo/mbcnt_hi.
  //   2. For each output dword j (0..1, each holds 4 i8 values):
  //      For each in-dword element i (0..3):
  //        a. src_lane = group_base + (4 * j + i)
  //        b. base    = ds_bpermute(src_lane * 4, addr32)
  //        c. val_i8  = LOAD_I8(base + l_in_group)
  //        d. accumulate into outDw[j] at byte slot i.
  //   The direct byte-offset addressing into the source lane's
  //   contiguous data is byte-correct for i8 elements; no `* sizeof`
  //   scaling is needed because each element is one byte wide. Total:
  //   8 ds_bpermute + 8 i8 LDS loads + 8 OR/shifts per lane.
  //
  // EXEC gating: ds_bpermute is convergent and must run with all
  // hardware lanes participating; emit OUTSIDE emitUnderExec, same
  // contract as DS_LOAD_TR16_B128. Inactive-lane reads of the result
  // are not propagated to side effects under a correct source kernel
  // (each lane only reads its own packed-i8 result).
  auto EmitDsLoadTr8B64 = [&]() {
    Value *Addr32 = Op.src(0);
    for (unsigned K = 1; K < Op.nSrcs(); K++) {
      if (Di.isImm(Op.srcIdx(K))) {
        int64_t Imm = Di.getImm(Op.srcIdx(K));
        if (Imm != 0)
          Addr32 = Ctx.B.CreateAdd(Addr32,
                       ConstantInt::get(Ctx.I32Ty, Imm), "ds_off32");
        break;
      }
    }

    Function *MbcntLo = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_mbcnt_lo);
    Function *MbcntHi = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_mbcnt_hi);
    Value *AllOnes = ConstantInt::getSigned(Ctx.I32Ty, -1);
    Value *Zero32 = ConstantInt::get(Ctx.I32Ty, 0);
    Value *Lo = Ctx.B.CreateCall(MbcntLo, {AllOnes, Zero32}, "lane_lo");
    Value *LaneId = Ctx.B.CreateCall(MbcntHi, {AllOnes, Lo}, "lane_id");

    Value *LInGroup = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(7),
                                       "l_in_grp");
    Value *GroupBase = Ctx.B.CreateAnd(LaneId,
        Ctx.B.CreateNot(Ctx.B.getInt32(7)), "grp_base");
    // Each i8 = 1 byte, so the per-element byte offset is just
    // l_in_group; no shift needed.

    Function *Bperm = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_ds_bpermute);

    Type *PtrLdsTy = PointerType::get(Ctx.C, 3);
    auto *I8Ty = Type::getInt8Ty(Ctx.C);

    // SPE-gate the entire bpermute-addressed load + pack + store
    // sequence so phantom lanes don't issue the per-element LDS
    // loads with bpermute-derived addresses that may land outside
    // the WG's LDS allocation.  Same rationale as the simpler
    // DS_READ loads further down (and the handle-flat.cpp global/
    // flat-load fix in the same commit chain): the bpermute
    // selector is `groupBase + 4j + i` where `groupBase` comes
    // from the lane's own `mbcnt_hi`-derived lane id -- for a
    // phantom target lane whose hardware lane index falls outside
    // the source-kernel WG's modelled range, `groupBase` points
    // at an arbitrary "source lane" that may or may not have
    // written anything useful to LDS, and the resulting `ldAddr`
    // can easily exceed the kernel's `group_segment_fixed_size`
    // allocation.
    //
    // Convergence-contract question this gate raises: wrapping
    // `ds_bpermute` in `emitUnderExec` restricts participation to
    // SPE-active lanes, whereas the hardware instruction is
    // convergent and normally (under `init_whole_wave`) has all 64
    // target lanes contributing selectors + source-VGPR data.
    // Why is that restriction safe for this transpose pattern?
    //
    //   * Under a full-wave launch
    //     (`max_flat_workgroup_size >= targetWaveSize`),
    //     `saved_exec` covers every target lane and SPE
    //     never gates any lane off -- HW EXEC stays -1 inside
    //     the diamond.  `ds_bpermute` participation is identical
    //     to the pre-gate behaviour; no semantic drift.
    //
    //   * Under a phantom-lane launch
    //     (`max_flat_workgroup_size < targetWaveSize`, e.g. a
    //     wave32 source kernel's 32-thread WG running on a
    //     wave64 target), the phantom lanes (target lanes
    //     32..63 for a 32-thread WG on wave64) have no
    //     source-kernel workitem.  The source kernel's
    //     TR8/TR16 transpose was written to operate within a
    //     single 8-lane group; those groups live ENTIRELY
    //     inside the first source wave (lanes 0..31 on wave32,
    //     identical target lanes 0..31 under the
    //     phantom-lane-triggered MODREP fallback chosen by
    //     `raiser.cpp`).  Gating the phantom lanes out of the
    //     bpermute therefore preserves the source kernel's
    //     intent exactly -- they were never supposed to
    //     contribute to the transpose; the pre-gate behaviour
    //     of "HW EXEC=-1 forces phantom lanes to contribute
    //     undef-derived VGPRs to the collective" was the bug,
    //     not the fix.
    //
    //   * If a future source kernel ever emits a TR8/TR16
    //     across a >32-lane transpose group (a gfx1250-specific
    //     wave32 construction that crosses from source wave N
    //     to source wave N+1 within a single hardware group),
    //     the gate would over-refuse participation; that shape
    //     doesn't exist in any corpus kernel today and would
    //     need a new projection entry (not a relaxation of the
    //     gate) if it ever surfaced.
    //
    // The `storeVGPR32` destination writes at the end of each
    // iteration use the low-level `ctx.Regs.storeVGPR32` path
    // (not the auto-gating `ctx.storeVGPR32`) so we don't nest a
    // second SPE diamond around each write -- harmless but
    // wasteful IR.
    ParsedReg Dest = Op.dst();
    Ctx.emitUnderExec([&] {
      // 2 output dwords, each holds 4 i8 values from 4 different
      // source lanes within the 8-lane transpose group.
      Value *OutDw[2];
      for (unsigned J = 0; J < 2; J++) {
        Value *Acc = ConstantInt::get(Ctx.I32Ty, 0);
        for (unsigned I = 0; I < 4; I++) {
          Value *SrcLane = Ctx.B.CreateAdd(GroupBase,
                              Ctx.B.getInt32(4 * J + I));
          // ds_bpermute selector is byte-addressed (lane_id << 2).
          Value *Base = Ctx.B.CreateCall(Bperm,
              {Ctx.B.CreateShl(SrcLane, Ctx.B.getInt32(2)), Addr32},
              "bp_base");
          Value *LdAddr = Ctx.B.CreateAdd(Base, LInGroup, "ld_addr");
          Value *Ptr = Ctx.B.CreateIntToPtr(
              Ctx.B.CreateZExt(LdAddr, Ctx.I64Ty), PtrLdsTy, "tr8_p");
          Value *ValI8 = Ctx.B.CreateLoad(I8Ty, Ptr, "tr8_b");
          Value *ValI32 = Ctx.B.CreateZExt(ValI8, Ctx.I32Ty);
          Value *Shifted = Ctx.B.CreateShl(ValI32,
                              Ctx.B.getInt32(8 * I));
          Acc = Ctx.B.CreateOr(Acc, Shifted, "tr8_pack");
        }
        OutDw[J] = Acc;
      }
      for (unsigned J = 0; J < 2; J++)
        Ctx.Regs.storeVGPR32(Ctx.B, Dest.BaseIdx + J, OutDw[J]);
    });

    Hr.Handled = true;
  };

  if (Sop == CanonicalOp::DS_READ_B64_TR_B8 ||
      Sop == CanonicalOp::DS_LOAD_TR8_B64) {
    EmitDsLoadTr8B64();
    return Hr;
  }

  if (Sop == CanonicalOp::DS_LOAD_TR16_B128) {
    Value *Addr = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty, "ds_addr");
    for (unsigned K = 1; K < Op.nSrcs(); K++) {
      if (Di.isImm(Op.srcIdx(K))) {
        int64_t Imm = Di.getImm(Op.srcIdx(K));
        if (Imm != 0)
          Addr = Ctx.B.CreateAdd(Addr, ConstantInt::get(Ctx.I64Ty, Imm), "ds_off");
        break;
      }
    }

    Type *PtrLdsTy = PointerType::get(Ctx.C, 3);

    // Optimized transpose via LDS re-reads.
    // Instead of 32 bpermute+select per transpose (which causes register
    // pressure and spills), bpermute only the base address from each source
    // lane, then do direct i16 LDS loads for the transposed elements.
    // This is 8 bpermute + 8 LDS loads = 16 ops vs 32 bpermute + selects.

    // Compute the 32-bit LDS base address (before the 128-bit load).
    Value *Addr32 = Op.src(0);
    for (unsigned K = 1; K < Op.nSrcs(); K++) {
      if (Di.isImm(Op.srcIdx(K))) {
        int64_t Imm = Di.getImm(Op.srcIdx(K));
        if (Imm != 0)
          Addr32 = Ctx.B.CreateAdd(Addr32,
                       ConstantInt::get(Ctx.I32Ty, Imm), "ds_off32");
        break;
      }
    }

    Function *MbcntLo = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_mbcnt_lo);
    Function *MbcntHi = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_mbcnt_hi);
    Value *AllOnes = ConstantInt::getSigned(Ctx.I32Ty, -1);
    Value *Zero32 = ConstantInt::get(Ctx.I32Ty, 0);
    Value *Lo = Ctx.B.CreateCall(MbcntLo, {AllOnes, Zero32}, "lane_lo");
    Value *LaneId = Ctx.B.CreateCall(MbcntHi, {AllOnes, Lo}, "lane_id");

    // L_in_group = lane_id % 8
    Value *LInGroup = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(7), "l_in_grp");
    // group_base = (lane_id / 8) * 8
    Value *GroupBase = Ctx.B.CreateAnd(LaneId,
        Ctx.B.CreateNot(Ctx.B.getInt32(7)), "grp_base");
    // Byte offset for element L_in_group (each i16 = 2 bytes)
    Value *ElemOff = Ctx.B.CreateShl(LInGroup, Ctx.B.getInt32(1), "elem_off");

    Function *Bperm = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_ds_bpermute);

    auto *I16Ty = Type::getInt16Ty(Ctx.C);

    // SPE-gate the bpermute-addressed per-element LDS loads + pack
    // + store sequence (same rationale as the DS_LOAD_TR8 gating
    // above).  Phantom lanes on a sub-wave-width launch would
    // otherwise issue `CreateLoad(i16Ty, ptrLo, ...)` and
    // `CreateLoad(i16Ty, ptrHi, ...)` against ds_bpermute-derived
    // LDS addresses that fall outside the WG's
    // `group_segment_fixed_size` allocation; the stores at the end
    // of the loop use low-level `ctx.Regs.storeVGPR32` instead of
    // the auto-gating `ctx.storeVGPR32` to avoid nesting a second
    // SPE diamond around each write.
    ParsedReg Dest = Op.dst();
    Ctx.emitUnderExec([&] {
      Value *OutDw[4];
      for (unsigned J = 0; J < 4; J++) {
        Value *SrcLo = Ctx.B.CreateAdd(GroupBase, Ctx.B.getInt32(2 * J));
        Value *SrcHi = Ctx.B.CreateAdd(GroupBase, Ctx.B.getInt32(2 * J + 1));

        // Get source lane's LDS base address via ds_bpermute.
        Value *BaseLo = Ctx.B.CreateCall(Bperm,
            {Ctx.B.CreateShl(SrcLo, Ctx.B.getInt32(2)), Addr32},
            "bp_base_lo");
        Value *BaseHi = Ctx.B.CreateCall(Bperm,
            {Ctx.B.CreateShl(SrcHi, Ctx.B.getInt32(2)), Addr32},
            "bp_base_hi");

        // LDS address for element L_in_group in source lane's contiguous data.
        Value *LdAddrLo = Ctx.B.CreateAdd(BaseLo, ElemOff, "ld_addr_lo");
        Value *LdAddrHi = Ctx.B.CreateAdd(BaseHi, ElemOff, "ld_addr_hi");

        // Load i16 from LDS (address space 3).
        Value *PtrLo = Ctx.B.CreateIntToPtr(
            Ctx.B.CreateZExt(LdAddrLo, Ctx.I64Ty), PtrLdsTy, "tr_p_lo");
        Value *PtrHi = Ctx.B.CreateIntToPtr(
            Ctx.B.CreateZExt(LdAddrHi, Ctx.I64Ty), PtrLdsTy, "tr_p_hi");
        Value *ValLo = Ctx.B.CreateLoad(I16Ty, PtrLo, "tr_lo");
        Value *ValHi = Ctx.B.CreateLoad(I16Ty, PtrHi, "tr_hi");

        // Pack two i16 into one i32: (hi << 16) | lo
        Value *Lo32 = Ctx.B.CreateZExt(ValLo, Ctx.I32Ty);
        Value *Hi32 = Ctx.B.CreateZExt(ValHi, Ctx.I32Ty);
        OutDw[J] = Ctx.B.CreateOr(
            Ctx.B.CreateShl(Hi32, Ctx.B.getInt32(16)), Lo32, "tr_out");
      }

      for (unsigned J = 0; J < 4; J++)
        Ctx.Regs.storeVGPR32(Ctx.B, Dest.BaseIdx + J, OutDw[J]);
    });

    Hr.Handled = true;
    return Hr;
  }
  // ---------------------------------------------------------------------
  // DS_READ2 / DS_WRITE2 family -- two independent memory accesses at
  // per-offset, unit-scaled byte addresses.
  //
  // Hardware contract (DSInstructions.td DS_1A_Off8_RET /
  // DS_1A2D_Off8_NORET base classes):
  //
  //   Access 0 target byte address = vaddr + offset0 * unit
  //   Access 1 target byte address = vaddr + offset1 * unit
  //
  // where `offsetN` is the raw 8-bit MCInst field value (0..255) and
  // `unit` is fixed per opcode:
  //
  //     DS_READ2_B32      / DS_WRITE2_B32      : unit =   4  (dword)
  //     DS_READ2_B64      / DS_WRITE2_B64      : unit =   8  (qword)
  //     DS_READ2ST64_B32  / DS_WRITE2ST64_B32  : unit = 256  (64 dwords)
  //     DS_READ2ST64_B64  / DS_WRITE2ST64_B64  : unit = 512  (64 qwords)
  //
  // The two accesses are semantically INDEPENDENT: the compiler selects
  // DS_READ2_B32 with a gap routinely (e.g. `offset0:0 offset1:2` reads
  // bytes 0 and 8, leaving bytes 4-7 untouched), and the *ST64 variants
  // are specifically designed to reach byte offsets a plain
  // offset0+1-raw DS_READ2 cannot (every raw-field-1 increment jumps
  // 64 dwords in the *ST64 case vs 1 dword in the non-*ST64 case).
  //
  // Prior lift shape -- a single contiguous `<N x i32>` load at
  // `vaddr + offset0` (byte offset *unscaled*, offset1 ignored) -- was
  // a silent miscompile for any (offset0,offset1) pair other than the
  // happy (0,1) case, where the two bugs cancel because bytes 0..7 of
  // a contiguous `<2 x i32>` load coincide with the hardware's bytes 0
  // and 4 reads. The corpus repeatedly violates that happy case (see
  // e.g. kerneldex scope_discovery___matmul_ogs_*: `offset0:4 offset1:6`,
  // `offset0:64 offset1:66`, …). This block therefore emits two
  // independent loads/stores at the correctly-scaled byte addresses,
  // writing the two results into the dest VGPR pair in MCInst order
  // (dw0 <- access 0, dw1 <- access 1 for B32; four-dword destination
  // in lo/hi pairs for B64).
  //
  // MC operand layouts (from DSInstructions.td):
  //
  //   DS_1A_Off8_RET     : (outs vdst) (ins addr, offset0, offset1, gds)
  //   DS_1A2D_Off8_NORET : (outs)      (ins addr, data0, data1,
  //                                          offset0, offset1, gds)
  //
  // `buildSrcMap` records all non-modifier sources in MCInst order, so
  //   READ2  : srcMap = [addr,   offset0, offset1, gds]
  //   WRITE2 : srcMap = [addr,   data0,   data1,   offset0, offset1, gds]
  //
  // We resolve the two immediate offsets via `OpName::offset0` /
  // `OpName::offset1` rather than positional srcMap indexing -- the
  // named lookup documents intent and is robust against future srcMap
  // layout changes.
  //
  // EXEC gating mirrors the single-offset DS_READ_*/DS_WRITE_* paths:
  // reads are not EXEC-guarded (writing to a dead destination VGPR on
  // inactive lanes is semantically equivalent under the scalar SPMD
  // model); writes go through `emitUnderExec` so that LDS side effects
  // are confined to active lanes.
  //
  // Alignment on the emitted loads/stores equals the access width
  // (Align(4) for B32, Align(8) for B64). AMDGPU LDS hardware requires
  // that granularity and the source ISA's encoding guarantees it;
  // stating it explicitly on `CreateAlignedLoad` / `CreateAlignedStore`
  // prevents the backend from falling back to a conservative Align(1)
  // and emitting byte-granular expansions.
  auto Ds2Classify = [](CanonicalOp S) -> std::tuple<bool /*isRead*/,
                                                int  /*widthBits*/,
                                                int  /*unitBytes*/> {
    switch (S) {
    case CanonicalOp::DS_READ2_B32:       return {true,  32,   4};
    case CanonicalOp::DS_READ2_B64:       return {true,  64,   8};
    case CanonicalOp::DS_READ2ST64_B32:   return {true,  32, 256};
    case CanonicalOp::DS_READ2ST64_B64:   return {true,  64, 512};
    case CanonicalOp::DS_WRITE2_B32:      return {false, 32,   4};
    case CanonicalOp::DS_WRITE2_B64:      return {false, 64,   8};
    case CanonicalOp::DS_WRITE2ST64_B32:  return {false, 32, 256};
    case CanonicalOp::DS_WRITE2ST64_B64:  return {false, 64, 512};
    default:                        return {false,  0,   0};
    }
  };
  {
    auto [ds2IsRead, ds2WidthBits, ds2UnitBytes] = Ds2Classify(Sop);
    if (ds2UnitBytes > 0) {
      unsigned Opc = Di.Inst.getOpcode();
      int Off0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset0);
      int Off1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset1);
      if (Off0Idx < 0 || Off1Idx < 0 ||
          static_cast<unsigned>(Off0Idx) >= Di.Inst.getNumOperands() ||
          static_cast<unsigned>(Off1Idx) >= Di.Inst.getNumOperands() ||
          !Di.Inst.getOperand(static_cast<unsigned>(Off0Idx)).isImm() ||
          !Di.Inst.getOperand(static_cast<unsigned>(Off1Idx)).isImm()) {
        Hr.Failure = RaiseFailure::unsupportedInstructionForm(
            Di, "DS",
            "DS_READ2/WRITE2 missing OpName::offset0 or OpName::offset1 "
            "immediate operand -- operand table mismatch");
        return Hr;
      }
      int64_t RawOff0 =
          Di.Inst.getOperand(static_cast<unsigned>(Off0Idx)).getImm();
      int64_t RawOff1 =
          Di.Inst.getOperand(static_cast<unsigned>(Off1Idx)).getImm();
      int64_t ByteOff0 = RawOff0 * ds2UnitBytes;
      int64_t ByteOff1 = RawOff1 * ds2UnitBytes;

      Value *Vaddr = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty, "ds2_addr");
      auto *LdsPtrTy = PointerType::get(Ctx.C, 3);
      auto MakePtr = [&](int64_t ByteOff, const char *Name) -> Value * {
        Value *A = ByteOff == 0
                       ? Vaddr
                       : Ctx.B.CreateAdd(Vaddr,
                                         ConstantInt::get(Ctx.I64Ty, ByteOff),
                                         "ds2_off");
        return Ctx.B.CreateIntToPtr(A, LdsPtrTy, Name);
      };
      Value *Ptr0 = MakePtr(ByteOff0, "ds2_p0");
      Value *Ptr1 = MakePtr(ByteOff1, "ds2_p1");
      Align Access(static_cast<uint64_t>(ds2WidthBits / 8));

      if (ds2IsRead) {
        ParsedReg Dest = Op.dst();
        auto SubReg = [&Dest](int Off) {
          ParsedReg R = Dest;
          R.BaseIdx = Dest.BaseIdx + Off;
          R.Width = 1;
          return R;
        };
        // SPE-gate the pair of DS loads so phantom lanes (their
        // tid-derived ptr0 / ptr1 addresses point to unmodelled
        // offsets outside the WG's LDS allocation) don't execute
        // the loads.  Matches the ds_write gating in the `else`
        // branch below and the handle-flat.cpp global/flat-load
        // fix from the same commit.
        Ctx.emitUnderExec([&] {
          if (ds2WidthBits == 32) {
            Value *V0 = Ctx.B.CreateAlignedLoad(Ctx.I32Ty, Ptr0, Access,
                                                 "ds2_ld0");
            Value *V1 = Ctx.B.CreateAlignedLoad(Ctx.I32Ty, Ptr1, Access,
                                                 "ds2_ld1");
            Ctx.Regs.writeReg32(Ctx.B, SubReg(0), V0);
            Ctx.Regs.writeReg32(Ctx.B, SubReg(1), V1);
          } else { // 64
            Value *V0 = Ctx.B.CreateAlignedLoad(Ctx.I64Ty, Ptr0, Access,
                                                 "ds2_ld0");
            Value *V1 = Ctx.B.CreateAlignedLoad(Ctx.I64Ty, Ptr1, Access,
                                                 "ds2_ld1");
            Value *Lo0 = Ctx.B.CreateTrunc(V0, Ctx.I32Ty, "ds2_ld0_lo");
            Value *Hi0 = Ctx.B.CreateTrunc(
                Ctx.B.CreateLShr(V0, ConstantInt::get(Ctx.I64Ty, 32)),
                Ctx.I32Ty, "ds2_ld0_hi");
            Value *Lo1 = Ctx.B.CreateTrunc(V1, Ctx.I32Ty, "ds2_ld1_lo");
            Value *Hi1 = Ctx.B.CreateTrunc(
                Ctx.B.CreateLShr(V1, ConstantInt::get(Ctx.I64Ty, 32)),
                Ctx.I32Ty, "ds2_ld1_hi");
            Ctx.Regs.writeReg32(Ctx.B, SubReg(0), Lo0);
            Ctx.Regs.writeReg32(Ctx.B, SubReg(1), Hi0);
            Ctx.Regs.writeReg32(Ctx.B, SubReg(2), Lo1);
            Ctx.Regs.writeReg32(Ctx.B, SubReg(3), Hi1);
          }
        });
        Hr.Handled = true;
        return Hr;
      } else {
        ParsedReg Data0 = Op.srcReg(1);
        ParsedReg Data1 = Op.srcReg(2);
        if (ds2WidthBits == 32) {
          Value *V0 = Ctx.Regs.readReg32(Ctx.B, Data0);
          Value *V1 = Ctx.Regs.readReg32(Ctx.B, Data1);
          Ctx.emitUnderExec([&] {
            Ctx.B.CreateAlignedStore(V0, Ptr0, Access);
            Ctx.B.CreateAlignedStore(V1, Ptr1, Access);
          });
        } else { // 64
          Value *V0 = Ctx.Regs.readReg64(Ctx.B, Data0);
          Value *V1 = Ctx.Regs.readReg64(Ctx.B, Data1);
          Ctx.emitUnderExec([&] {
            Ctx.B.CreateAlignedStore(V0, Ptr0, Access);
            Ctx.B.CreateAlignedStore(V1, Ptr1, Access);
          });
        }
        Hr.Handled = true;
        return Hr;
      }
    }
  }

  bool IsDsRead = Sop >= CanonicalOp::DS_READ_B32 && Sop <= CanonicalOp::DS_READ_I8;
  bool IsDsWrite = Sop >= CanonicalOp::DS_WRITE_B32 && Sop <= CanonicalOp::DS_WRITE_B8;
  if (IsDsRead || IsDsWrite) {
    auto [dwords, loadBits, isSigned] = DsClassify(Sop);

    // dsClassify returns `{-1, 0, false}` for any CanonicalOp it doesn't
    // model as single-offset (notably the DS_READ2/WRITE2 family,
    // whose SemOps sit inside the DS_READ_*/DS_WRITE_* enum range
    // but are intercepted by the dedicated two-offset block above).
    // Reaching here with `dwords < 0` means a new DS CanonicalOp landed in
    // the enum range without a classifier entry and without a
    // dedicated-block intercept -- loud-fail before the generic
    // vector-load path would otherwise produce a bogus
    // `FixedVectorType::get(i32, (unsigned)-1)` crash.
    if (dwords < 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "DS",
          "single-offset DS generic path reached with an unclassified "
          "CanonicalOp -- add a dsClassify entry or a dedicated handler block");
      return Hr;
    }

    Value *Addr = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty, "ds_addr");

    for (unsigned K = 1; K < Op.nSrcs(); K++) {
      if (Di.isImm(Op.srcIdx(K))) {
        int64_t Imm = Di.getImm(Op.srcIdx(K));
        if (Imm != 0)
          Addr = Ctx.B.CreateAdd(Addr, ConstantInt::get(Ctx.I64Ty, Imm), "ds_off");
        break;
      }
    }

    Value *Ptr = Ctx.B.CreateIntToPtr(Addr, PointerType::get(Ctx.C, 3));

    if (IsDsRead) {
      ParsedReg Dest = Op.dst();
      // SPE-gate the DS load itself, not just the VGPR write-back.
      // The ds_write counterpart below (~line 547) is already wrapped
      // in `emitUnderExec`; the asymmetric pre-2026-04-22 handling
      // (loads outside, stores inside) meant WaveNative phantom
      // lanes -- whose VGPR-derived LDS addresses are `undef` /
      // stale-slot data -- would issue DS loads against arbitrary
      // offsets.  Out-of-WG-LDS-allocation offsets are UB on the
      // hardware (typically returning 0, but the specific WG's LDS
      // slab could overlap another WG's and expose cross-WG data);
      // in-range offsets would read from some *other* lane's slot
      // within the same WG and silently corrupt downstream compute.
      // Gating matches the global-load fix in `handle-flat.cpp`
      // (same commit) that addressed the matmul_fp16 HIP-700 fault.
      Ctx.emitUnderExec([&] {
        if (dwords == 0) {
          Type *MemTy = Type::getIntNTy(Ctx.C, loadBits);
          Value *V = Ctx.B.CreateLoad(MemTy, Ptr, "ds_ld");
          Ctx.Regs.writeReg32(
              Ctx.B, Dest,
              isSigned ? Ctx.B.CreateSExt(V, Ctx.I32Ty)
                       : Ctx.B.CreateZExt(V, Ctx.I32Ty));
        } else if (dwords == 1) {
          Ctx.Regs.writeReg32(Ctx.B, Dest,
                              Ctx.B.CreateLoad(Ctx.I32Ty, Ptr, "ds_ld"));
        } else {
          auto *VecTy = FixedVectorType::get(Ctx.I32Ty, dwords);
          Ctx.Regs.writeRegVec(Ctx.B, Dest,
                                Ctx.B.CreateLoad(VecTy, Ptr, "ds_ld"));
        }
      });
      Hr.Handled = true;
    return Hr;
    }
    if (IsDsWrite) {
      ParsedReg StData = Op.srcReg(1);
      if (dwords == 0) {
        Type *MemTy = Type::getIntNTy(Ctx.C, loadBits);
        Value *Val = Ctx.B.CreateTrunc(Ctx.Regs.readReg32(Ctx.B, StData), MemTy);
        Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Ptr); });
      } else if (dwords == 1) {
        Value *Val = Ctx.Regs.readReg32(Ctx.B, StData);
        Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Ptr); });
      } else {
        auto *VecTy = FixedVectorType::get(Ctx.I32Ty, dwords);
        Value *Val = Ctx.Regs.readRegVec(Ctx.B, StData, VecTy);
        Ctx.emitUnderExec([&] { Ctx.B.CreateStore(Val, Ptr); });
      }
      Hr.Handled = true;
    return Hr;
    }
  }
  // D16_HI partial-store family: ds_store_b16_d16_hi /
  // ds_store_b8_d16_hi (DSInstructions.td §604-606, gfx8+ behind
  // SubtargetPredicate=HasD16LoadStore). Both shift the source
  // VGPR right by 16 to surface its UPPER half, then truncate to
  // 16 or 8 bits and store to LDS at addr=src(0)+immOffset.
  //
  // The corpus pattern (tensilelite gemm tile spill, 18 kernels)
  // is `ds_store_b16_d16_hi v0, v1 offset:N`: pack the upper f16
  // of a packed-bf16 register pair into LDS as part of the K-strip
  // tile redistribution. A regression that wrote bits [15:0]
  // instead of [31:16] would silently corrupt every other tile.
  //
  // Test back-reference: lit_tests/ds_store_b16_d16_hi/ pins the
  // i32 -> lshr 16 -> trunc i16 -> store-under-EXEC chain. The
  // `ds_st_d16_hi` value-name on the trunc is the canonical
  // breadcrumb a maintainer can grep for.
  if (Sop == CanonicalOp::DS_WRITE_B16_D16_HI ||
      Sop == CanonicalOp::DS_WRITE_B8_D16_HI) {
    Value *Addr = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty, "ds_addr");
    for (unsigned K = 1; K < Op.nSrcs(); K++) {
      if (Di.isImm(Op.srcIdx(K))) {
        int64_t Imm = Di.getImm(Op.srcIdx(K));
        if (Imm != 0)
          Addr = Ctx.B.CreateAdd(Addr, ConstantInt::get(Ctx.I64Ty, Imm),
                                  "ds_off");
        break;
      }
    }
    Value *Ptr = Ctx.B.CreateIntToPtr(Addr, PointerType::get(Ctx.C, 3));

    ParsedReg StData = Op.srcReg(1);
    Value *Raw = Ctx.Regs.readReg32(Ctx.B, StData);
    // Surface the UPPER half of the source VGPR. For B16_D16_HI
    // this is bits [31:16]; for B8_D16_HI this is bits [23:16].
    // The lshr 16 + trunc-to-16 step is identical for both -- the
    // only difference is a further trunc to i8 for the B8 variant.
    Value *Hi16 = Ctx.B.CreateTrunc(
        Ctx.B.CreateLShr(Raw, Ctx.B.getInt32(16), "ds_st_hi16_shr"),
        Type::getInt16Ty(Ctx.C), "ds_st_d16_hi");
    Value *ToStore = (Sop == CanonicalOp::DS_WRITE_B8_D16_HI)
                          ? Ctx.B.CreateTrunc(Hi16, Type::getInt8Ty(Ctx.C),
                                              "ds_st_d8_hi")
                          : Hi16;
    Ctx.emitUnderExec([&] { Ctx.B.CreateStore(ToStore, Ptr); });
    Hr.Handled = true;
    return Hr;
  }

  // ds_add_f64 / ds_add_rtn_f64. LDS 64-bit FP add, native on
  // gfx90a/gfx940-family; on subtargets without it AtomicExpandPass
  // expands `atomicrmw fadd double addrspace(3)*` into a
  // ds_cmpst_b64-based CAS loop. The `_RTN` form publishes the
  // pre-add memory value to a dst VGPR pair (di.NumDefs > 0); the
  // non-RTN form discards the result.
  if (Sop == CanonicalOp::DS_ADD_F64) {
    assert(((Di.TsFlags & SIInstrFlags::IsAtomicRet) != 0) == (Di.NumDefs > 0) &&
           "ds_add_f64: IsAtomicRet disagrees with numDefs");
    Value *Addr = Ctx.B.CreateZExt(Op.src(0), Ctx.I64Ty, "ds_addr");
    for (unsigned K = 1; K < Op.nSrcs(); K++) {
      if (Di.isImm(Op.srcIdx(K))) {
        int64_t Imm = Di.getImm(Op.srcIdx(K));
        if (Imm != 0)
          Addr = Ctx.B.CreateAdd(Addr, ConstantInt::get(Ctx.I64Ty, Imm),
                                  "ds_off");
        break;
      }
    }
    Value *Ptr = Ctx.B.CreateIntToPtr(Addr, PointerType::get(Ctx.C, 3));
    // vdata is a 2-VGPR pair; read as i64, bitcast to f64 for the FP
    // atomic.
    ParsedReg StData = Op.srcReg(1);
    Value *Data = Ctx.B.CreateBitCast(
        Ctx.Regs.readReg64(Ctx.B, StData), Ctx.F64Ty);
    Ctx.emitUnderExec([&] {
      auto *Rmw = Ctx.B.CreateAtomicRMW(
          AtomicRMWInst::FAdd, Ptr, Data, MaybeAlign(),
          AtomicOrdering::SequentiallyConsistent);
      if (Di.NumDefs > 0)
        Ctx.Regs.writeReg64(Ctx.B, Op.dst(),
                            Ctx.B.CreateBitCast(Rmw, Ctx.I64Ty));
    });
    Hr.Handled = true;
    return Hr;
  }

  if (Sop == CanonicalOp::DS_BPERMUTE_B32) {
    // Backwards permute: per-lane GATHER. Each lane reads the `src1`
    // value from a *source* lane whose index is `src0 >> 2` (the
    // selector is pre-scaled by the source compiler to match DS byte
    // addressing). The previous handler was an identity copy of
    // src(1), which silently collapsed `__shfl_xor(x, 1)` to `x` and
    // any shuffle-based reduction to "every lane keeps its own
    // partial" -- exactly the `lane_swap` / `block_sum_shfl`
    // compare_correctness failures. Lower through the native
    // `amdgcn.ds_bpermute` intrinsic so the backend emits the real
    // cross-lane gather on the target ISA.
    //
    // Wave32-source selectors are source-wave-local byte offsets. Under
    // WaveNative cross-widening, one wave64 target wave carries two source
    // wave32 instances (lanes 0..31 and 32..63). A raw target `ds_bpermute`
    // would interpret selector 0 from lane 32 as "read hardware lane 0",
    // but source gfx1250 semantics mean "read lane 0 of this source wave",
    // i.e. hardware lane 32. Rebase the selector into the current source-wave
    // half before calling the wave64 intrinsic.
    //
    // EXEC gating. We emit the intrinsic *outside* `emitUnderExec`.
    // `amdgcn.ds_bpermute` is convergent -- all lanes of the hardware
    // wave must participate or the result in inactive lanes is
    // undefined. For lanes that are inactive in the source kernel,
    // their `src1` input is the ambient VGPR value (possibly the
    // 0xA5A5… sentinel), but since no lane *reads* from an inactive
    // lane under a correct selector, the undef propagation does not
    // affect the active-lane outputs. If a future handler needs to
    // emit `ds_bpermute` on a value that was written inside an SPE
    // diamond, it must first broadcast the diamond result through a
    // `readfirstlane` / explicit VGPR move outside the diamond,
    // otherwise the cross-lane read will pick up `undef`.
    Value *Index = Op.src(0);
    if (Ctx.Projection.numSourceWavesPerTarget() > 1 &&
        Ctx.Isa.isWave32() && !Ctx.TargetIsa.isWave32()) {
      constexpr uint32_t kSourceWaveLanes = 32;
      constexpr uint32_t kDwordBytes = 4;
      constexpr uint32_t kSourceWaveBytes = kSourceWaveLanes * kDwordBytes;
      Value *LocalIndex = Ctx.B.CreateAnd(
          Index, Ctx.B.getInt32(kSourceWaveBytes - 1), "bperm_local_addr");
      Value *LaneId = Ctx.emitLaneIdx();
      Value *SourceWaveLaneBase = Ctx.B.CreateAnd(
          LaneId, Ctx.B.getInt32(~(kSourceWaveLanes - 1)),
          "bperm_srcwave_lane_base");
      Value *SourceWaveByteBase = Ctx.B.CreateShl(
          SourceWaveLaneBase, Ctx.B.getInt32(2), "bperm_srcwave_byte_base");
      Index = Ctx.B.CreateOr(LocalIndex, SourceWaveByteBase,
                             "bperm_srcwave_addr");
    }
    Value *Src = Op.src(1);
    Function *Bperm = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
    Value *Gathered = Ctx.B.CreateCall(Bperm, {Index, Src}, "bperm");
    Ctx.writeReg32(Op.dst(), Gathered);
    Hr.Handled = true;
    return Hr;
  }
  if (Sop == CanonicalOp::DS_SWIZZLE_B32) {
    // P6 lowering -- see the ds_swizzle_b32 row of hotswap/docs/wave-
    // size-translation.md §5.3: lift `ds_swizzle_b32` through
    // `llvm.amdgcn.ds.swizzle`. The intrinsic signature is
    //   declare i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 immarg %offset)
    // (`ImmArg<ArgIndex<1>>`), so the second operand MUST be a
    // ConstantInt -- we read the 16-bit `offset` field from the MC
    // operand table and materialise it as `i32 immarg`.
    //
    // MODREP: wave-width projection. The 16-bit swizzle imm encodes
    // one of seven swizzle modes (SIDefines.h `Swizzle::Id`),
    // partitioned into four valid top-nibble/byte envelopes plus a
    // RESERVED region:
    //
    //   QUAD_PERM   (top byte = 0x80)   -- independent 4-lane quads.
    //   BITMASK_PERM (top bit = 0)      -- per-lane swizzle via three
    //                                      5-bit AND/OR/XOR masks;
    //                                      includes BROADCAST / SWAP
    //                                      / REVERSE sub-encodings.
    //   FFT_MODE    (top nibble = 0xE)  -- 5-bit FFT swizzle selector.
    //   ROTATE_MODE (top nibble = 0xC)  -- 5-bit size + 1-bit dir.
    //   RESERVED    (top nibble in
    //                {0x9,0xA,0xB,0xD,0xF}) -- undefined semantics.
    //
    // All four valid envelopes are wave-size-oblivious under modulo-
    // replication. The argument is the same for all four: wave64
    // ds_swizzle hardware preserves bit 5 of the lane id, so each
    // 32-lane half independently performs the same permutation. The
    // bit-5 preservation is a hardware contract documented neither
    // in IntrinsicsAMDGPU.td nor in AMDGPUUsage.rst, but it is a
    // wave64-GPU-family-wide property:
    //
    //   * The encoding (16-bit `Swizzle::EncBits`) has been stable
    //     since gfx7 (the first GPU with `ds_swizzle_b32`); the
    //     same single MC opcode (DS_SWIZZLE_B32) covers every
    //     wave64 target the transpiler can emit to (gfx7 hawaii,
    //     gfx8 fiji, gfx9 vega, gfx90a MI200, gfx942 CDNA3, gfx950).
    //   * Upstream LLVM's `test/CodeGen/AMDGPU/llvm.amdgcn.ds.
    //     swizzle.ll` runs on hawaii (gfx7 wave64) and fiji (gfx8
    //     wave64) and asserts the intrinsic lowers to a SINGLE
    //     `ds_swizzle_b32 ... offset:swizzle(...)` instruction with
    //     no extra arithmetic. If wave64 hardware did not preserve
    //     bit 5, the upper-half lanes would need fixup code to
    //     match the lower-half pattern, and that test (and every
    //     downstream wave64 user since gfx7) would have caught it.
    //   * Verified directly on gfx942 (CDNA3) wave64 across all
    //     four envelopes via inline-asm probes (results below); the
    //     observed per-32-lane-half independence is consistent with
    //     LLVM's wave64-family-wide assumption.
    //
    // The combined argument -- encoding stability + LLVM's wave64-
    // family-wide implicit reliance + direct empirical verification
    // on gfx942 -- is what justifies applying the safe-set decision
    // to every wave64 target the transpiler emits to. If the
    // contract is ever re-questioned for a specific target,
    // regenerate the probes from the documentation below for that
    // target.
    //
    // Per-envelope empirical evidence (gfx942 wave64):
    //
    //   * BROADCAST(32, 5) (imm=0x00A0, BITMASK_PERM): lanes 0..31
    //     -> lane 5, lanes 32..63 -> lane 37 (= 32+5).
    //   * SWAP-1           (imm=0x041F, BITMASK_PERM): lane 32<->33,
    //     34<->35, …, 62<->63. Each 32-lane half pairs internally.
    //   * FFT 0x00         (imm=0xE000): 5-bit-reverse within each
    //     32-lane half (lanes 32..63 produce lower-half-result + 32).
    //   * FFT 0x10         (imm=0xE010): 4-bit-reverse within each
    //     16-lane group, again per-half-independent.
    //   * FFT 0x1F         (imm=0xE01F): identity within each half.
    //   * ROTATE-LEFT-1    (imm=0xC020): lane 31 -> lane 0 (lower
    //     half), lane 63 -> lane 32 (upper half).
    //   * ROTATE-LEFT-31   (imm=0xC3E0): lane 0 -> lane 31 (lower
    //     half), lane 32 -> lane 63 (upper half).
    //
    // Reproduction: small HIP probes that issue `ds_swizzle_b32` via
    // inline asm against per-lane lane-id VGPRs and print each
    // lane's destination. Every probe confirms the upper-half
    // independence the modulo-replication argument requires.
    //
    // Historical regression gate: a wave32 source kernel using
    // `ds_swizzle_b32 offset:swizzle(SWAP,2)`
    // and runs it on gfx942 wave64, verifying the per-32-lane-half
    // XOR-2 BITMASK_PERM swizzle pattern across all 64 lanes. The
    // SWAP-2 pattern is intentionally distinct from the GPT-OSS
    // `sum_bitmatrix_rows` corpus pattern (SWAP-1) so the test
    // catches imm-extraction bugs that happen to round-trip
    // SWAP-1.
    //
    // The classifier (`dsSwizzleSafeForModRep`) accepts all four
    // valid envelopes and refuses only the RESERVED top-nibble
    // region, where hardware semantics are undefined and a silent
    // lift would map the source kernel's imm onto whatever the
    // wave64 backend happens to do. Phase 1.4.5 surfaces the
    // refusal with a `RESERVED top-nibble envelope` detail line so
    // operators can identify the offending imm at triage time.
    //
    // Same-wave raises bypass the classifier (it early-returns when
    // src/tgt wave sizes match) and reach here unconditionally; for
    // same-wave every imm is correct by construction (the source and
    // target hardware execute the same `ds_swizzle_b32`
    // byte-for-byte), so the same intrinsic emit covers both paths.
    //
    // gfx942 backend support: confirmed via a separate verification
    // run -- the lifted IR `call i32 @llvm.amdgcn.ds.swizzle(i32 %v,
    // i32 1055)` lowers through `llc -mcpu=gfx942
    // -mattr=+wavefrontsize64` to a direct `ds_swizzle_b32 ...
    // offset:swizzle(SWAP,1)` instruction with `.wavefront_size: 64`
    // metadata. No emulation needed (unlike P2's permlane16 path,
    // where the gfx942 backend lacked native isel and we routed
    // through ds_bpermute).
    //
    // EXEC gating: `amdgcn.ds.swizzle` is convergent (`isConvergent`
    // in DSInstructions.td); same convergence reasoning as the
    // `ds_bpermute` handler above applies -- emit OUTSIDE
    // `emitUnderExec` so all hardware lanes participate, and trust
    // that inactive-lane reads of the result do not feed any
    // observable side effect under a correct source kernel.
    // The 16-bit imm is extracted once at decode time into
    // `di.dsSwizzleImm` (see `decode.cpp::decodeDsSwizzleImm`); the
    // decoder enforces the unsigned 16-bit range and refuses to
    // populate the field on missing / non-immediate / out-of-range
    // operands. A `!di.hasDsSwizzleImm` here means the decoder
    // rejected this exact site, which the cross-wave classifier
    // already mirrors as a refusal -- but a same-wave raise bypasses
    // the classifier and still reaches us, so refuse loudly here too
    // for symmetry with the cross-wave path.
    if (!Di.HasDsSwizzleImm) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "DS",
          "ds_swizzle_b32 missing/invalid OpName::offset immediate "
          "operand -- decoder rejected the 16-bit imm");
      return Hr;
    }
    // Read the data input via `OpName::addr` rather than positional
    // `op.src(0)`. ds_swizzle_b32's MCInst layout per
    // DSInstructions.td's DS_1A_RET base class is `(outs vdst), (ins
    // VGPR_32:$addr, Offset:$offset, gds)`; `buildSrcMap` (decode.cpp)
    // walks all non-modifier post-vdst operands, so srcMap[0] = $addr,
    // srcMap[1] = $offset, srcMap[2] = $gds. Using `op.src(0)` would
    // therefore work, but the named lookup is more explicit about
    // which operand is the data and is robust to future srcMap
    // refactors. (The other DS handlers in this file still use
    // positional `op.src(0)` for consistency with their existing
    // patterns; the named-lookup audit there is a system-wide cleanup
    // outside the scope of P6.)
    int AddrIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                              AMDGPU::OpName::addr);
    if (AddrIdx < 0 ||
        static_cast<unsigned>(AddrIdx) >= Di.Inst.getNumOperands() ||
        !Di.Inst.getOperand(static_cast<unsigned>(AddrIdx)).isReg()) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "DS",
          "ds_swizzle_b32 missing OpName::addr VGPR operand -- operand "
          "table mismatch");
      return Hr;
    }
    Value *Src = Ctx.readOp32(Di, static_cast<unsigned>(AddrIdx));
    Function *Swiz = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_ds_swizzle);
    Value *Result = Ctx.B.CreateCall(
        Swiz,
        {Src, ConstantInt::get(Ctx.I32Ty, Di.DsSwizzleImm)},
        "ds_swiz");
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  return Hr;
}

} // namespace COMGR::hotswap
