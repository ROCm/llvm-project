//===- opcode-map.cpp - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "opcode-map.h"

#include <cstdint>
#include <optional>
#include <string>

// AMDGPU target-private headers. They expose:
//   AMDGPU::getMCOpcode           (declared in Utils/AMDGPUBaseInfo.h)
//   AMDGPU::getVOPe64 / getVOPe32 / getDPPOp32 / getDPPOp64 /
//   getSDWAOp / getBasicFromSDWAOp / getGlobalVaddrOp
//                                  (declared in SIInstrInfo.h, implemented
//                                   in the TableGen-generated
//                                   AMDGPUGenInstrInfo.inc under
//                                   `#define GET_INSTRMAP_INFO`, linked from
//                                   libLLVMAMDGPUUtils.a).
//
// SIInstrInfo.h drags in the CodeGen TargetInstrInfo base, which we do not
// use at runtime, but pulling it in is preferable to hand-rolling forward
// declarations that would silently go stale if LLVM changes a signature.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Maps a canonical AMDGPU pseudo opcode to the CanonicalOp the raiser dispatches on.
// The canonical form is what comes out of the canonicalization chain below:
//   MC opcode -> pseudo
//   pseudo    -> e64 (if VOP/VOPC has e32/e64 split)
//   pseudo    -> base (if SDWA/DPP)
//   pseudo    -> VADDR (if FLAT/GLOBAL SADDR)
//
// Using AMDGPU:: enum constants instead of strings gives us compile-time
// checking: if LLVM renames a pseudo, the build fails here rather than the
// lookup silently returning CanonicalOp::Unknown at runtime.
struct Entry {
  unsigned Opc;
  CanonicalOp Sem;
};

// Convenience macros for families with many addressing-mode variants that
// LLVM does not expose a single canonicalization helper for.
#define E(OP, SEM) Entry{AMDGPU::OP, CanonicalOp::SEM}

// MUBUF: four addressing modes (OFFSET/OFFEN/IDXEN/BOTHEN) per base opcode,
// plus the gfx12+ VBUFFER fork that was introduced to separate the buffer
// descriptor source.
#define MUBUF4(BASE, SEM) \
  E(BASE##_OFFSET, SEM), \
  E(BASE##_OFFEN,  SEM), \
  E(BASE##_IDXEN,  SEM), \
  E(BASE##_BOTHEN, SEM)
#define VBUF4(BASE, SEM) \
  E(BASE##_VBUFFER_OFFSET, SEM), \
  E(BASE##_VBUFFER_OFFEN,  SEM), \
  E(BASE##_VBUFFER_IDXEN,  SEM), \
  E(BASE##_VBUFFER_BOTHEN, SEM)

// SMEM scalar loads: three operand-source forms (IMM / SGPR / SGPR_IMM).
#define SMEM3(BASE, SEM) \
  E(BASE##_IMM,      SEM), \
  E(BASE##_SGPR,     SEM), \
  E(BASE##_SGPR_IMM, SEM)

static const Entry kCanonTable[] = {
    // ---------------------------------------------------------------------
    // SOPP
    // ---------------------------------------------------------------------
    E(S_ENDPGM, S_ENDPGM),
    E(S_NOP, S_NOP),
    E(S_BRANCH, S_BRANCH),
    E(S_CODE_END, S_CODE_END),
    E(S_CBRANCH_SCC0, S_CBRANCH_SCC0),
    E(S_CBRANCH_SCC1, S_CBRANCH_SCC1),
    E(S_CBRANCH_VCCZ, S_CBRANCH_VCCZ),
    E(S_CBRANCH_VCCNZ, S_CBRANCH_VCCNZ),
    E(S_CBRANCH_EXECZ, S_CBRANCH_EXECZ),
    E(S_CBRANCH_EXECNZ, S_CBRANCH_EXECNZ),
    E(S_WAITCNT, S_WAITCNT),
    E(S_WAIT_LOADCNT, S_WAIT_LOADCNT),
    E(S_WAIT_STORECNT, S_WAIT_STORECNT),
    E(S_WAIT_KMCNT, S_WAIT_KMCNT),
    E(S_WAIT_DSCNT, S_WAIT_DSCNT),
    E(S_WAIT_XCNT, S_WAIT_XCNT),
    // gfx1250 async-memory wait counters. Canonicalised here (not
    // left as `CanonicalOp::Unknown`) so that:
    //   * the raiser's generic `Unsupported instruction` diagnostic
    //     doesn't fire on these wait sites when the companion
    //     async DMA / TENSOR op has been emulated or lifted; and
    //   * `handle-sopp.cpp`'s generic SOPP no-op arm explicitly
    //     covers them.
    // See the `S_WAIT_ASYNCCNT` / `S_WAIT_TENSORCNT` CanonicalOp doc
    // block in `canonical-op.h` for the cross-target correctness
    // argument (IR dataflow carries happens-before after the
    // synchronous `load`+`store` emulation completes).
    E(S_WAIT_ASYNCCNT, S_WAIT_ASYNCCNT),
    E(S_WAIT_TENSORCNT, S_WAIT_TENSORCNT),
    E(S_WAIT_LOADCNT_DSCNT, S_WAIT_LOADCNT_DSCNT),
    E(S_WAITCNT_DEPCTR, S_WAIT_ALU),
    E(S_CLAUSE, S_CLAUSE),
    E(S_DELAY_ALU, S_DELAY_ALU),
    E(S_SET_GPR_IDX_ON, S_SET_GPR_IDX_ON),
    E(S_SET_GPR_IDX_OFF, S_SET_GPR_IDX_OFF),
    E(S_SETVSKIP, S_SETVSKIP),

    // Barriers. GFX < 12 has a single SOPP `S_BARRIER` pseudo; GFX12+ splits
    // it into a SOPP wait and a SOP1 signal (both IMM and M0 forms). All
    // `_ISFIRST`/`_INIT`/`_JOIN`/`_LEAVE` variants are intentionally left
    // unmapped -- the raiser does not model them yet.
    E(S_BARRIER, S_BARRIER),
    E(S_BARRIER_WAIT, S_BARRIER_WAIT),
    E(S_BARRIER_SIGNAL_IMM, S_BARRIER_SIGNAL),
    E(S_BARRIER_SIGNAL_M0, S_BARRIER_SIGNAL),

    // s_sendmsg / s_sendmsghalt.
    E(S_SENDMSG, S_SENDMSG),
    E(S_SENDMSGHALT, S_SENDMSGHALT),

    // ---------------------------------------------------------------------
    // SMEM scalar loads
    // ---------------------------------------------------------------------
    SMEM3(S_LOAD_DWORD,    S_LOAD_B32),
    SMEM3(S_LOAD_DWORDX2,  S_LOAD_B64),
    SMEM3(S_LOAD_DWORDX3,  S_LOAD_B96),
    SMEM3(S_LOAD_DWORDX4,  S_LOAD_B128),
    SMEM3(S_LOAD_DWORDX8,  S_LOAD_B256),
    SMEM3(S_LOAD_DWORDX16, S_LOAD_B512),
    // gfx12+ scalar narrow loads. All four types expose the full
    // IMM/SGPR/SGPR_IMM triad so the existing SMEM3 macro applies
    // unchanged. Handler in handle-smem.cpp.
    SMEM3(S_LOAD_U8,       S_LOAD_U8),
    SMEM3(S_LOAD_I8,       S_LOAD_I8),
    SMEM3(S_LOAD_U16,      S_LOAD_U16),
    SMEM3(S_LOAD_I16,      S_LOAD_I16),
    SMEM3(S_STORE_DWORD,   S_STORE_B32),
    SMEM3(S_STORE_DWORDX2, S_STORE_B64),
    SMEM3(S_STORE_DWORDX4, S_STORE_B128),

    // ---------------------------------------------------------------------
    // SOPC
    // ---------------------------------------------------------------------
    E(S_CMP_EQ_U32, S_CMP_EQ_U32), E(S_CMP_LG_U32, S_CMP_LG_U32),
    E(S_CMP_GT_U32, S_CMP_GT_U32), E(S_CMP_GE_U32, S_CMP_GE_U32),
    E(S_CMP_LT_U32, S_CMP_LT_U32), E(S_CMP_LE_U32, S_CMP_LE_U32),
    E(S_CMP_EQ_U64, S_CMP_EQ_U64), E(S_CMP_LG_U64, S_CMP_LG_U64),
    E(S_CMP_EQ_I32, S_CMP_EQ_I32), E(S_CMP_LG_I32, S_CMP_LG_I32),
    E(S_CMP_GT_I32, S_CMP_GT_I32), E(S_CMP_GE_I32, S_CMP_GE_I32),
    E(S_CMP_LT_I32, S_CMP_LT_I32), E(S_CMP_LE_I32, S_CMP_LE_I32),
    E(S_CMP_EQ_F32, S_CMP_EQ_F32), E(S_CMP_LG_F32, S_CMP_LG_F32),
    E(S_CMP_GT_F32, S_CMP_GT_F32), E(S_CMP_GE_F32, S_CMP_GE_F32),
    E(S_CMP_LT_F32, S_CMP_LT_F32), E(S_CMP_LE_F32, S_CMP_LE_F32),
    E(S_CMP_NEQ_F32, S_CMP_NEQ_F32),
    E(S_CMP_NGT_F32, S_CMP_NGT_F32), E(S_CMP_NGE_F32, S_CMP_NGE_F32),
    E(S_CMP_NLT_F32, S_CMP_NLT_F32), E(S_CMP_NLE_F32, S_CMP_NLE_F32),
    E(S_CMP_NLG_F32, S_CMP_NLG_F32),
    E(S_CMP_O_F32, S_CMP_O_F32), E(S_CMP_U_F32, S_CMP_U_F32),
    E(S_CMP_EQ_F16, S_CMP_EQ_F16), E(S_CMP_LG_F16, S_CMP_LG_F16),
    E(S_CMP_GT_F16, S_CMP_GT_F16), E(S_CMP_GE_F16, S_CMP_GE_F16),
    E(S_CMP_LT_F16, S_CMP_LT_F16), E(S_CMP_LE_F16, S_CMP_LE_F16),
    E(S_CMP_NEQ_F16, S_CMP_NEQ_F16),
    E(S_CMP_NGT_F16, S_CMP_NGT_F16), E(S_CMP_NGE_F16, S_CMP_NGE_F16),
    E(S_CMP_NLT_F16, S_CMP_NLT_F16), E(S_CMP_NLE_F16, S_CMP_NLE_F16),
    E(S_CMP_NLG_F16, S_CMP_NLG_F16),
    E(S_CMP_O_F16, S_CMP_O_F16), E(S_CMP_U_F16, S_CMP_U_F16),

    // ---------------------------------------------------------------------
    // SOPK
    // ---------------------------------------------------------------------
    E(S_MOVK_I32, S_MOVK_I32),
    E(S_ADDK_I32, S_ADDK_I32),
    E(S_MULK_I32, S_MULK_I32),
    E(S_CMPK_GE_I32, S_CMPK_GE_I32), E(S_CMPK_GT_I32, S_CMPK_GT_I32),
    E(S_CMPK_LE_I32, S_CMPK_LE_I32), E(S_CMPK_LT_I32, S_CMPK_LT_I32),
    E(S_CMPK_GE_U32, S_CMPK_GE_U32), E(S_CMPK_GT_U32, S_CMPK_GT_U32),
    E(S_CMPK_LE_U32, S_CMPK_LE_U32), E(S_CMPK_LT_U32, S_CMPK_LT_U32),
    E(S_CMPK_EQ_I32, S_CMPK_EQ_I32), E(S_CMPK_EQ_U32, S_CMPK_EQ_U32),
    E(S_CMPK_LG_I32, S_CMPK_LG_I32), E(S_CMPK_LG_U32, S_CMPK_LG_U32),
    E(S_GETREG_B32, S_GETREG_B32),
    E(S_SETREG_B32, S_SETREG_B32),
    E(S_SETREG_IMM32_B32, S_SETREG_IMM32_B32),

    // ---------------------------------------------------------------------
    // SOP1
    // ---------------------------------------------------------------------
    E(S_MOV_B32, S_MOV_B32), E(S_MOV_B64, S_MOV_B64),
    E(S_NOT_B32, S_NOT_B32), E(S_NOT_B64, S_NOT_B64),
    E(S_BREV_B32, S_BREV_B32),
    E(S_FF0_I32_B32, S_FF0_I32_B32), E(S_FF0_I32_B64, S_FF0_I32_B64),
    E(S_FF1_I32_B32, S_FF1_I32_B32), E(S_FF1_I32_B64, S_FF1_I32_B64),
    E(S_BCNT1_I32_B32, S_BCNT1_I32_B32),
    E(S_BCNT1_I32_B64, S_BCNT1_I32_B64),
    E(S_FLBIT_I32_B32, S_FLBIT_I32_B32), E(S_FLBIT_I32_B64, S_FLBIT_I32_B64),
    E(S_FLBIT_I32, S_FLBIT_I32), E(S_FLBIT_I32_I64, S_FLBIT_I32_I64),
    E(S_SEXT_I32_I8, S_SEXT_I32_I8), E(S_SEXT_I32_I16, S_SEXT_I32_I16),
    E(S_CVT_F16_F32, S_CVT_F16_F32),
    E(S_CVT_F32_F16, S_CVT_F32_F16),
    E(S_CVT_HI_F32_F16, S_CVT_HI_F32_F16),
    E(S_CVT_F32_U32, S_CVT_F32_U32), E(S_CVT_F32_I32, S_CVT_F32_I32),
    E(S_CVT_U32_F32, S_CVT_U32_F32), E(S_CVT_I32_F32, S_CVT_I32_F32),
    // Scalar F32-to-F32 integral rounding. LLVM TableGen declares these
    // SOP1_F32_Inst forms with fceil/ffloor/ftrunc/froundeven, and the AMD
    // ISA manual records 32-bit F32 source and destination operands.
    E(S_CEIL_F32, S_CEIL_F32), E(S_FLOOR_F32, S_FLOOR_F32),
    E(S_TRUNC_F32, S_TRUNC_F32), E(S_RNDNE_F32, S_RNDNE_F32),
    E(S_AND_SAVEEXEC_B32, S_AND_SAVEEXEC_B32),
    E(S_OR_SAVEEXEC_B32, S_OR_SAVEEXEC_B32),
    E(S_XOR_SAVEEXEC_B32, S_XOR_SAVEEXEC_B32),
    E(S_ANDN2_SAVEEXEC_B32, S_ANDN2_SAVEEXEC_B32),
    E(S_ORN2_SAVEEXEC_B32, S_ORN2_SAVEEXEC_B32),
    // GFX9 B64 aliases: handler uses EXEC-width-aware ops, same semantics.
    E(S_AND_SAVEEXEC_B64, S_AND_SAVEEXEC_B32),
    E(S_OR_SAVEEXEC_B64, S_OR_SAVEEXEC_B32),
    E(S_XOR_SAVEEXEC_B64, S_XOR_SAVEEXEC_B32),
    E(S_ANDN2_SAVEEXEC_B64, S_ANDN2_SAVEEXEC_B32),
    E(S_ORN2_SAVEEXEC_B64, S_ORN2_SAVEEXEC_B32),
    E(S_GETPC_B64, S_GETPC_B64),
    // gfx1250 asm renames `S_SETPC_B64` to `s_set_pc_i64`
    // (SOPInstructions.td:2208 declares the rename via
    // `SOP1_Real_gfx1250<0x048, "s_set_pc_i64">`); the LLVM MC opcode
    // remains `S_SETPC_B64`. We canonicalise both names onto
    // CanonicalOp::S_SET_PC_I64 so the handler dispatches uniformly across
    // gfx versions; the new name is the CanonicalOp because that is how the
    // corpus surfaces it (gfx1250 disasm) and what the CanonicalOp enum
    // comment in canonical-op.h documents.
    E(S_SETPC_B64, S_SET_PC_I64),
    // gfx1250 asm renames `S_SWAPPC_B64` to `s_swap_pc_i64`
    // (SOPInstructions.td:336 declares the SOP1_64 with isCall=1;
    // line 2311 renames the asm string via
    // `SOP1_Real_gfx1250<0x049, "s_swap_pc_i64">`); the LLVM MC
    // opcode remains `S_SWAPPC_B64`. We canonicalise both names
    // onto CanonicalOp::S_SWAP_PC_I64 so the handler dispatches uniformly
    // across gfx versions; the new name is the CanonicalOp because that
    // is how the corpus surfaces it (gfx1250 disasm) and what the
    // CanonicalOp enum comment in canonical-op.h documents.
    E(S_SWAPPC_B64, S_SWAP_PC_I64),
    E(S_ADD_PC_I64, S_ADD_PC_I64),
    E(S_ABS_I32, S_ABS_I32),
    E(S_SET_VGPR_MSB, S_SET_VGPR_MSB),
    E(S_BITSET0_B32, S_BITSET0_B32),
    E(S_BITSET1_B32, S_BITSET1_B32),
    E(S_BITSET0_B64, S_BITSET0_B64),
    E(S_BITSET1_B64, S_BITSET1_B64),
    // SOPC bit-test family (SOPInstructions.td:1411-1414).  Each LLVM
    // MC opcode is a pseudo that every real-subtarget form
    // (S_BITCMP0_B32_gfx{10,11,12,13}, etc.) is canonicalised onto
    // during the MC->pseudo step in `OpcodeMap::canonicalize`, so a
    // single entry per variant covers every AMDGPU generation back to
    // gfx6.  Writes SCC only -- no scalar destination register -- which
    // matches the SOPC compare shape already implemented alongside in
    // handle-sopc.cpp.
    E(S_BITCMP0_B32, S_BITCMP0_B32),
    E(S_BITCMP1_B32, S_BITCMP1_B32),
    E(S_BITCMP0_B64, S_BITCMP0_B64),
    E(S_BITCMP1_B64, S_BITCMP1_B64),
    // S_CMOV uses [SCC] but does not write SCC, and has no _e64 form
    // (SOP1 has only one encoding). The B64 alias is included even
    // though the corpus first surfaced only the B32 form, because
    // the B64 form ships in the same gfx1250 SOP1 table at adjacent
    // opcode (.td: 0x002 / 0x003) and the handler is symmetric.
    E(S_CMOV_B32, S_CMOV_B32),
    E(S_CMOV_B64, S_CMOV_B64),

    // ---------------------------------------------------------------------
    // SOP2
    // ---------------------------------------------------------------------
    E(S_ADD_U32, S_ADD_U32), E(S_ADD_I32, S_ADD_U32),
    E(S_ADDC_U32, S_ADDC_U32),
    E(S_SUB_U32, S_SUB_U32), E(S_SUB_I32, S_SUB_U32),
    E(S_SUBB_U32, S_SUBB_U32),
    // S_ADD_U64 is intentionally mapped below in the gfx12-rename block
    // alongside S_SUB_U64 -- both surface as `CanonicalOp::S_{ADD,SUB}_NC_U64`
    // to mirror the gfx12 `s_{add,sub}_nc_u64` assembler mnemonics used
    // downstream in `handle-sop2.cpp` and the lit fixtures.  A naive
    // `E(S_ADD_U64, S_ADD_U64)` row used to live here and lost the
    // routing race silently to `canonToSem.try_emplace` (which keeps
    // the FIRST insertion -- opcode-map.cpp:1506), which meant every
    // `s_add_u64` lift actually decoded as `CanonicalOp::S_ADD_U64` rather
    // than the intended `S_ADD_NC_U64`.  The handler's defensive
    // `sop == S_ADD_U64 || sop == S_ADD_NC_U64` disjunct masked the
    // miscompile into a "same behavior either way" no-op -- which is
    // the exact pattern a diligent reviewer catches as dead code.
    // Cleaned up in the same commit that adds this comment.
    E(S_AND_B32, S_AND_B32), E(S_AND_B64, S_AND_B64),
    E(S_OR_B32, S_OR_B32), E(S_OR_B64, S_OR_B64),
    E(S_XOR_B32, S_XOR_B32), E(S_XOR_B64, S_XOR_B64),
    E(S_ANDN2_B32, S_ANDN2_B32), E(S_ANDN2_B64, S_ANDN2_B64),
    E(S_ORN2_B32, S_ORN2_B32), E(S_ORN2_B64, S_ORN2_B64),
    E(S_NAND_B32, S_NAND_B32), E(S_NAND_B64, S_NAND_B64),
    E(S_NOR_B32, S_NOR_B32), E(S_NOR_B64, S_NOR_B64),
    E(S_XNOR_B32, S_XNOR_B32), E(S_XNOR_B64, S_XNOR_B64),
    E(S_ABSDIFF_I32, S_ABSDIFF_I32),
    E(S_LSHL_B32, S_LSHL_B32), E(S_LSHL_B64, S_LSHL_B64),
    E(S_LSHR_B32, S_LSHR_B32), E(S_LSHR_B64, S_LSHR_B64),
    E(S_ASHR_I32, S_ASHR_I32), E(S_ASHR_I64, S_ASHR_I64),
    E(S_MUL_I32, S_MUL_I32), E(S_MUL_HI_U32, S_MUL_HI_U32),
    E(S_MUL_HI_I32, S_MUL_HI_I32),
    E(S_MUL_U64, S_MUL_U64), E(S_MUL_F32, S_MUL_F32),
    E(S_ADD_F32, S_ADD_F32), E(S_SUB_F32, S_SUB_F32),
    E(S_FMAAK_F32, S_FMAAK_F32),
    E(S_FMAMK_F32, S_FMAMK_F32),
    E(S_FMAC_F32, S_FMAC_F32),
    // LLVM's canonical pseudo is `S_{MIN,MAX}_F32` on both gfx11
    // (`s_{min,max}_f32`) and gfx12+ (`s_{min,max}_num_f32`, via the
    // subtarget-specific real rename). Route it through NUM CanonicalOps so the
    // handler cannot be confused with the IEEE-2019 NaN-propagating
    // `S_{MINIMUM,MAXIMUM}_F32` family.
    E(S_MAX_F32, S_MAX_NUM_F32), E(S_MIN_F32, S_MIN_NUM_F32),
    // gfx12+ scalar IEEE-2019 NaN-propagating f16/f32 extrema, distinct from
    // the NUM family above; lower to `llvm.{maximum,minimum}.{f16,f32}`.
    E(S_MAXIMUM_F16, S_MAXIMUM_F16), E(S_MINIMUM_F16, S_MINIMUM_F16),
    E(S_MAXIMUM_F32, S_MAXIMUM_F32), E(S_MINIMUM_F32, S_MINIMUM_F32),
    E(S_BFE_U32, S_BFE_U32), E(S_BFE_I32, S_BFE_I32),
    E(S_BFE_I64, S_BFE_I64),
    E(S_BFM_B32, S_BFM_B32), E(S_BFM_B64, S_BFM_B64),
    E(S_CSELECT_B32, S_CSELECT_B32), E(S_CSELECT_B64, S_CSELECT_B64),
    E(S_MIN_I32, S_MIN_I32), E(S_MIN_U32, S_MIN_U32),
    E(S_MAX_I32, S_MAX_I32), E(S_MAX_U32, S_MAX_U32),
    E(S_PACK_LL_B32_B16, S_PACK_LL_B32_B16),
    E(S_PACK_LH_B32_B16, S_PACK_LH_B32_B16),
    E(S_LSHL1_ADD_U32, S_LSHL1_ADD_U32), E(S_LSHL2_ADD_U32, S_LSHL2_ADD_U32),
    E(S_LSHL3_ADD_U32, S_LSHL3_ADD_U32), E(S_LSHL4_ADD_U32, S_LSHL4_ADD_U32),
    // gfx12 `s_add_nc_u64` (renamed from `s_add_u64` in the
    // assembler -- see SOPInstructions.td 2300-ish range, same
    // pattern as `s_sub_u64 ... "s_sub_nc_u64"` below).  LLVM's
    // pseudo is still `S_ADD_U64`; surface it as
    // `CanonicalOp::S_ADD_NC_U64` so the handler dispatch, the lit
    // fixtures (`lit_tests/s_sub_nc_u64/` and any future
    // `s_add_nc_u64` sibling), and the runtime diagnostics all
    // speak the same gfx12+ mnemonic.  This row is the ONLY
    // `S_ADD_U64` entry in `kCanonTable` -- an earlier version
    // of this file had a stray `E(S_ADD_U64, S_ADD_U64)` row in
    // the SOP2 family block above that won the
    // `canonToSem.try_emplace` race and quietly routed every
    // `s_add_u64` lift to the wrong CanonicalOp.  See the comment in
    // that block for the audit trail.
    E(S_ADD_U64, S_ADD_NC_U64),
    // gfx12 `s_sub_nc_u64` (renamed from `s_sub_u64` in the
    // assembler -- see SOPInstructions.td 2311
    // `S_SUB_U64 ... "s_sub_nc_u64"`). LLVM's pseudo is still
    // `S_SUB_U64`; surface it as `CanonicalOp::S_SUB_NC_U64` to mirror
    // the S_ADD_NC_U64 convention.
    E(S_SUB_U64, S_SUB_NC_U64),

    // ---------------------------------------------------------------------
    // VOP1 (canonical form is `_e64` after getVOPe64 collapses e32)
    // ---------------------------------------------------------------------
    E(V_MOV_B32_e64, V_MOV_B32),
    E(V_MOV_B64_e64, V_MOV_B64),
    E(V_MOV_B64_PSEUDO, V_MOV_B64),
    // V_MOV_B16 is gfx11+ true16 / fake16. The pseudo-alias pass strips the
    // `_t16_` and `_fake16_` infixes before lookup so only the bare e64 form
    // needs an entry here.
    E(V_MOV_B16_e64, V_MOV_B16),
    E(V_SWAP_B32, V_SWAP_B32),
    E(V_NOP_e64, V_NOP),
    E(V_NOT_B32_e64, V_NOT_B32),
    E(V_BFREV_B32_e64, V_BFREV_B32),
    E(V_FFBH_U32_e64, V_FFBH_U32),
    E(V_FFBL_B32_e64, V_FFBL_B32),
    E(V_FFBH_I32_e64, V_FFBH_I32),
    E(V_PRNG_B32_e64, V_PRNG_B32),
    E(V_CVT_F32_I32_e64, V_CVT_F32_I32),
    E(V_CVT_F32_U32_e64, V_CVT_F32_U32),
    E(V_CVT_I32_F32_e64, V_CVT_I32_F32),
    E(V_CVT_U32_F32_e64, V_CVT_U32_F32),
    // gfx11+ true16/fake16 v_cvt_u32_u16. Real subtarget opcodes
    // (`_t16_e64`, `_fake16_e64`, `_t16_e32`, `_fake16_e32`, plus their
    // _dpp/_dpp8/_sdwa variants) canonicalize onto these base pseudos via
    // the suffix-stripping pass in `OpcodeMap::canonicalize`.
    E(V_CVT_U32_U16_e32, V_CVT_U32_U16),
    E(V_CVT_U32_U16_e64, V_CVT_U32_U16),
    E(V_CVT_F16_F32_e64, V_CVT_F16_F32),
    E(V_CVT_F32_F16_e64, V_CVT_F32_F16),
    E(V_CVT_F32_BF16_e64, V_CVT_F32_BF16),
    E(V_CVT_PK_F32_FP8_e64, V_CVT_PK_F32_FP8),
    E(V_CVT_PK_F32_BF8_e64, V_CVT_PK_F32_BF8),
    E(V_CVT_F32_FP8_e64, V_CVT_F32_FP8),
    E(V_CVT_F32_BF8_e64, V_CVT_F32_BF8),
    // gfx1250-only packed-8 FP4 -> BF16 scaled convert
    // (VOP3Instructions.td:1788). The real form
    // `V_CVT_SCALE_PK8_BF16_FP4_e64_gfx1250` canonicalises onto the
    // `_e64` pseudo via the MC->pseudo step in
    // `OpcodeMap::canonicalize` (getMCOpcode-derived table); a single
    // entry therefore covers every real subtarget slot.
    E(V_CVT_SCALE_PK8_BF16_FP4_e64, V_CVT_SCALE_PK8_BF16_FP4),
    E(V_CVT_F32_UBYTE0_e64, V_CVT_F32_UBYTE0),
    E(V_CVT_F32_UBYTE1_e64, V_CVT_F32_UBYTE1),
    E(V_CVT_F32_UBYTE2_e64, V_CVT_F32_UBYTE2),
    E(V_CVT_F32_UBYTE3_e64, V_CVT_F32_UBYTE3),
    // F32<->F64 converts are registered on `_e64` only.  On gfx11+/gfx12 the
    // production encodings are VOP1/DPP `_e64` forms; `_e32` slots exist on
    // older ISAs (see VOP1Instructions.td) but are not emitted for the gfx1250
    // corpus this table targets.  An unmapped `_e32` MC opcode would surface
    // as UnsupportedOpcode rather than silently using the wrong profile.
    E(V_CVT_F32_F64_e64, V_CVT_F32_F64),
    E(V_CVT_F64_F32_e64, V_CVT_F64_F32),
    E(V_CVT_F64_U32_e64, V_CVT_F64_U32),
    E(V_CVT_F64_I32_e64, V_CVT_F64_I32),
    E(V_CVT_U32_F64_e64, V_CVT_U32_F64),
    E(V_CVT_I32_F64_e64, V_CVT_I32_F64),
    E(V_RCP_IFLAG_F32_e64, V_RCP_IFLAG_F32),
    E(V_RCP_F32_e64, V_RCP_F32),
    E(V_RSQ_F32_e64, V_RSQ_F32),
    E(V_SQRT_F32_e64, V_SQRT_F32),
    E(V_EXP_F32_e64, V_EXP_F32),
    E(V_LOG_F32_e64, V_LOG_F32),
    E(V_FREXP_EXP_I32_F64_e64, V_FREXP_EXP_I32_F64),
    // Targets with native tanh support use `llvm.amdgcn.tanh.*`; targets
    // without native support lower through matching OCML entry points.
    E(V_TANH_F32_e64, V_TANH_F32),
    E(V_S_EXP_F32_e64, V_S_EXP_F32),
    E(V_S_LOG_F32_e64, V_S_LOG_F32),
    E(V_S_RCP_F32_e64, V_S_RCP_F32),
    E(V_S_RSQ_F32_e64, V_S_RSQ_F32),
    E(V_S_SQRT_F32_e64, V_S_SQRT_F32),
    E(V_FLOOR_F32_e64, V_FLOOR_F32),
    E(V_CEIL_F32_e64, V_CEIL_F32),
    E(V_CEIL_F64_e64, V_CEIL_F64),
    E(V_TRUNC_F32_e64, V_TRUNC_F32),
    E(V_RNDNE_F32_e64, V_RNDNE_F32),
    E(V_FRACT_F32_e64, V_FRACT_F32),
    E(V_FREXP_MANT_F32_e64, V_FREXP_MANT_F32),
    E(V_READFIRSTLANE_B32, V_READFIRSTLANE_B32),
    E(V_FLOOR_F16_e64, V_FLOOR_F16),
    E(V_CVT_F16_U16_e64, V_CVT_F16_U16),
    E(V_CVT_U16_F16_e64, V_CVT_U16_F16),

    // ---------------------------------------------------------------------
    // VOP2 / VOP3
    // ---------------------------------------------------------------------
    E(V_ADD_F32_e64, V_ADD_F32),
    E(V_SUB_F32_e64, V_SUB_F32),
    E(V_SUBREV_F32_e64, V_SUBREV_F32),
    E(V_MUL_F32_e64, V_MUL_F32),
    E(V_FMAC_F32_e64, V_FMAC_F32),
    E(V_FMA_F32_e64, V_FMA_F32),
    E(V_FMAMK_F32, V_FMAMK_F32),
    E(V_FMAAK_F32, V_FMAAK_F32),
    E(V_MAX_F32_e64, V_MAX_NUM_F32),
    E(V_MIN_F32_e64, V_MIN_NUM_F32),
    // gfx11+ IEEE-754 2019 maximum/minimum. These differ from V_MAX_NUM_F32 /
    // V_MIN_NUM_F32 in their NaN semantics: maximum/minimum propagate NaN,
    // while maxnum/minnum return the non-NaN operand.
    E(V_MAXIMUM_F32_e64, V_MAXIMUM_F32),
    E(V_MINIMUM_F32_e64, V_MINIMUM_F32),
    // IEEE-754 2019 ternary NaN-propagating max/min (gfx11+/gfx12 only,
    // gated by `HasMinimum3Maximum3F32` in AMDGPU.td:194).  Lowered as a
    // chain of two 2-source `llvm.maximum`/`llvm.minimum` calls in
    // handle-valu.cpp; llc on gfx950 selects v_max3/v_min3 when NaN
    // propagation isn't material, or two 2-source ops otherwise.
    E(V_MAXIMUM3_F32_e64, V_MAXIMUM3_F32),
    E(V_MINIMUM3_F32_e64, V_MINIMUM3_F32),
    // gfx11+/gfx12 IEEE-754 2019 NaN-propagating clamp pair. These are the
    // non-.NUM forms at VOP3 opcodes 0x26c / 0x26d, distinct from
    // V_MINMAX_NUM_F32's maxnum(minnum(...), ...) semantics.
    E(V_MAXIMUMMINIMUM_F32_e64, V_MAXIMUMMINIMUM_F32),
    E(V_MINIMUMMAXIMUM_F32_e64, V_MINIMUMMAXIMUM_F32),
    E(V_DIV_FIXUP_F32_e64, V_DIV_FIXUP_F32),
    E(V_DIV_FMAS_F32_e64, V_DIV_FMAS_F32),
    E(V_DIV_SCALE_F32_e64, V_DIV_SCALE_F32),
    // LLVM's no-carry 32-bit add/sub pseudos are just `V_ADD_U32` etc.; the
    // carry-in-carry-out form is the older `V_ADDC_U32`/`V_SUBB_U32` family.
    E(V_ADD_U32_e64, V_ADD_NC_U32),
    E(V_SUB_U32_e64, V_SUB_NC_U32),
    E(V_SUBREV_U32_e64, V_SUBREV_NC_U32),
    // GFX9 VOP3-only signed add/sub (saddsat/ssubsat when clamp is set,
    // plain add/sub otherwise). Distinct from the legacy GFX10+ NC variants.
    E(V_ADD_I32_e64, V_ADD_I32),
    E(V_SUB_I32_e64, V_SUB_I32),
    E(V_ADD_CO_U32_e64, V_ADD_CO_U32),
    E(V_ADDC_U32_e64, V_ADD_CO_CI_U32),
    E(V_SUB_CO_U32_e64, V_SUB_CO_U32),
    E(V_SUBREV_CO_U32_e64, V_SUBREV_CO_U32),
    E(V_SUBB_U32_e64, V_SUB_CO_CI_U32),
    E(V_SUBBREV_U32_e64, V_SUBREV_CO_CI_U32),
    E(V_AND_B32_e64, V_AND_B32),
    E(V_OR_B32_e64, V_OR_B32),
    E(V_XOR_B32_e64, V_XOR_B32),
    E(V_XNOR_B32_e64, V_XNOR_B32),
    E(V_LSHLREV_B32_e64, V_LSHLREV_B32),
    E(V_LSHRREV_B32_e64, V_LSHRREV_B32),
    E(V_ASHRREV_I32_e64, V_ASHRREV_I32),
    E(V_CNDMASK_B32_e64, V_CNDMASK_B32),
    E(V_MUL_LO_U32_e64, V_MUL_LO_U32),
    E(V_MUL_HI_U32_e64, V_MUL_HI_U32),
    E(V_MUL_HI_I32_e64, V_MUL_HI_I32),
    E(V_MUL_I32_I24_e64, V_MUL_I32_I24),
    E(V_MUL_U32_U24_e64, V_MUL_U32_U24),
    E(V_MUL_HI_U32_U24_e64, V_MUL_HI_U32_U24),
    E(V_MUL_HI_I32_I24_e64, V_MUL_HI_I32_I24),
    E(V_MAD_I32_I24_e64, V_MAD_I32_I24),
    E(V_MAD_I32_I24_e64_gfx11, V_MAD_I32_I24),
    E(V_MAD_I32_I24_e64_gfx12, V_MAD_I32_I24),
    E(V_MAD_U32_U24_e64, V_MAD_U32_U24),
    E(V_MAD_U32_e64, V_MAD_U32),
    E(V_ADD3_U32_e64, V_ADD3_U32),
    E(V_LSHL_ADD_U32_e64, V_LSHL_ADD_U32),
    E(V_ADD_LSHL_U32_e64, V_ADD_LSHL_U32),
    E(V_LSHL_OR_B32_e64, V_LSHL_OR_B32),
    E(V_AND_OR_B32_e64, V_AND_OR_B32),
    E(V_OR3_B32_e64, V_OR3_B32),
    // gfx9+ ternary xor+add (VOP3Instructions.td:684, iselect
    // pattern at :831 -- `add(xor(a, b), c)`).
    E(V_XAD_U32_e64, V_XAD_U32),
    // gfx6+ funnel-shift right (VOP3Instructions.td:218,
    // gfx11/gfx12 opcode 0x216 at :1787). The `_t16` and
    // `_fake16` variants collapse onto V_ALIGNBIT_B32_e64 via
    // sameSemanticShape.
    E(V_ALIGNBIT_B32_e64, V_ALIGNBIT_B32),
    // gfx10+ ternary xor (VOP3Instructions.td:1348). VOP3
    // ternaries only have the e64 form.
    E(V_XOR3_B32_e64, V_XOR3_B32),
    // VOP3 true16 16-bit no-carry add/sub. Unsigned reals keep the
    // V_{ADD,SUB}_NC_U16 pseudo names; signed forms use LLVM's older
    // V_{ADD,SUB}_I16 pseudos even when gfx10+ real mnemonics spell them
    // v_{add,sub}_nc_i16. Older non-_nc signed real spellings share the same
    // no-carry semantic shape through those pseudos. sameSemanticShape
    // collapses true16/fake16 variants onto the base pseudo entries below.
    E(V_ADD_NC_U16_e64, V_ADD_NC_U16),
    E(V_SUB_NC_U16_e64, V_SUB_NC_U16),
    E(V_ADD_I16_e64, V_ADD_NC_I16),
    E(V_SUB_I16_e64, V_SUB_NC_I16),
    // gfx1250 add-then-min/max VOP3. The real subtarget opcodes canonicalize
    // through this pseudo via AMDGPU::getMCOpcode-derived tables.
    E(V_ADD_MIN_U32_e64, V_ADD_MIN_U32),
    E(V_ADD_MAX_U32_e64, V_ADD_MAX_U32),
    E(V_ADD_MIN_I32_e64, V_ADD_MIN_I32),
    E(V_ADD_MAX_I32_e64, V_ADD_MAX_I32),
    E(V_BFE_U32_e64, V_BFE_U32),
    E(V_BFE_I32_e64, V_BFE_I32),
    // gfx6+ VOP3 bit-field insert. Ternary, e64-only (no VOP1/VOP2
    // form). Hardware semantic: `dst = (src0 & src1) | (~src0 & src2)`
    // -- src0 is the mask, src1 provides bits where mask=1, src2
    // provides bits where mask=0. See VOP3Instructions.td
    // `AMDGPUbfiPattern`. Same canonical opcode across gfx11/gfx12;
    // libdevice's asin lowering emits this as its sign-stitching
    // step, which was the real-world coverage gap that motivated
    // the handler.
    E(V_BFI_B32_e64, V_BFI_B32),
    E(V_PERM_B32_e64, V_PERM_B32),
    E(V_MBCNT_LO_U32_B32_e64, V_MBCNT_LO_U32_B32),
    E(V_MBCNT_HI_U32_B32_e64, V_MBCNT_HI_U32_B32),
    E(V_READLANE_B32, V_READLANE_B32),
    E(V_WRITELANE_B32, V_WRITELANE_B32),
    E(V_MED3_F32_e64, V_MED3_NUM_F32),
    E(V_MAX3_F32_e64, V_MAX3_NUM_F32),
    E(V_MIN3_F32_e64, V_MIN3_NUM_F32),
    // gfx11 V_MINMAX_F32 (opcode 0x25f) and the gfx12-renamed
    // V_MINMAX_NUM_F32 (opcode 0x268) share the same minnum-of-
    // maxnum semantics; both LLVM pseudos canonicalize onto
    // V_MINMAX_F32_e64 so a single mapping suffices.
    E(V_MINMAX_F32_e64, V_MINMAX_NUM_F32),
    E(V_MAXMIN_F32_e64, V_MAXMIN_NUM_F32),
    // gfx11/gfx12 VOP3 integer 3-way min/max. DPP/DPP8 modifier
    // variants canonicalize through the same opcode-map stripping path used by
    // the other VOP3+DPP entries, so a base e64 row is the semantic key.
    E(V_MAX3_U32_e64, V_MAX3_U32),
    E(V_MIN3_U32_e64, V_MIN3_U32),
    E(V_MAX3_I32_e64, V_MAX3_I32),
    E(V_MIN3_I32_e64, V_MIN3_I32),
    E(V_MED3_I32_e64, V_MED3_I32),
    E(V_BITOP3_B32_e64, V_BITOP3_B32),
    E(V_BITOP3_B16_e64, V_BITOP3_B16),
    E(V_FMA_MIX_F32, V_FMA_MIX_F32),
    // gfx9.5+/gfx1250 BF16-narrow sibling (VOP3PInstructions.td:109).
    // Same VOP3P operand shape (3 srcs + op_sel / op_sel_hi modifiers)
    // as V_FMA_MIX_F32; only the narrow-half element type changes
    // (bf16 vs f16). The real form V_FMA_MIX_F32_BF16_gfx1250 is
    // canonicalized onto this pseudo via the MC->pseudo step in
    // `canonicalize` (getMCOpcode-derived table), mirroring the
    // V_FMA_MIX_F32 entry above.
    E(V_FMA_MIX_F32_BF16, V_FMA_MIX_F32_BF16),
    // F16-result mixed FMA siblings. Same source-selection modifiers as
    // V_FMA_MIX_F32, but the f32 FMA result is rounded to F16 and written
    // to one half of the tied output register.
    E(V_FMA_MIXLO_F16, V_FMA_MIXLO_F16),
    E(V_FMA_MIXHI_F16, V_FMA_MIXHI_F16),
    // gfx1250 BF16-result mixed FMA siblings. Same source-selection modifiers
    // as V_FMA_MIX_F32_BF16, but the f32 FMA result is rounded to BF16 and
    // written to one half of the tied output register.
    E(V_FMA_MIXLO_BF16, V_FMA_MIXLO_BF16),
    E(V_FMA_MIXHI_BF16, V_FMA_MIXHI_BF16),
    E(V_ADD_F16_e64, V_ADD_F16),
    E(V_MUL_F16_e64, V_MUL_F16),
    E(V_SUB_F16_e64, V_SUB_F16),
    E(V_SUBREV_F16_e64, V_SUBREV_F16),
    E(V_MAC_F16_e64, V_MAC_F16),
    E(V_MADMK_F16, V_MADMK_F16),
    E(V_MADAK_F16, V_MADAK_F16),
    E(V_FMAC_F16_e64, V_FMAC_F16),
    // VOP3 F16 explicit-source fused multiply-add. LLVM emits two parallel
    // pseudos because gfx9 introduced op_sel on the original gfx8 V_FMA_F16:
    //   V_FMA_F16_e64        -- gfx8 ("vi") form (no op_sel)
    //   V_FMA_F16_gfx9_e64   -- gfx9+ op_sel form (also the base for t16/
    //                            fake16 collapse below)
    // Both map to the same llvm.fma.f16 lowering; the gfx9 op_sel handling
    // lives in the handler. The `_t16_` / `_fake16_` 16-bit encoding
    // variants on top of the gfx9 form are folded onto V_FMA_F16_gfx9_e64
    // by the sameSemanticShape stripping pass in `buildPseudoAliasMap`.
    E(V_FMA_F16_e64, V_FMA_F16),
    E(V_FMA_F16_gfx9_e64, V_FMA_F16),
    E(V_MAX_F16_e64, V_MAX_NUM_F16),
    E(V_MIN_F16_e64, V_MIN_NUM_F16),
    E(V_MINMAX_F16_e64, V_MINMAX_NUM_F16),
    E(V_MAXMIN_F16_e64, V_MAXMIN_NUM_F16),
    E(V_MAXIMUM_F16_e64, V_MAXIMUM_F16),
    E(V_MINIMUM_F16_e64, V_MINIMUM_F16),
    E(V_MAXIMUM3_F16_e64, V_MAXIMUM3_F16),
    E(V_MINIMUM3_F16_e64, V_MINIMUM3_F16),
    E(V_MAXIMUMMINIMUM_F16_e64, V_MAXIMUMMINIMUM_F16),
    E(V_MINIMUMMAXIMUM_F16_e64, V_MINIMUMMAXIMUM_F16),
    E(V_LDEXP_F16_e64, V_LDEXP_F16),
    E(V_TANH_F16_e64, V_TANH_F16),
    E(V_LDEXP_F32_e64, V_LDEXP_F32),
    E(V_DOT2C_I32_I16_e64, V_DOT2C_I32_I16),
    E(V_DOT4C_I32_I8_e64, V_DOT4C_I32_I8),
    E(V_DOT8C_I32_I4_e64, V_DOT8C_I32_I4),
    E(V_DOT4_I32_IU8, V_DOT4_I32_IU8),
    E(V_DOT4_I32_IU8_gfx11, V_DOT4_I32_IU8),
    E(V_DOT4_I32_IU8_gfx12, V_DOT4_I32_IU8),
    E(V_PK_FMAC_F16_e64, V_PK_FMAC_F16),
    E(V_MAX_U16_e64, V_MAX_U16),
    E(V_MIN_U16_e64, V_MIN_U16),
    E(V_MAX_I16_e64, V_MAX_I16),
    E(V_MIN_I16_e64, V_MIN_I16),
    E(V_ADD_U16_e64, V_ADD_U16),
    E(V_SUB_U16_e64, V_SUB_U16),
    E(V_SUBREV_U16_e64, V_SUBREV_U16),
    E(V_MUL_LO_U16_e64, V_MUL_LO_U16),
    E(V_ASHRREV_I16_e64, V_ASHRREV_I16),
    E(V_LSHRREV_B16_e64, V_LSHRREV_B16),
    E(V_LSHLREV_B16_e64, V_LSHLREV_B16),
    E(V_PACK_B32_F16_e64, V_PACK_B32_F16),
    E(V_CVT_PK_BF16_F32_e64, V_CVT_PK_BF16_F32),
    E(V_CVT_PK_BF8_F32_e64, V_CVT_PK_BF8_F32),
    E(V_CVT_PK_FP8_F32_e64, V_CVT_PK_FP8_F32),
    E(V_CVT_PKRTZ_F16_F32_e64, V_CVT_PKRTZ_F16_F32),
    E(V_CVT_PK_F16_F32_e64, V_CVT_PK_F16_F32),
    E(V_CVT_SCALEF32_PK_FP4_F32_e64, V_CVT_SCALEF32_PK_FP4_F32),
    // gfx1250-only packed-8 scaled FP8 conversion (VOP3, opcode 0x2c3,
    // VOP3Instructions.td:2384).  Same-target maps to native intrinsic
    // `int_amdgcn_cvt_scalef32_pk8_fp8_f32`; cross-target paths with FP8
    // conversion support use the software emulation in handle-valu.cpp.
    E(V_CVT_SCALEF32_PK8_FP8_F32_e64, V_CVT_SCALEF32_PK8_FP8_F32),
    E(V_BFM_B32_e64, V_BFM_B32),

    // ---------------------------------------------------------------------
    // FP64
    // ---------------------------------------------------------------------
    E(V_ADD_F64_e64, V_ADD_F64),
    E(V_MUL_F64_e64, V_MUL_F64),
    E(V_FMA_F64_e64, V_FMA_F64),
    E(V_FMAC_F64_e64, V_FMAC_F64),
    // VOP2-only mandatory-literal FMA forms. The pseudos have no e32/e64
    // split (analogous to V_FMAMK_F32 / V_FMAAK_F32 above); subtarget reals
    // (V_FMAMK_F64_gfx1250 / V_FMAAK_F64_gfx1250) collapse back via
    // OpcodeMap::canonicalize.
    E(V_FMAMK_F64, V_FMAMK_F64),
    E(V_FMAAK_F64, V_FMAAK_F64),
    E(V_MAX_NUM_F64_e64, V_MAX_NUM_F64),
    E(V_MIN_NUM_F64_e64, V_MIN_NUM_F64),
    // V_RCP_F64 has no DPP (`VOP1_Real_..._NO_DPP_...`); the e32
    // form is collapsed to e64 by getVOPe64 before lookup, so a
    // single e64 entry covers both encodings.
    E(V_RCP_F64_e64, V_RCP_F64),
    E(V_LDEXP_F64_e64, V_LDEXP_F64),

    // ---------------------------------------------------------------------
    // More VOP3 32-bit int min/max and lane perms
    // ---------------------------------------------------------------------
    E(V_MAX_U32_e64, V_MAX_U32),
    E(V_MIN_U32_e64, V_MIN_U32),
    E(V_MAX_I32_e64, V_MAX_I32),
    E(V_MIN_I32_e64, V_MIN_I32),
    E(V_PERMLANE16_B32_e64, V_PERMLANE16_B32),
    E(V_PERMLANEX16_B32_e64, V_PERMLANEX16_B32),
    E(V_PERMLANE64_B32, V_PERMLANE64_B32),
    E(V_PERMLANE16_SWAP_B32_e32, V_PERMLANE16_SWAP_B32),
    E(V_PERMLANE16_SWAP_B32_e64, V_PERMLANE16_SWAP_B32),
    E(V_PERMLANE32_SWAP_B32_e32, V_PERMLANE32_SWAP_B32),
    E(V_PERMLANE32_SWAP_B32_e64, V_PERMLANE32_SWAP_B32),

    // ---------------------------------------------------------------------
    // VOPC (V_CMP_* and V_CMPX_*) are NOT enumerated here. They're picked up
    // by scanning canonical pseudo names in `OpcodeMap::build` and routed to
    // the two collapsed CanonicalOps (V_CMP / V_CMPX) with metadata (predicate,
    // element width, int/float) in the `vcmp_` side table.
    //
    // ---------------------------------------------------------------------
    // VOP3P (canonical is bare pseudo name, no _e64)
    // ---------------------------------------------------------------------
    E(V_PK_ADD_F32, V_PK_ADD_F32),
    E(V_PK_MUL_F32, V_PK_MUL_F32),
    E(V_PK_FMA_F32, V_PK_FMA_F32),
    E(V_PK_FMA_F16, V_PK_FMA_F16),
    E(V_PK_ADD_F16, V_PK_ADD_F16),
    E(V_PK_MUL_F16, V_PK_MUL_F16),
    // gfx1250-only packed BF16 arithmetic. The `_gfx1250` real opcodes
    // collapse to these bare pseudos via the MC->pseudo map, matching the
    // existing packed BF16 TableGen family.
    E(V_PK_ADD_BF16, V_PK_ADD_BF16),
    E(V_PK_MUL_BF16, V_PK_MUL_BF16),
    E(V_PK_MIN_NUM_BF16, V_PK_MIN_NUM_BF16),
    E(V_PK_MAX_NUM_BF16, V_PK_MAX_NUM_BF16),
    E(V_PK_FMA_BF16, V_PK_FMA_BF16),
    // LLVM has no `V_PK_MAX_F32`/`V_PK_MIN_F32` pseudo (only F16 variants);
    // leave the matching CanonicalOps unmapped until one appears.
    E(V_PK_MOV_B32, V_PK_MOV_B32),
    // Packed `<2 x i16>` int family. LLVM emits the bare TableGen pseudo
    // (`V_PK_ADD_U16` / `V_PK_LSHLREV_B16` / `V_PK_MUL_LO_U16`); the
    // gfx10/gfx11/gfx12/vi realtriples (`_gfx10`, `_vi`, etc.) all
    // canonicalize back to it through the pseudo-alias step. Siblings
    // V_PK_LSHRREV_B16 / V_PK_ASHRREV_I16 / V_PK_SUB_U16
    // / V_PK_MAX_{I,U}16 / V_PK_MIN_{I,U}16 share the same handler shape
    // and are intentionally NOT pre-enumerated -- the no-fallback policy
    // says we wait for a corpus producer rather than ship dead lift code.
    E(V_PK_ADD_U16, V_PK_ADD_U16),
    E(V_PK_LSHLREV_B16, V_PK_LSHLREV_B16),
    E(V_PK_MUL_LO_U16, V_PK_MUL_LO_U16),

    // ---------------------------------------------------------------------
    // 64-bit vector
    // ---------------------------------------------------------------------
    E(V_LSHLREV_B64_e64, V_LSHLREV_B64),
    // V_LSHRREV_B64 / V_ASHRREV_I64 -- gfx8+ 64-bit logical / arithmetic
    // right shift with reversed operands (`dst = src1 >> shamt`). Only
    // the plain `_e64` pseudo exists; per-ISA realtriples (gfx10 / gfx11
    // / gfx1250) all collapse back to it through the disassembler's
    // pseudo-alias step (TableGen treats the gfx1250 form as a `_e64`
    // realtriple, not a `_t16_e64` half-precision variant -- see
    // VOP3Instructions.td:2341 vs the b16 family at 2336-2340).
    E(V_LSHRREV_B64_e64, V_LSHRREV_B64),
    E(V_ASHRREV_I64_e64, V_ASHRREV_I64),
    E(V_LSHL_ADD_U64_e64, V_LSHL_ADD_U64),
    // LLVM's 64-bit no-carry add pseudo is simply `V_ADD_U64_e64`.
    E(V_ADD_U64_e64, V_ADD_NC_U64),
    // gfx1250 VOP2 64-bit no-carry subtract. LLVM's real opcode is
    // `V_SUB_U64_e64` while the printed ISA mnemonic is `v_sub_nc_u64`;
    // route by opcode, not by mnemonic text. VOP2Instructions.td marks it
    // non-commutable and SIInstructions.td gives the i64 `src0 - src1`
    // pattern, so the handler must preserve operand order exactly.
    E(V_SUB_U64_e64, V_SUB_NC_U64),
    // gfx1250 VOP3 64-bit integer min/max.  VOP3Instructions.td attaches
    // smax/umax/smin/umin patterns respectively; the handler mirrors the
    // manual's compare-and-select semantics on the full i64 lane value.
    E(V_MAX_I64_e64, V_MAX_I64),
    E(V_MAX_U64_e64, V_MAX_U64),
    E(V_MIN_I64_e64, V_MIN_I64),
    E(V_MIN_U64_e64, V_MIN_U64),
    // gfx1250 VOP2 V_MUL_U64; canonicalizer collapses `_gfx1250` suffix
    // onto the bare `V_MUL_U64_e64` pseudo.
    E(V_MUL_U64_e64, V_MUL_U64),
    E(V_MAD_U64_U32_e64, V_MAD_U64_U32),
    // LLVM no longer exposes a distinct carry-out variant; `V_MAD_CO_U64_U32`
    // CanonicalOp stays unmapped until one reappears.
    // gfx1250-only no-carry 64-bit multiply-add opcodes
    // (`VOP3Only_Realtriple_gfx1250` at VOP3Instructions.td:2129 / 2130,
    // encodings 0x2fa / 0x2fb).  Unlike the older `V_MAD_U64_U32` /
    // `V_MAD_I64_I32` family, these pseudos don't have an `_e64` suffix
    // variant with a carry/overflow sink, so one MCOpcode <-> one CanonicalOp;
    // see the `V_MAD_NC_*` block in `canonical-op.h` for the semantics
    // description and `handle-valu.cpp`'s handler arm for how the
    // canonical `add(mul(sext/zext s0, sext/zext s1), s2_i64)` lowering
    // re-enters the backend's `SelectMad64_32` pattern matcher on both
    // gfx1250 and gfx942 targets.
    E(V_MAD_NC_U64_U32_e64, V_MAD_NC_U64_U32),
    E(V_MAD_NC_I64_I32_e64, V_MAD_NC_I64_I32),

    // ---------------------------------------------------------------------
    // FLAT
    // ---------------------------------------------------------------------
    E(FLAT_LOAD_UBYTE, FLAT_LOAD_UBYTE), E(FLAT_LOAD_SBYTE, FLAT_LOAD_SBYTE),
    E(FLAT_LOAD_USHORT, FLAT_LOAD_USHORT), E(FLAT_LOAD_SSHORT, FLAT_LOAD_SSHORT),
    E(FLAT_LOAD_DWORD, FLAT_LOAD_DWORD), E(FLAT_LOAD_DWORDX2, FLAT_LOAD_DWORDX2),
    E(FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX3), E(FLAT_LOAD_DWORDX4, FLAT_LOAD_DWORDX4),
    E(FLAT_STORE_BYTE, FLAT_STORE_BYTE), E(FLAT_STORE_SHORT, FLAT_STORE_SHORT),
    E(FLAT_STORE_SHORT_D16_HI, FLAT_STORE_SHORT_D16_HI),
    E(FLAT_STORE_DWORD, FLAT_STORE_DWORD), E(FLAT_STORE_DWORDX2, FLAT_STORE_DWORDX2),
    E(FLAT_STORE_DWORDX3, FLAT_STORE_DWORDX3), E(FLAT_STORE_DWORDX4, FLAT_STORE_DWORDX4),
    E(GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_UBYTE), E(GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_SBYTE),
    E(GLOBAL_LOAD_USHORT, GLOBAL_LOAD_USHORT), E(GLOBAL_LOAD_SSHORT, GLOBAL_LOAD_SSHORT),
    E(GLOBAL_LOAD_SHORT_D16_HI, GLOBAL_LOAD_SHORT_D16_HI),
    E(GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORD), E(GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX2),
    E(GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX3), E(GLOBAL_LOAD_DWORDX4, GLOBAL_LOAD_DWORDX4),
    E(GLOBAL_STORE_BYTE, GLOBAL_STORE_BYTE), E(GLOBAL_STORE_SHORT, GLOBAL_STORE_SHORT),
    E(GLOBAL_STORE_BYTE_D16_HI, GLOBAL_STORE_BYTE_D16_HI),
    E(GLOBAL_STORE_SHORT_D16_HI, GLOBAL_STORE_SHORT_D16_HI),
    E(GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORD), E(GLOBAL_STORE_DWORDX2, GLOBAL_STORE_DWORDX2),
    E(GLOBAL_STORE_DWORDX3, GLOBAL_STORE_DWORDX3), E(GLOBAL_STORE_DWORDX4, GLOBAL_STORE_DWORDX4),
    // Scratch/private-segment VMEM. Keep these as explicit CanonicalOps rather than
    // routing through GLOBAL_*: the handler must preserve scratch swizzling and
    // KD private-segment ABI semantics, not global address-space semantics.
    E(SCRATCH_LOAD_DWORD_ST, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SADDR, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SVS, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORDX2_ST, SCRATCH_LOAD_DWORDX2),
    E(SCRATCH_LOAD_DWORDX2_SADDR, SCRATCH_LOAD_DWORDX2),
    E(SCRATCH_LOAD_DWORDX2, SCRATCH_LOAD_DWORDX2),
    E(SCRATCH_LOAD_DWORDX2_SVS, SCRATCH_LOAD_DWORDX2),
    E(SCRATCH_LOAD_DWORDX3_ST, SCRATCH_LOAD_DWORDX3),
    E(SCRATCH_LOAD_DWORDX3_SADDR, SCRATCH_LOAD_DWORDX3),
    E(SCRATCH_LOAD_DWORDX3, SCRATCH_LOAD_DWORDX3),
    E(SCRATCH_LOAD_DWORDX3_SVS, SCRATCH_LOAD_DWORDX3),
    E(SCRATCH_LOAD_DWORDX4_ST, SCRATCH_LOAD_DWORDX4),
    E(SCRATCH_LOAD_DWORDX4_SADDR, SCRATCH_LOAD_DWORDX4),
    E(SCRATCH_LOAD_DWORDX4, SCRATCH_LOAD_DWORDX4),
    E(SCRATCH_LOAD_DWORDX4_SVS, SCRATCH_LOAD_DWORDX4),
    E(SCRATCH_LOAD_DWORD_ST_gfx11, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_ST_gfx12, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_ST_gfx13, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SADDR_gfx11, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SADDR_gfx12, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SADDR_gfx13, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_gfx11, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_gfx12, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_gfx13, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SVS_gfx11, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SVS_gfx12, SCRATCH_LOAD_DWORD),
    E(SCRATCH_LOAD_DWORD_SVS_gfx13, SCRATCH_LOAD_DWORD),
    E(SCRATCH_STORE_DWORD_ST, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SADDR, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SVS, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORDX2_ST, SCRATCH_STORE_DWORDX2),
    E(SCRATCH_STORE_DWORDX2_SADDR, SCRATCH_STORE_DWORDX2),
    E(SCRATCH_STORE_DWORDX2, SCRATCH_STORE_DWORDX2),
    E(SCRATCH_STORE_DWORDX2_SVS, SCRATCH_STORE_DWORDX2),
    E(SCRATCH_STORE_DWORDX3_ST, SCRATCH_STORE_DWORDX3),
    E(SCRATCH_STORE_DWORDX3_SADDR, SCRATCH_STORE_DWORDX3),
    E(SCRATCH_STORE_DWORDX3, SCRATCH_STORE_DWORDX3),
    E(SCRATCH_STORE_DWORDX3_SVS, SCRATCH_STORE_DWORDX3),
    E(SCRATCH_STORE_DWORDX4_ST, SCRATCH_STORE_DWORDX4),
    E(SCRATCH_STORE_DWORDX4_SADDR, SCRATCH_STORE_DWORDX4),
    E(SCRATCH_STORE_DWORDX4, SCRATCH_STORE_DWORDX4),
    E(SCRATCH_STORE_DWORDX4_SVS, SCRATCH_STORE_DWORDX4),
    E(SCRATCH_STORE_DWORD_ST_gfx11, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_ST_gfx12, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_ST_gfx13, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SADDR_gfx11, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SADDR_gfx12, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SADDR_gfx13, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_gfx11, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_gfx12, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_gfx13, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SVS_gfx11, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SVS_gfx12, SCRATCH_STORE_DWORD),
    E(SCRATCH_STORE_DWORD_SVS_gfx13, SCRATCH_STORE_DWORD),

    // ---------------------------------------------------------------------
    // FLAT atomics (canonicalized to non-SADDR via getGlobalVaddrOp; no-return form)
    // ---------------------------------------------------------------------
    E(FLAT_ATOMIC_ADD, FLAT_ATOMIC_ADD),
    E(FLAT_ATOMIC_SUB, FLAT_ATOMIC_SUB),
    E(FLAT_ATOMIC_AND, FLAT_ATOMIC_AND),
    E(FLAT_ATOMIC_OR, FLAT_ATOMIC_OR),
    E(FLAT_ATOMIC_XOR, FLAT_ATOMIC_XOR),
    E(FLAT_ATOMIC_SMIN, FLAT_ATOMIC_SMIN),
    E(FLAT_ATOMIC_SMAX, FLAT_ATOMIC_SMAX),
    E(FLAT_ATOMIC_UMIN, FLAT_ATOMIC_UMIN),
    E(FLAT_ATOMIC_UMAX, FLAT_ATOMIC_UMAX),
    E(FLAT_ATOMIC_SWAP, FLAT_ATOMIC_SWAP),
    E(FLAT_ATOMIC_CMPSWAP, FLAT_ATOMIC_CMPSWAP),
    E(FLAT_ATOMIC_ADD_F32, FLAT_ATOMIC_ADD_F32),

    // ---------------------------------------------------------------------
    // GLOBAL atomics
    // ---------------------------------------------------------------------
    E(GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_ADD),
    E(GLOBAL_ATOMIC_SUB, GLOBAL_ATOMIC_SUB),
    E(GLOBAL_ATOMIC_AND, GLOBAL_ATOMIC_AND),
    E(GLOBAL_ATOMIC_OR, GLOBAL_ATOMIC_OR),
    E(GLOBAL_ATOMIC_XOR, GLOBAL_ATOMIC_XOR),
    E(GLOBAL_ATOMIC_SMIN, GLOBAL_ATOMIC_SMIN),
    E(GLOBAL_ATOMIC_SMAX, GLOBAL_ATOMIC_SMAX),
    E(GLOBAL_ATOMIC_UMIN, GLOBAL_ATOMIC_UMIN),
    E(GLOBAL_ATOMIC_UMAX, GLOBAL_ATOMIC_UMAX),
    E(GLOBAL_ATOMIC_SWAP, GLOBAL_ATOMIC_SWAP),
    E(GLOBAL_ATOMIC_CMPSWAP, GLOBAL_ATOMIC_CMPSWAP),
    E(GLOBAL_ATOMIC_ADD_F32, GLOBAL_ATOMIC_ADD_F32),
    E(GLOBAL_ATOMIC_PK_ADD_BF16, GLOBAL_ATOMIC_PK_ADD_BF16),
    E(GLOBAL_ATOMIC_PK_ADD_F16, GLOBAL_ATOMIC_PK_ADD_F16),

    // ---------------------------------------------------------------------
    // SMEM atomics (enumerate addressing forms: IMM / SGPR / SGPR_IMM)
    // ---------------------------------------------------------------------
    E(S_ATOMIC_SWAP_IMM, S_ATOMIC_SWAP),
    E(S_ATOMIC_SWAP_SGPR, S_ATOMIC_SWAP),
    E(S_ATOMIC_SWAP_SGPR_IMM, S_ATOMIC_SWAP),
    // S_ATOMIC_DEC (scalar wrap-decrement) -- the three addressing forms
    // (IMM / SGPR / SGPR_IMM) share a CanonicalOp.  The `_RTN` variants the
    // disassembler emits for the GLC=1 returned-old-value form collapse
    // onto these non-RTN pseudos via the `_RTN` alias rule below; the
    // handler keys off `di.numDefs` (== MCInstrDesc::getNumDefs()) for
    // the write-back decision, so a single lowering covers both halves
    // of the contract.
    E(S_ATOMIC_DEC_IMM, S_ATOMIC_DEC),
    E(S_ATOMIC_DEC_SGPR, S_ATOMIC_DEC),
    E(S_ATOMIC_DEC_SGPR_IMM, S_ATOMIC_DEC),

    // ---------------------------------------------------------------------
    // DS
    // ---------------------------------------------------------------------
    E(DS_LOAD_TR16_B128, DS_LOAD_TR16_B128),
    E(DS_READ_B64_TR_B16, DS_READ_B64_TR_B16),
    E(DS_READ_B64_TR_B8, DS_READ_B64_TR_B8),
    // gfx1250 64-bit transposed LDS load (i8 element, 8 elements per
    // lane post-transpose). Distinct LLVM MC pseudo from the gfx950
    // `DS_READ_B64_TR_B8` sibling -- same semantics, different isel
    // gate (isGFX1250Plus vs HasGFX950Insts). The handler folds both
    // paths into a shared hand-rolled emulation; canonicalising the
    // CanonicalOp on the gfx1250 spelling matches the disassembly the
    // raise_cli operator sees.
    E(DS_LOAD_TR8_B64, DS_LOAD_TR8_B64),
    E(DS_READ_B32, DS_READ_B32), E(DS_READ_B64, DS_READ_B64),
    // 96-bit LDS load. LLVM MC keeps the legacy `DS_READ_B96`
    // pseudo name for what gfx11+ disassembles as `ds_load_b96`
    // (DSInstructions.td:1578); we canonicalise on the gfx11+
    // spelling, mirroring the s_set_pc_i64 / DS_LOAD_TR8_B64
    // precedent. The handler dispatches via the existing generic
    // DS read range -- see dsClassify in handle-ds.cpp for the
    // {dwords=3, loadBits=96} entry.
    E(DS_READ_B96, DS_READ_B96),
    E(DS_READ_B128, DS_READ_B128),
    E(DS_READ2_B32, DS_READ2_B32), E(DS_READ2_B64, DS_READ2_B64),
    // gfx11+ stride-64 two-address LDS loads (DSInstructions.td:1529,
    // 1542 -- `ds_load_2addr_stride64_b{32,64}`). Same two-offset MC
    // shape as the non-ST64 siblings; the canonicalization chain
    // collapses the _gfx9/_gfx11/_gfx12 reals onto the pseudo
    // forms here, mirroring the DS_READ2_B{32,64} entries above.
    E(DS_READ2ST64_B32, DS_READ2ST64_B32),
    E(DS_READ2ST64_B64, DS_READ2ST64_B64),
    E(DS_READ_U16, DS_READ_U16), E(DS_READ_I16, DS_READ_I16),
    E(DS_READ_U8, DS_READ_U8), E(DS_READ_I8, DS_READ_I8),
    E(DS_WRITE_B32, DS_WRITE_B32), E(DS_WRITE_B64, DS_WRITE_B64),
    // Symmetric 96-bit LDS store. gfx11+ asm spelling is
    // `ds_store_b96` (DSInstructions.td:1576); the LLVM MC opcode
    // remains `DS_WRITE_B96`.
    E(DS_WRITE_B96, DS_WRITE_B96),
    E(DS_WRITE_B128, DS_WRITE_B128),
    E(DS_WRITE2_B32, DS_WRITE2_B32), E(DS_WRITE2_B64, DS_WRITE2_B64),
    // gfx11+ stride-64 two-address LDS stores -- mirror of the
    // DS_READ2ST64 read-side entries above.
    E(DS_WRITE2ST64_B32, DS_WRITE2ST64_B32),
    E(DS_WRITE2ST64_B64, DS_WRITE2ST64_B64),
    E(DS_WRITE_B16, DS_WRITE_B16), E(DS_WRITE_B8, DS_WRITE_B8),
    // gfx8+ HasD16LoadStore D16_HI store family (DSInstructions.td
    // §604-606). Stores bits [31:16] (B16_HI) or bits [23:16] (B8_HI)
    // of the source VGPR to LDS -- same VGPR/i32 source operand
    // shape as their non-_HI siblings, so the canonical-table macro
    // routes both encoding forks (gfx10 m0-based vs gfx11+ no-m0)
    // through the same CanonicalOp.
    E(DS_WRITE_B16_D16_HI, DS_WRITE_B16_D16_HI),
    E(DS_WRITE_B8_D16_HI, DS_WRITE_B8_D16_HI),
    E(DS_BPERMUTE_B32, DS_BPERMUTE_B32),
    // ds_swizzle_b32 -- wave-width-specific cross-lane shuffle. The
    // handler refuses with `unsupportedInstructionForm` until the P6 rewrite
    // lands (see the ds_swizzle_b32 row of hotswap/docs/wave-size-
    // translation.md §5.3); the wave-size classifier (Phase 1.4.5)
    // flags it as a Class 2 obstruction (wave-size-translation.md
    // §6) in the cross-wave case.
    E(DS_SWIZZLE_B32, DS_SWIZZLE_B32),

    // ---------------------------------------------------------------------
    // MUBUF direct-to-LDS loads (distinct semantics from VGPR-dest loads)
    // ---------------------------------------------------------------------
    // LLVM only ships DWORD and DWORDX4 LDS pseudos -- the DWORDX2_LDS CanonicalOp
    // stays unmapped until an LLVM pseudo exists for it.
    MUBUF4(BUFFER_LOAD_DWORD_LDS, BUFFER_LOAD_DWORD_LDS),
    MUBUF4(BUFFER_LOAD_DWORDX4_LDS, BUFFER_LOAD_DWORDX4_LDS),

    // ---------------------------------------------------------------------
    // MUBUF loads
    // ---------------------------------------------------------------------
    // gfx10- uses `BUFFER_LOAD_DWORD_*`; gfx11+ (vbuffer) uses the same
    // pseudo name with a `_VBUFFER_` infix rather than a `B32` rename.
    MUBUF4(BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORD),
    VBUF4(BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORD),
    MUBUF4(BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX2),
    MUBUF4(BUFFER_LOAD_DWORDX3, BUFFER_LOAD_DWORDX3),
    MUBUF4(BUFFER_LOAD_DWORDX4, BUFFER_LOAD_DWORDX4),
    MUBUF4(BUFFER_LOAD_UBYTE, BUFFER_LOAD_UBYTE),
    MUBUF4(BUFFER_LOAD_SBYTE, BUFFER_LOAD_SBYTE),
    MUBUF4(BUFFER_LOAD_USHORT, BUFFER_LOAD_USHORT),
    MUBUF4(BUFFER_LOAD_SSHORT, BUFFER_LOAD_SSHORT),
    // D16 partial-write loads: the 16-bit (or 8-bit, sign/zero-extended
    // to i16) datum lands in the lo half (`_D16`) or hi half
    // (`_D16_HI`) of the destination VGPR, preserving the other 16
    // bits (BUFInstructions.td:1155-1177, predicate
    // `D16PreservesUnusedBits`). gfx10- uses the legacy MUBUF
    // encodings; gfx11+/gfx1250 fork to VBUFFER (BUFInstructions.td
    // 2610-2620 maps `_d16_*` mnemonics to the VBUFFER pseudos). Both
    // encodings dispatch to the same partial-write handler in
    // handle-mubuf.cpp via the CanonicalOp; the `D16PreservesUnusedBits`
    // semantics is enforced there by reading the prior dst and
    // merging.
    MUBUF4(BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_SHORT_D16),
    VBUF4(BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_SHORT_D16),
    MUBUF4(BUFFER_LOAD_SHORT_D16_HI, BUFFER_LOAD_SHORT_D16_HI),
    VBUF4(BUFFER_LOAD_SHORT_D16_HI, BUFFER_LOAD_SHORT_D16_HI),
    MUBUF4(BUFFER_LOAD_UBYTE_D16, BUFFER_LOAD_UBYTE_D16),
    VBUF4(BUFFER_LOAD_UBYTE_D16, BUFFER_LOAD_UBYTE_D16),
    MUBUF4(BUFFER_LOAD_UBYTE_D16_HI, BUFFER_LOAD_UBYTE_D16_HI),
    VBUF4(BUFFER_LOAD_UBYTE_D16_HI, BUFFER_LOAD_UBYTE_D16_HI),
    MUBUF4(BUFFER_LOAD_SBYTE_D16, BUFFER_LOAD_SBYTE_D16),
    VBUF4(BUFFER_LOAD_SBYTE_D16, BUFFER_LOAD_SBYTE_D16),
    MUBUF4(BUFFER_LOAD_SBYTE_D16_HI, BUFFER_LOAD_SBYTE_D16_HI),
    VBUF4(BUFFER_LOAD_SBYTE_D16_HI, BUFFER_LOAD_SBYTE_D16_HI),

    // ---------------------------------------------------------------------
    // MUBUF stores
    // ---------------------------------------------------------------------
    MUBUF4(BUFFER_STORE_DWORD, BUFFER_STORE_DWORD),
    VBUF4(BUFFER_STORE_DWORD, BUFFER_STORE_DWORD),
    MUBUF4(BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX2),
    MUBUF4(BUFFER_STORE_DWORDX3, BUFFER_STORE_DWORDX3),
    MUBUF4(BUFFER_STORE_DWORDX4, BUFFER_STORE_DWORDX4),
    MUBUF4(BUFFER_STORE_BYTE, BUFFER_STORE_BYTE),
    MUBUF4(BUFFER_STORE_SHORT, BUFFER_STORE_SHORT),

    // ---------------------------------------------------------------------
    // MUBUF atomics (non-return; handlers don't distinguish RTN yet)
    // ---------------------------------------------------------------------
    MUBUF4(BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_ADD),
    MUBUF4(BUFFER_ATOMIC_SUB, BUFFER_ATOMIC_SUB),
    MUBUF4(BUFFER_ATOMIC_AND, BUFFER_ATOMIC_AND),
    MUBUF4(BUFFER_ATOMIC_OR, BUFFER_ATOMIC_OR),
    MUBUF4(BUFFER_ATOMIC_XOR, BUFFER_ATOMIC_XOR),
    // Class 3 non-commutative atomics (hotswap/docs/wave-size-
    // translation.md §6). The wave-size classifier flags them in
    // the cross-wave case before
    // dispatch ever reaches handle-mubuf.cpp's switch. Same-wave and
    // same-target lifts are still modeled with raw-buffer atomics there.
    MUBUF4(BUFFER_ATOMIC_SWAP, BUFFER_ATOMIC_SWAP),
    MUBUF4(BUFFER_ATOMIC_CMPSWAP, BUFFER_ATOMIC_CMPSWAP),
    MUBUF4(BUFFER_ATOMIC_ADD_F32, BUFFER_ATOMIC_ADD_F32),
    MUBUF4(BUFFER_ATOMIC_PK_ADD_BF16, BUFFER_ATOMIC_PK_ADD_BF16),
    MUBUF4(BUFFER_ATOMIC_PK_ADD_F16, BUFFER_ATOMIC_PK_ADD_F16),
    // gfx11+/gfx12 VBUFFER fork for the buffer atomics. The asm
    // spelling on gfx11+/gfx1250 renames `BUFFER_ATOMIC_ADD` to
    // `buffer_atomic_add_u32` (BUFInstructions.td:2789 declares
    // `defm BUFFER_ATOMIC_ADD : MUBUF_Real_Atomic_gfx11_gfx12<0x035,
    // "buffer_atomic_add_u32">`); LLVM MC keeps the legacy
    // `BUFFER_ATOMIC_*` pseudo names but suffixes them with
    // `_VBUFFER_<addressing>` to distinguish the gfx12 buffer-
    // descriptor encoding from the legacy MUBUF one. The decoder in
    // mubuf-addr.cpp explicitly recognises both encodings (keys on
    // ParsedReg::Kind rather than operand position) so the existing
    // BUFFER_ATOMIC_* CanonicalOps and raw-buffer atomic lowering in
    // handle-mubuf.cpp's `sop >= BUFFER_ATOMIC_ADD &&
    // sop <= BUFFER_ATOMIC_PK_ADD_F16` branch work unchanged for
    // VBUFFER atomics. Without these entries the gfx1250 corpus
    // (e.g. scope_discovery___sum_bitmatrix_rows refused on
    // `buffer_atomic_add_u32 v0, v1, s[4:7], null offen`) would
    // refuse with `unsupportedOpcode` despite the underlying CanonicalOp
    // being present.
    //
    // PK_ADD_BF16 / PK_ADD_F16 are intentionally omitted -- they have
    // no VBUFFER Real form in BUFInstructions.td (the gfx12 buffer
    // packed-add fork uses a different mnemonic family); a stray
    // VBUF4 entry would expand to AMDGPU enum values that don't
    // exist and fail to compile.
    VBUF4(BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_ADD),
    VBUF4(BUFFER_ATOMIC_SUB, BUFFER_ATOMIC_SUB),
    VBUF4(BUFFER_ATOMIC_AND, BUFFER_ATOMIC_AND),
    VBUF4(BUFFER_ATOMIC_OR, BUFFER_ATOMIC_OR),
    VBUF4(BUFFER_ATOMIC_XOR, BUFFER_ATOMIC_XOR),
    VBUF4(BUFFER_ATOMIC_SWAP, BUFFER_ATOMIC_SWAP),
    VBUF4(BUFFER_ATOMIC_CMPSWAP, BUFFER_ATOMIC_CMPSWAP),
    VBUF4(BUFFER_ATOMIC_ADD_F32, BUFFER_ATOMIC_ADD_F32),

    // ---------------------------------------------------------------------
    // AGPR moves
    // ---------------------------------------------------------------------
    E(V_ACCVGPR_READ_B32_e64, V_ACCVGPR_READ_B32),
    E(V_ACCVGPR_WRITE_B32_e64, V_ACCVGPR_WRITE_B32),

    // ---------------------------------------------------------------------
    // MFMA shapes with a single TableGen pseudo per intrinsic.
    // Register-class variants (`_vgprcd_`, `_mac_`) collapse to the base
    // `_e64` via pseudoAlias, so we only list the base pseudo here.
    // ---------------------------------------------------------------------
    E(V_MFMA_F32_16X16X16F16_e64, V_MFMA_F32_16x16x16_F16),
    E(V_MFMA_F32_32X32X8F16_e64,  V_MFMA_F32_32x32x8_F16),
    E(V_MFMA_F32_16X16X4F32_e64,  V_MFMA_F32_16x16x4_F32),
    E(V_MFMA_F32_32X32X1F32_e64,  V_MFMA_F32_32x32x1_F32),
    E(V_MFMA_F32_32X32X2F32_e64,  V_MFMA_F32_32x32x2_F32),
    E(V_MFMA_F32_4X4X1F32_e64,    V_MFMA_F32_4x4x1_F32),
    E(V_MFMA_F32_16X16X1F32_e64,  V_MFMA_F32_16x16x1_F32),
    E(V_MFMA_F32_32X32X4F16_e64,  V_MFMA_F32_32x32x4_F16),
    E(V_MFMA_F32_16X16X4F16_e64,  V_MFMA_F32_16x16x4_F16),
    E(V_MFMA_F32_4X4X4F16_e64,    V_MFMA_F32_4x4x4_F16),
    E(V_MFMA_I32_16X16X32I8_e64,  V_MFMA_I32_16x16x32_I8),
    E(V_MFMA_I32_32X32X16I8_e64,  V_MFMA_I32_32x32x16_I8),
    E(V_MFMA_I32_32X32X4I8_e64,   V_MFMA_I32_32x32x4_I8),
    E(V_MFMA_I32_16X16X4I8_e64,   V_MFMA_I32_16x16x4_I8),
    E(V_MFMA_I32_4X4X4I8_e64,     V_MFMA_I32_4x4x4_I8),
    E(V_MFMA_F32_16X16X8XF32_e64, V_MFMA_F32_16x16x8_XF32),
    E(V_MFMA_F32_32X32X4XF32_e64, V_MFMA_F32_32x32x4_XF32),
    E(V_MFMA_F32_32X32X2BF16_e64, V_MFMA_F32_32x32x2_BF16),
    E(V_MFMA_F32_16X16X2BF16_e64, V_MFMA_F32_16x16x2_BF16),
    E(V_MFMA_F32_4X4X2BF16_e64,   V_MFMA_F32_4x4x2_BF16),
    // Only the 16x16x16 and 32x32x8 1K shapes were in the legacy mnemonic
    // table; the 4x4x4/16x16x4/32x32x4 1K variants are distinct intrinsics
    // that the raiser does not (yet) model and therefore stay unmapped.
    E(V_MFMA_F32_16X16X16BF16_1K_e64, V_MFMA_F32_16x16x16_BF16_1K),
    E(V_MFMA_F32_32X32X8BF16_1K_e64,  V_MFMA_F32_32x32x8_BF16_1K),
    E(V_MFMA_F32_16X16X32_BF16_e64, V_MFMA_F32_16x16x32_BF16),
    E(V_MFMA_F32_32X32X16_BF16_e64, V_MFMA_F32_32x32x16_BF16),
    E(V_MFMA_F32_16X16X32_F16_e64,  V_MFMA_F32_16x16x32_F16),
    E(V_MFMA_F32_16X16X32_FP8_FP8_e64, V_MFMA_F32_16x16x32_FP8_FP8),
    E(V_MFMA_F32_16X16X32_FP8_BF8_e64, V_MFMA_F32_16x16x32_FP8_BF8),
    E(V_MFMA_F32_16X16X32_BF8_FP8_e64, V_MFMA_F32_16x16x32_BF8_FP8),
    E(V_MFMA_F32_16X16X32_BF8_BF8_e64, V_MFMA_F32_16x16x32_BF8_BF8),
    E(V_MFMA_F32_32X32X16_FP8_FP8_e64, V_MFMA_F32_32x32x16_FP8_FP8),
    E(V_MFMA_F32_32X32X16_FP8_BF8_e64, V_MFMA_F32_32x32x16_FP8_BF8),
    E(V_MFMA_F32_32X32X16_BF8_FP8_e64, V_MFMA_F32_32x32x16_BF8_FP8),
    E(V_MFMA_F32_32X32X16_BF8_BF8_e64, V_MFMA_F32_32x32x16_BF8_BF8),

    // ---------------------------------------------------------------------
    // MFMA F8F6F4: LLVM enumerates all 9 src0/src1 mantissa-type pairs as
    // separate pseudos; the raiser handles them identically.
    // ---------------------------------------------------------------------
    E(V_MFMA_F32_16X16X128_F8F6F4_f4_f4_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f4_f6_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f4_f8_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f6_f4_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f6_f6_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f6_f8_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f8_f4_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f8_f6_e64, V_MFMA_F32_16x16x128_F8F6F4),
    E(V_MFMA_F32_16X16X128_F8F6F4_f8_f8_e64, V_MFMA_F32_16x16x128_F8F6F4),

    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_e64, V_MFMA_SCALE_F32_16x16x128_F8F6F4),

    E(V_MFMA_F32_32X32X64_F8F6F4_f4_f4_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f4_f6_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f4_f8_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f6_f4_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f6_f6_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f6_f8_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f8_f4_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f8_f6_e64, V_MFMA_F32_32x32x64_F8F6F4),
    E(V_MFMA_F32_32X32X64_F8F6F4_f8_f8_e64, V_MFMA_F32_32x32x64_F8F6F4),

    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),
    E(V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_e64, V_MFMA_SCALE_F32_32x32x64_F8F6F4),

    // ---------------------------------------------------------------------
    // WMMA (gfx1250): twoaddr/threeaddr pseudo variants both mean the same
    // semantic op to our raiser.
    // ---------------------------------------------------------------------
    E(V_WMMA_F32_16X16X32_F16_w32_twoaddr, V_WMMA_F32_16x16x32_F16),
    E(V_WMMA_F32_16X16X32_F16_w32_threeaddr, V_WMMA_F32_16x16x32_F16),
    // 16x16x4 WMMA, 32-bit f32 inputs and accumulator (gfx1250 RDNA4
    // VOP3P opcode 0x05D, K=4 across 32 lanes -> per-lane <2 x f32>
    // for A and B). Lifts to the gfx1250-only intrinsic
    // amdgcn_wmma_f32_16x16x4_f32 in the same-target path; cross-
    // target (gfx942) refuses loudly because no K=4 f32 -> MFMA
    // decomposition exists in `wmma-lowering.cpp` yet.
    E(V_WMMA_F32_16X16X4_F32_w32_twoaddr, V_WMMA_F32_16x16x4_F32),
    E(V_WMMA_F32_16X16X4_F32_w32_threeaddr, V_WMMA_F32_16x16x4_F32),
    // 16x16x32 WMMA, BF16 inputs, F32 accumulator (gfx1250 RDNA4 VOP3P
    // opcode 0x062). Same fragment shape as the F16 variant -- the only
    // delta is the input element type, which routes to the bf16 MFMA
    // intrinsic on gfx942 (mfma_f32_16x16x16bf16_1k) and to
    // amdgcn_wmma_f32_16x16x32_bf16 when the target supports WMMA12.
    E(V_WMMA_F32_16X16X32_BF16_w32_twoaddr, V_WMMA_F32_16x16x32_BF16),
    E(V_WMMA_F32_16X16X32_BF16_w32_threeaddr, V_WMMA_F32_16x16x32_BF16),
    // 16x16x64 WMMA, 8-bit element types (gfx1250 RDNA4 VOP3P opcodes
    // 0x06a..0x06d). All four AB combinations share the per-lane
    // fragment shape (A,B: <8 x i32>, C/D: <8 x f32>) and route through
    // the gfx942 MFMA lowering with a per-variant intrinsic dispatch
    // (see `runGroupPass` in wmma-lowering.cpp). Both `_twoaddr` and
    // `_threeaddr` MC pseudo variants represent the same semantic op,
    // mirroring the F16/BF16 mapping above.
    E(V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr,   V_WMMA_F32_16x16x64_FP8_FP8),
    E(V_WMMA_F32_16X16X64_FP8_FP8_w32_threeaddr, V_WMMA_F32_16x16x64_FP8_FP8),
    E(V_WMMA_F32_16X16X64_FP8_BF8_w32_twoaddr,   V_WMMA_F32_16x16x64_FP8_BF8),
    E(V_WMMA_F32_16X16X64_FP8_BF8_w32_threeaddr, V_WMMA_F32_16x16x64_FP8_BF8),
    E(V_WMMA_F32_16X16X64_BF8_FP8_w32_twoaddr,   V_WMMA_F32_16x16x64_BF8_FP8),
    E(V_WMMA_F32_16X16X64_BF8_FP8_w32_threeaddr, V_WMMA_F32_16x16x64_BF8_FP8),
    E(V_WMMA_F32_16X16X64_BF8_BF8_w32_twoaddr,   V_WMMA_F32_16x16x64_BF8_BF8),
    E(V_WMMA_F32_16X16X64_BF8_BF8_w32_threeaddr, V_WMMA_F32_16x16x64_BF8_BF8),
    // 16x16x64 WMMA, IU8 (signed/unsigned 8-bit integer inputs, i32
    // accumulator; gfx1250 RDNA4 VOP3P opcode 0x072). The MC opcode
    // sign-extension knobs travel through the `neg_lo` operand bits
    // rather than the opcode itself, so there's a single CanonicalOp for
    // both signed and unsigned interpretations. On gfx942 this
    // routes to mfma_i32_16x16x32_i8 (i64 packed A/B, <4 x i32>
    // accumulator). Both `_twoaddr` and `_threeaddr` MC pseudo
    // variants represent the same semantic op, mirroring the
    // F16/BF16 / FP8/BF8 mappings above.
    E(V_WMMA_I32_16X16X64_IU8_w32_twoaddr,   V_WMMA_I32_16x16x64_IU8),
    E(V_WMMA_I32_16X16X64_IU8_w32_threeaddr, V_WMMA_I32_16x16x64_IU8),
    // ---------------------------------------------------------------------
    // Scaled WMMA F8F6F4 (gfx1250 RDNA4 -- VOP3PX2 paired form, real opcode
    // 0x033 + 0x35 / 0x3a, FLATInstructions / VOP3PInstructions.td:1987).
    // The 9 mantissa-pair pseudos (`f4_f4`, `f4_f6`, …, `f8_f8`) all share
    // the same CanonicalOp because the per-format element distinction (BF8 vs
    // FP8 within f8, etc.) travels as the `matrix_a_fmt` /
    // `matrix_b_fmt` named-immediate operands rather than the opcode
    // suffix. Each mantissa-pair has both a `_twoaddr` (src2 tied to
    // vdst) and `_threeaddr` (independent vdst) MC pseudo because the
    // `WMMAInstGFX12` multiclass emits both -- the canonicalization
    // chain treats them as the same semantic op (matching the
    // F8F6F4-MFMA collapse rule above and the `_twoaddr / _threeaddr`
    // collapse rule used for the K=32 / K=64 WMMA family). The
    // companion `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4` intrinsic
    // is gfx1250-only (`isGFX125xOnly`); cross-target gfx942 lift is a
    // loud refusal in `handle-valu-vop3p.cpp` because the K=128 +
    // matrix-fmt + scale-exponent decomposition into MFMA is not
    // modelled in `wmma-lowering.cpp`. The non-paired
    // `V_WMMA_F32_16X16X128_F8F6F4` family (no SCALE prefix) is not
    // currently observed in the kerneldex corpus; it can be added
    // alongside this entry when a kernel surfaces it.
    // ---------------------------------------------------------------------
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr,   V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    E(V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr, V_WMMA_SCALE_F32_16x16x128_F8F6F4),
    // ---------------------------------------------------------------------
    // VIMAGE TENSOR (gfx1250 RDNA4 -- VIMAGE 0xc4 / 0xc5).
    // The disassembler's MC opcodes are the `_gfx1250` reals
    // (`TENSOR_LOAD_TO_LDS_d{2,4}_gfx1250`,
    // `TENSOR_STORE_FROM_LDS_d{2,4}_gfx1250`,
    // `MIMGInstructions.td:2115`); the canonicalization chain in
    // `OpcodeMap::canonicalize` collapses each real onto its pseudo
    // (`TENSOR_LOAD_TO_LDS_d{2,4}` / `TENSOR_STORE_FROM_LDS_d{2,4}`)
    // via `buildMcToPseudoMap`. Both the `_d2` (up-to-2D) and `_d4`
    // (up-to-4D) operand-shape variants share a CanonicalOp because their
    // semantic intent and their cross-target refusal contract are
    // identical; `handleVIMAGE` uses `di.Mnemonic` directly when it
    // needs to discriminate (e.g., a future native-target intrinsic
    // lowering that zero-fills D# group 2/3 for the `_d2` form).
    // ---------------------------------------------------------------------
    E(TENSOR_LOAD_TO_LDS_d2,    TENSOR_LOAD_TO_LDS),
    E(TENSOR_LOAD_TO_LDS_d4,    TENSOR_LOAD_TO_LDS),
    E(TENSOR_STORE_FROM_LDS_d2, TENSOR_STORE_FROM_LDS),
    E(TENSOR_STORE_FROM_LDS_d4, TENSOR_STORE_FROM_LDS),
    // ---------------------------------------------------------------------
    // FLAT async global -> LDS (gfx1250 -- VFLAT 0x60-0x62 reals).
    // The disassembler's MC opcodes are the `_gfx1250` (or
    // `_SADDR_gfx1250`) reals declared by
    // `VFLAT_Real_AllAddr_gfx1250` (FLATInstructions.td:2003-2018)
    // off the `FLAT_Global_Load_LDS_Pseudo<…, IsAsync=1>` family
    // (FLATInstructions.td:391-417). The canonicalization chain in
    // `OpcodeMap::canonicalize` collapses each real onto its pseudo
    // (`GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` / `_SADDR`
    // sibling) via `buildMcToPseudoMap`; both the plain and SADDR
    // variants per width share a CanonicalOp because the lift differs
    // only in the source-bank wiring (vaddr-only VGPR_64 vs
    // saddr+voff SGPR_64+VGPR_32). `handleFLAT` uses `op.nSrcs()`
    // (4 for plain, 5 for SADDR) to pick the operand-decoder path,
    // exactly mirroring how `handleVIMAGE` discriminates
    // `tensor_load_to_lds_d2` vs `_d4`.
    // ---------------------------------------------------------------------
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B8,        GLOBAL_LOAD_ASYNC_TO_LDS_B8),
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B8_SADDR,  GLOBAL_LOAD_ASYNC_TO_LDS_B8),
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B32,       GLOBAL_LOAD_ASYNC_TO_LDS_B32),
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B32_SADDR, GLOBAL_LOAD_ASYNC_TO_LDS_B32),
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B64,       GLOBAL_LOAD_ASYNC_TO_LDS_B64),
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B64_SADDR, GLOBAL_LOAD_ASYNC_TO_LDS_B64),
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B128,      GLOBAL_LOAD_ASYNC_TO_LDS_B128),
    E(GLOBAL_LOAD_ASYNC_TO_LDS_B128_SADDR,GLOBAL_LOAD_ASYNC_TO_LDS_B128),

    // ---------------------------------------------------------------------
    // FLAT VMEM prefetch (gfx1250 RDNA4 -- VFLAT 0x05D, hint-class).
    //
    // The disassembler returns the gfx1250 reals
    // `GLOBAL_PREFETCH_B8_gfx1250` (plain VGPR_64 vaddr) and
    // `GLOBAL_PREFETCH_B8_SADDR_gfx1250` (SReg_64 + VGPR_32 SADDR;
    // FLATInstructions.td:545-553 + FLATInstructions.td:3677-3678,
    // `VFLAT_Real_AllAddr_gfx1250<0x05d>`). The canonicalization chain
    // collapses each `_gfx1250` real onto its pseudo via
    // `buildMcToPseudoMap`. The two addressing-mode pseudos share a
    // single CanonicalOp because they map onto the same intrinsic
    // (`int_amdgcn_global_prefetch`, IntrinsicsAMDGPU.td:3211); the
    // handler discriminates on `op.nSrcs()` (3 vs 4) the same way the
    // GLOBAL_LOAD_ASYNC_TO_LDS_B* family above does.
    //
    // `FLAT_PREFETCH_B8{,_SADDR}_gfx1250` (plain VGPR_64 / SADDR) lift
    // to `int_amdgcn_flat_prefetch`.
    // ---------------------------------------------------------------------
    E(GLOBAL_PREFETCH_B8,       GLOBAL_PREFETCH_B8),
    E(GLOBAL_PREFETCH_B8_SADDR, GLOBAL_PREFETCH_B8),
    E(FLAT_PREFETCH_B8,         FLAT_PREFETCH_B8),
    E(FLAT_PREFETCH_B8_SADDR,   FLAT_PREFETCH_B8),
};

#undef SMEM3
#undef VBUF4
#undef MUBUF4
#undef E

// Iteration bound for SIEncodingFamily: the enum in SIDefines.h is a closed
// numeric set with GFX13 as the current maximum, so we scan [0, GFX13] when
// inverting the pseudo -> MC map.  If LLVM adds a new family the next
// enumerator value appears here automatically and the build still compiles;
// the static_assert keeps us honest if LLVM ever renames the sentinel we use.
static_assert(SIEncodingFamily::GFX13 >= SIEncodingFamily::SI,
              "SIEncodingFamily enum layout changed unexpectedly");
constexpr unsigned KNumEncodingFamilies =
    static_cast<unsigned>(SIEncodingFamily::GFX13) + 1;

// Build a reverse map MC-opcode -> canonical pseudo by scanning every pseudo
// opcode across all subtarget generations. This is ~O(N * 15) work at init
// time (N ~= 70k AMDGPU opcodes on recent LLVM), which is well under a
// millisecond on modern hardware and done once per raiser.
DenseMap<unsigned, unsigned>
buildMcToPseudoMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    for (unsigned Gen = 0; Gen < KNumEncodingFamilies; ++Gen) {
      int Mc = AMDGPU::getMCOpcode(P, Gen);
      if (Mc > 0 && static_cast<unsigned>(Mc) != P)
        Result.try_emplace(static_cast<unsigned>(Mc), P);
    }
  }
  return Result;
}

// Rule predicates: an optional semantic invariant the alias must preserve.
// Every time an alias step is committed (source pseudo S collapses onto
// target pseudo T), the firing rule's predicate is evaluated against
// MCInstrDesc(S) and MCInstrDesc(T). A violation means LLVM renamed or
// repurposed a pseudo in a way that breaks our naming contract, and is
// reported as a fatal error at init time rather than silently producing
// wrong IR at runtime.
using RulePredicate = bool (*)(const MCInstrDesc &Src, const MCInstrDesc &Tgt);

// `_RTN` collapse: source must be an atomic with a return value; target must
// be the same atomic without one. The raiser uses `numDefs` as the
// "publishes old value" signal, so that invariant must also hold.
static bool atomicRetToNoRet(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  constexpr uint64_t KRet = SIInstrFlags::IsAtomicRet;
  constexpr uint64_t KNoRet = SIInstrFlags::IsAtomicNoRet;
  return (Src.TSFlags & KRet) && (Tgt.TSFlags & KNoRet) &&
         Src.getNumDefs() > 0 && Tgt.getNumDefs() == 0;
}

// `_vgprcd_` / `_mac_` collapse: both source and target must be MFMA
// (matrix-accumulate) pseudos.
static bool bothAreMAI(const MCInstrDesc &Src, const MCInstrDesc &Tgt) {
  constexpr uint64_t KMAI = SIInstrFlags::IsMAI;
  return (Src.TSFlags & KMAI) && (Tgt.TSFlags & KMAI);
}

// `_nosdst_` collapse: starting with GFX11, VOPC CMPX instructions no longer
// write a scalar destination register (EXEC receives the mask directly) and
// LLVM represents this as a `_nosdst_` variant. The non-`_nosdst_` target
// form keeps the scalar dst (for older subtargets). Both forms share
// dispatch-relevant TSFlags; the raiser's CMPX handler only writes EXEC and
// ignores the optional sdst, so collapsing the variant onto the base is
// safe. The source has one fewer def when the base includes sdst (e64
// forms) and the same number of defs otherwise (e32, where both lack sdst).

// Bits we require to be identical between source and target for an alias
// collapse to be considered semantically safe. Deliberately excludes encoding
// variation flags like `VOP3_OPSEL` (set on `_t16_` op-sel encodings but not
// on the base `_e64`) and `renamedInGFX9` (set only on the subtarget-specific
// pseudo). Everything listed below represents *what the handler dispatches
// on*: instruction family (SOP/VOP/FLAT/DS/...), atomic kind, and MAI.
static constexpr uint64_t KSemanticShapeMask =
    // Instruction families.
    SIInstrFlags::SOP1 | SIInstrFlags::SOP2 | SIInstrFlags::SOPC |
    SIInstrFlags::SOPK | SIInstrFlags::SOPP | SIInstrFlags::VOP1 |
    SIInstrFlags::VOP2 | SIInstrFlags::VOPC | SIInstrFlags::VOP3 |
    SIInstrFlags::VOP3P | SIInstrFlags::SDWA | SIInstrFlags::DPP |
    SIInstrFlags::MUBUF | SIInstrFlags::MTBUF | SIInstrFlags::SMRD |
    SIInstrFlags::FLAT | SIInstrFlags::DS | SIInstrFlags::MIMG |
    // Semantic classification (atomic kind, MAI).
    SIInstrFlags::IsAtomicRet | SIInstrFlags::IsAtomicNoRet |
    SIInstrFlags::IsMAI;

// Subtarget-/operand-class variants (`_gfx9`, `_t16_`, `_fake16_`, `_agpr`,
// etc.) may legitimately toggle encoding flags such as `VOP3_OPSEL` or
// `renamedInGFX9` between source and target, but they must preserve the
// instruction's dispatch identity: same family, same atomic kind, same MAI
// classification, same def arity. A violation means LLVM renamed or
// repurposed a pseudo in a way our alias map cannot safely collapse.
static bool sameSemanticShape(const MCInstrDesc &Src,
                              const MCInstrDesc &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Src.getNumDefs() == Tgt.getNumDefs();
}

// `_nosdst_` collapse: same dispatch identity as the base, and the source
// never has more defs than the target (the scalar dst is either dropped
// entirely or added back on the target's e64 form).
static bool nosdstDropsScalarDef(const MCInstrDesc &Src,
                                 const MCInstrDesc &Tgt) {
  return (Src.TSFlags & KSemanticShapeMask) ==
             (Tgt.TSFlags & KSemanticShapeMask) &&
         Tgt.getNumDefs() >= Src.getNumDefs() &&
         Tgt.getNumDefs() - Src.getNumDefs() <= 1;
}

// Build an alias map that collapses "parallel" pseudos LLVM generates for the
// same semantic instruction into a single canonical pseudo. Examples:
//   DS_WRITE_B16_gfx9        -> DS_WRITE_B16
//   V_ADD_F16_t16_e64        -> V_ADD_F16_e64
//   V_ADD_F16_fake16_e64     -> V_ADD_F16_e64
// LLVM does not expose a helper for this collapse, so we match on pseudo name
// at init time. Name lookups are confined to this one-shot scan over
// `MCII.getNumOpcodes()`; runtime lookups remain pure DenseMap hits.
DenseMap<unsigned, unsigned>
buildPseudoAliasMap(const MCInstrInfo &MCII) {
  unsigned NumOpc = MCII.getNumOpcodes();

  llvm::StringMap<unsigned> ByName;
  for (unsigned P = 0; P < NumOpc; ++P)
    ByName.try_emplace(MCII.getName(P), P);

  struct Rule {
    llvm::StringRef Needle;
    bool IsSuffix;
    // Optional semantic check on (source, target) MCInstrDesc. A null
    // predicate means "no validation yet" (see the older subtarget/operand
    // markers below).
    RulePredicate Pred;
  };
  // Subtarget-specific markers ("_gfx9", "_gfx1250", ...) and operand-size
  // markers ("_t16_", "_fake16_") that LLVM injects into the pseudo name.
  // A single pseudo can carry multiple markers (e.g.
  // `V_BITOP3_B16_gfx1250_fake16_e64`), so the outer loop below applies these
  // rules iteratively until the name stops shrinking.
  static const Rule Rules[] = {
      // Subtarget-specific markers. LLVM emits a dedicated pseudo per
      // subtarget (e.g. `_gfx9`, `_gfx1250`, `_vi_gfx9`) with the same
      // TableGen class as the base; collapsing them is sound as long as
      // TSFlags and def arity match.
      {"_vi_gfx9", true, sameSemanticShape},
      {"_gfx9", true, sameSemanticShape},
      {"_gfx1250", true, sameSemanticShape},
      {"_gfx1250_", false, sameSemanticShape},
      {"_pseudo_", false, sameSemanticShape},
      // True16 / Fake16 mark the 16-bit operand encoding variant; LLVM has
      // no dedicated TSFlag bit for this (the distinction lives in
      // True16Predicate on the TableGen side), and the t16 encoding toggles
      // `VOP3_OPSEL`. We cross-check that dispatch-relevant TSFlags and def
      // arity are preserved, but tolerate encoding-bit drift.
      {"_t16_", false, sameSemanticShape},
      {"_fake16_", false, sameSemanticShape},
      // `_OP_SEL_` infix marks the gfx11+ encoding variant for VOP1 cvt
      // instructions (e.g. v_cvt_f32_{fp8,bf8}, v_cvt_pk_f32_{fp8,bf8}).
      // The OP_SEL pseudo carries an extra `byte_sel` immediate operand and
      // toggles `VOP3_OPSEL`/`maybeAtomic`/`ASYNC_CNT` bits relative to the
      // base e64 pseudo, but dispatch identity (instruction family + def
      // arity + atomic/MAI classification) is preserved. Collapsing onto
      // the base pseudo lets a single CanonicalOp handler service both the
      // pre-gfx11 SDWA/byte_sel-via-disassembly form and the gfx11+ encoded
      // byte_sel form; the handler reads the byte_sel from the disassembly
      // text (`op_sel:`), which is identical for both.
      {"_OP_SEL_", false, sameSemanticShape},
      // GFX11+ VOPC CMPX family drops the scalar destination register; the
      // raiser's CMPX handler only touches EXEC so the `_nosdst_` form
      // collapses cleanly onto the base pseudo of the same encoding width.
      {"_nosdst_", false, nosdstDropsScalarDef},
      // MFMA register-class modifiers.  `_vgprcd_` marks a VGPR destination
      // variant; `_mac_` marks a multiply-accumulate (tied dst/src2) variant.
      // Both keep the same TableGen intrinsic and semantic shape, so they
      // collapse onto the base `_e64` pseudo.
      {"_vgprcd_", false, bothAreMAI},
      {"_mac_", false, bothAreMAI},
      // Atomic return-value variants: LLVM emits distinct `_RTN` pseudos for
      // the forms that return the pre-modification value, plus `_agpr`
      // variants that just pick an AGPR destination register class. These
      // pseudos carry the same TableGen intrinsic and identical semantics;
      // the only difference is whether the handler should write the result
      // back, which the raiser already derives from `di.numDefs`
      // (MCInstrDesc::getNumDefs()). Collapse them onto the non-RTN pseudo
      // so both forms share a single CanonicalOp.
      {"_agpr", true, sameSemanticShape},
      {"_RTN", true, atomicRetToNoRet},
  };

  // Returns the index of the firing rule or -1 if no rule applies.
  auto StripOnce = [&](llvm::StringRef Name, std::string &Out) -> int {
    for (size_t I = 0; I < std::size(Rules); ++I) {
      const Rule &R = Rules[I];
      if (R.IsSuffix) {
        if (!Name.ends_with(R.Needle))
          continue;
        Out = Name.drop_back(R.Needle.size()).str();
        return static_cast<int>(I);
      }
      size_t Pos = Name.find(R.Needle);
      if (Pos == llvm::StringRef::npos)
        continue;
      Out = (Name.substr(0, Pos).str() + std::string("_") +
             Name.substr(Pos + R.Needle.size()).str());
      return static_cast<int>(I);
    }
    return -1;
  };

  DenseMap<unsigned, unsigned> Alias;
  for (const auto &Kv : ByName) {
    std::string Cur = Kv.first().str();
    unsigned CurOpc = Kv.second;
    unsigned FinalOpc = Kv.second;
    while (true) {
      std::string Next;
      int RuleIdx = StripOnce(Cur, Next);
      if (RuleIdx < 0)
        break;
      auto It = ByName.find(Next);
      if (It != ByName.end() && It->second != Kv.second) {
        const Rule &R = Rules[RuleIdx];
        if (R.Pred && !R.Pred(MCII.get(CurOpc), MCII.get(It->second))) {
          report_fatal_error(
              Twine("opcode_map: alias rule '") + R.Needle +
              "' broke its semantic invariant while collapsing '" + Cur +
              "' -> '" + Next +
              "'. LLVM likely renamed or repurposed a pseudo; update the "
              "alias rules or the predicate.");
        }
        CurOpc = It->second;
        FinalOpc = CurOpc;
      }
      Cur = std::move(Next);
    }
    if (FinalOpc != Kv.second)
      Alias.try_emplace(Kv.second, FinalOpc);
  }
  return Alias;
}

// Build a reverse DPP map: DPP opcode -> base VOP opcode. LLVM only provides
// forward mappings (base -> DPP32 / DPP64), so we invert by scanning.
DenseMap<unsigned, unsigned>
buildDppToBaseMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    int D32 = AMDGPU::getDPPOp32(P);
    if (D32 > 0)
      Result.try_emplace(static_cast<unsigned>(D32), P);
    int D64 = AMDGPU::getDPPOp64(P);
    if (D64 > 0)
      Result.try_emplace(static_cast<unsigned>(D64), P);
  }
  return Result;
}

// Canonicalize any MC opcode `mc` to the pseudo form matched in kCanonTable.
// The chain is:
//   MC -> pseudo                (TableGen Subtarget map)
//   pseudo -> base VOP          (strip DPP / SDWA)
//   e32 -> e64                  (collapse VOP encoding variants)
//   SADDR -> VADDR              (FLAT/GLOBAL global-saddr table)
unsigned canonicalize(unsigned Mc,
                      const MCInstrInfo &MCII,
                      const DenseMap<unsigned, unsigned> &McToPseudo,
                      const DenseMap<unsigned, unsigned> &PseudoAlias,
                      const DenseMap<unsigned, unsigned> &DppToBase) {
  unsigned P = Mc;

  // MC (subtarget-specific real) -> pseudo.
  if (auto It = McToPseudo.find(P); It != McToPseudo.end())
    P = It->second;

  // Parallel-pseudo alias -> base pseudo (strips _gfx9, _t16_, _fake16_).
  if (auto It = PseudoAlias.find(P); It != PseudoAlias.end())
    P = It->second;

  // DPP -> base. This handles both VOP2-like _dpp pseudos and VOP3-like
  // _e64_dpp pseudos; the reverse map was built from both getDPPOp32 and
  // getDPPOp64.
  if (auto It = DppToBase.find(P); It != DppToBase.end())
    P = It->second;

  // SDWA -> base. LLVM provides a forward helper for this direction.
  int Base = AMDGPU::getBasicFromSDWAOp(P);
  if (Base > 0)
    P = static_cast<unsigned>(Base);

  // e32 -> e64.
  int E64 = AMDGPU::getVOPe64(P);
  if (E64 > 0)
    P = static_cast<unsigned>(E64);

  // Re-apply the pseudo-alias step. `getVOPe64` can resolve an `_e32` pseudo
  // (e.g. `V_LSHLREV_B64_pseudo_e32`) to an `_e64` pseudo with a parallel
  // variant marker (`V_LSHLREV_B64_pseudo_e64`) that only collapses once
  // both the `_pseudo_` infix and `_e64` suffix are visible together.
  if (auto It = PseudoAlias.find(P); It != PseudoAlias.end())
    P = It->second;

  // FLAT/GLOBAL SADDR -> VADDR. Only applicable to instructions tagged with
  // the FLAT format flag; the helper returns -1 for non-FLAT opcodes but
  // checking the flag first avoids the lookup for every non-FLAT opcode.
  if (P < MCII.getNumOpcodes() &&
      (MCII.get(P).TSFlags & SIInstrFlags::FLAT) != 0) {
    int Vaddr = AMDGPU::getGlobalVaddrOp(P);
    if (Vaddr > 0)
      P = static_cast<unsigned>(Vaddr);
  }

  return P;
}

// Parse a canonical vector-compare pseudo name into (predicate, bits, kind).
// Accepted shape: `V_CMP_<PRED>_<TYPE><BITS>_e64` where
//   PRED  ∈ {EQ, NE, GT, GE, LT, LE, LG, NEQ, NLT, NLE, NGT, NGE, NLG, U, O,
//            CLASS}
//   TYPE  ∈ {U, I, F} (CLASS only ever appears with TYPE=F)
//   BITS  ∈ {16, 32, 64}
// and an optional `V_CMPX_` prefix plays the role of `V_CMP_`. Returns
// `std::nullopt` for anything else; caller is responsible for only passing
// compare-family pseudos.
//
// Rationale: LLVM exposes `AMDGPU::getVCMPXOpFromVCMP` as a V_CMP -> V_CMPX
// mapping, but no public helper that hands back a CmpInst::Predicate or
// element width. Rather than hand-list 100 opcode->metadata pairs we parse
// the pseudo name once at init time; the same token grammar is already
// hard-coded in LLVM's TableGen for these instructions.
//
// CLASS is special: `V_CMP_CLASS_F<bits>` is *not* a predicate compare. src1
// is an i32 mask of FP classes (signaling NaN, quiet NaN, ±inf, ±normal,
// ±subnormal, ±0), and the result lane bit is set iff src0's IEEE class
// matches any enabled bit in the mask. We collapse it onto the same
// `V_CMP` / `V_CMPX` CanonicalOps and signal the special-case lift via
// `VCmpMeta::isClass`; the dispatch in handle-valu-vcmp.cpp branches on
// that flag and emits `llvm.amdgcn.class.f<bits>` instead of an FCmp.
// This keeps the parser surface narrow (one extra grammar branch, no new
// CanonicalOps) and matches the wave-mask write-back path used by the other
// V_CMP forms.
std::optional<VCmpMeta> parseVCmpPseudoName(llvm::StringRef Name) {
  llvm::StringRef Rest = Name;
  if (!Rest.consume_front("V_CMPX_") && !Rest.consume_front("V_CMP_"))
    return std::nullopt;
  if (!Rest.consume_back("_e64"))
    return std::nullopt;

  auto [predTok, typeTok] = Rest.rsplit('_');
  if (predTok.empty() || typeTok.size() < 2)
    return std::nullopt;

  const char TypeCh = typeTok[0];
  unsigned Bits = 0;
  if (typeTok.drop_front().getAsInteger(10, Bits))
    return std::nullopt;
  if (Bits != 16 && Bits != 32 && Bits != 64)
    return std::nullopt;

  using llvm::CmpInst;
  VCmpMeta M{CmpInst::BAD_ICMP_PREDICATE, static_cast<uint8_t>(Bits), false};

  // V_CMP_CLASS_F<bits> / V_CMPX_CLASS_F<bits>: floating-point classification
  // mask, not a predicate compare. The handler takes the `isClass` branch and
  // ignores `pred`; we leave `pred` as BAD_ICMP_PREDICATE so any accidental
  // FCmp/ICmp use would assert loudly rather than silently miscompile.
  if (predTok == "CLASS") {
    if (TypeCh != 'F')
      return std::nullopt;
    M.IsFloat = true;
    M.IsClass = true;
    return M;
  }

  if (TypeCh == 'F') {
    M.IsFloat = true;
    // Float predicates: ordered variants set the O-prefix predicates;
    // N-prefixed AMDGPU names select the "unordered-or-..." complements.
    if (predTok == "EQ")        M.Pred = CmpInst::FCMP_OEQ;
    else if (predTok == "GT")   M.Pred = CmpInst::FCMP_OGT;
    else if (predTok == "GE")   M.Pred = CmpInst::FCMP_OGE;
    else if (predTok == "LT")   M.Pred = CmpInst::FCMP_OLT;
    else if (predTok == "LE")   M.Pred = CmpInst::FCMP_OLE;
    // LG ("less or greater"), NE, and NEQ all mean "ordered and !=" in
    // AMDGPU's model and all lower to FCMP_ONE.
    else if (predTok == "LG" || predTok == "NE" || predTok == "NEQ")
                                M.Pred = CmpInst::FCMP_ONE;
    else if (predTok == "NLT")  M.Pred = CmpInst::FCMP_UGE;
    else if (predTok == "NLE")  M.Pred = CmpInst::FCMP_UGT;
    else if (predTok == "NGT")  M.Pred = CmpInst::FCMP_ULE;
    else if (predTok == "NGE")  M.Pred = CmpInst::FCMP_ULT;
    // NLG ("not (less or greater)") is the unordered-or-equal complement.
    else if (predTok == "NLG")  M.Pred = CmpInst::FCMP_UEQ;
    else if (predTok == "U")    M.Pred = CmpInst::FCMP_UNO;
    else if (predTok == "O")    M.Pred = CmpInst::FCMP_ORD;
    else return std::nullopt;
  } else if (TypeCh == 'U' || TypeCh == 'I') {
    const bool IsSigned = TypeCh == 'I';
    if (predTok == "EQ")        M.Pred = CmpInst::ICMP_EQ;
    else if (predTok == "NE")   M.Pred = CmpInst::ICMP_NE;
    else if (predTok == "GT")   M.Pred = IsSigned ? CmpInst::ICMP_SGT
                                                   : CmpInst::ICMP_UGT;
    else if (predTok == "GE")   M.Pred = IsSigned ? CmpInst::ICMP_SGE
                                                   : CmpInst::ICMP_UGE;
    else if (predTok == "LT")   M.Pred = IsSigned ? CmpInst::ICMP_SLT
                                                   : CmpInst::ICMP_ULT;
    else if (predTok == "LE")   M.Pred = IsSigned ? CmpInst::ICMP_SLE
                                                   : CmpInst::ICMP_ULE;
    else return std::nullopt;
  } else {
    return std::nullopt;
  }

  return M;
}

} // namespace

CanonicalOp OpcodeMap::lookup(unsigned Opcode) const {
  auto It = Map.find(Opcode);
  return It != Map.end() ? It->second : CanonicalOp::Unknown;
}

const VCmpMeta *OpcodeMap::lookupVCmp(unsigned Opcode) const {
  auto It = Vcmp.find(Opcode);
  return It != Vcmp.end() ? &It->second : nullptr;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // Flatten the static kCanonTable into a DenseMap for O(1) lookups
  // during the subsequent scan over every MC opcode.  This flatten
  // step also serves as a duplicate-key audit for `kCanonTable`: the
  // earlier implementation used a plain `canonToSem.try_emplace` loop,
  // which silently keeps the first insertion on key collision.  That
  // silent-first-wins behaviour is exactly what let S_ADD_U64 get
  // mapped twice (once as SOP2-block `E(S_ADD_U64, S_ADD_U64)` and
  // once as gfx12-rename `E(S_ADD_U64, S_ADD_NC_U64)`) with the
  // second row silently losing the routing race and leaving every
  // `s_add_u64` lift routed through the wrong CanonicalOp -- see commit
  // eaee0a0e88 for the repair.  The loop below now aborts loudly on
  // duplicate keys so a future add that re-introduces the collision
  // is caught at transpiler init time instead of quietly miscompiling
  // everything under the duplicated opcode.
  //
  // `report_fatal_error` is used (rather than `assert`) so the check
  // is active in release builds too -- the cost is a single
  // `try_emplace` per table row at process start, which is
  // negligible for a ~800-entry table.
  //
  // "Same CanonicalOp twice" is also rejected.  In principle a redundant
  // row that maps the same MC opcode to the same CanonicalOp is just
  // noise the first-wins rule would swallow harmlessly, but we
  // refuse it anyway so the `kCanonTable` stays honest -- a
  // redundant row is always an editing mistake, never an
  // intentional design, and the abort makes the fix mechanical
  // (delete one of the two rows).
  DenseMap<unsigned, CanonicalOp> CanonToSem;
  CanonToSem.reserve(std::size(kCanonTable));
  for (const Entry &E : kCanonTable) {
    auto [existing, inserted] = CanonToSem.try_emplace(E.Opc, E.Sem);
    if (!inserted) {
      std::string Msg;
      raw_string_ostream Os(Msg);
      Os << "opcode-map.cpp: kCanonTable maps MC opcode '"
         << MCII.getName(E.Opc)
         << "' (enum value " << E.Opc << ") to TWO CanonicalOps: "
         << "first = CanonicalOp::" << canonicalOpName(existing->second)
         << ", second = CanonicalOp::" << canonicalOpName(E.Sem);
      if (existing->second == E.Sem) {
        Os << ".  (Both targets are the same -- the row is redundant; "
              "remove one.)";
      } else {
        Os << ".  `canonToSem.try_emplace` keeps the first insertion, "
              "so every `"
           << MCII.getName(E.Opc)
           << "` lift silently routes through CanonicalOp::"
           << canonicalOpName(existing->second)
           << " and the second row is dead.  Pick ONE CanonicalOp target "
              "and remove the loser row.";
      }
      report_fatal_error(Msg.c_str());
    }
  }

  const unsigned NumOpc = MCII.getNumOpcodes();
  const auto McToPseudo  = buildMcToPseudoMap(NumOpc);
  const auto PseudoAlias = buildPseudoAliasMap(MCII);
  const auto DppToBase   = buildDppToBaseMap(NumOpc);

  Map.clear();
  Vcmp.clear();
  // Heuristic: roughly a quarter of MC opcodes carry a CanonicalOp in practice;
  // resizing a few times is fine for a one-shot init.
  Map.reserve(NumOpc / 4);
  for (unsigned Mc = 0; Mc < NumOpc; ++Mc) {
    const unsigned Canon =
        canonicalize(Mc, MCII, McToPseudo, PseudoAlias, DppToBase);
    if (auto It = CanonToSem.find(Canon); It != CanonToSem.end()) {
      Map[Mc] = It->second;
      continue;
    }
    // The canonical pseudo was not enumerated in `kCanonTable`. Check if it
    // belongs to the V_CMP / V_CMPX family (which is handled via metadata
    // side-table rather than per-opcode enumeration). Use the canonical
    // pseudo's name so we don't have to re-canonicalize any DPP/SDWA
    // variants (those have already been folded by `canonicalize`).
    if (Canon >= NumOpc)
      continue;
    llvm::StringRef CanonName = MCII.getName(Canon);
    const bool IsCmp  = CanonName.starts_with("V_CMP_");
    const bool IsCmpX = CanonName.starts_with("V_CMPX_");
    if (!IsCmp && !IsCmpX)
      continue;
    if (auto Meta = parseVCmpPseudoName(CanonName)) {
      Map[Mc] = IsCmpX ? CanonicalOp::V_CMPX : CanonicalOp::V_CMP;
      Vcmp.try_emplace(Mc, *Meta);
    }
    // Names that start with V_CMP_ but don't parse (e.g. a hypothetical
    // future family) are left as CanonicalOp::Unknown so the raiser reports them
    // loudly rather than silently producing wrong IR.
  }
}

} // namespace COMGR::hotswap
