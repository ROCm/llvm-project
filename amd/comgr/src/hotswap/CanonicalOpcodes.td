//===- CanonicalOpcodes.td - Hotswap canonical opcodes --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The single source of truth for the hotswap raiser's instruction dispatch
// identity.
//
// Two things are declared here:
//
//   * the `CanonicalOp` enumerators, each with the documentation that ends up
//     as a comment above the enumerator in the generated canonical-op.h
//     include; and
//   * which AMDGPU pseudo opcodes canonicalize onto each of them.
//
// `hotswap-tblgen --gen-hotswap-canonical-opcodes` reads this file together
// with AMDGPU.td and emits both the enum and a dense MC-opcode -> CanonicalOp
// lookup table. Every encoding and subtarget variant LLVM generates for a
// pseudo (`_e32`, `_dpp`, `_sdwa`, `_t16_`, `_gfx1250`, ...) is folded onto
// the pseudo named here, so one `Canon` row covers a whole family across
// every AMDGPU generation. See CanonicalOpcodeEmitter.cpp for the collapse.
//
// Naming: the `CanonOp` defs carry a `COP_` prefix because AMDGPU.td already
// defines records named `S_ENDPGM`, `V_ADD_F32_e64`, ... in the same
// RecordKeeper. The prefix is stripped for the C++ enumerator.
//
//===----------------------------------------------------------------------===//

include "AMDGPU.td"

// One `CanonicalOp` enumerator. `Doc` is emitted verbatim as a comment above
// the enumerator in the generated header.
class CanonOpBase<code doc> {
  // The C++ enumerator. Derived from the def name so the two cannot drift;
  // the `COP_` prefix exists only to avoid colliding with the AMDGPU.td
  // instruction records in the same RecordKeeper.
  string EnumName = !substr(NAME, 4);
  code Doc = doc;
  // A range sentinel shares its value with a real canonical opcode and only
  // exists so the raiser can bounds-check a contiguous family. It is never
  // the answer `canonicalOpName` gives for that value.
  bit IsSentinel = 0;
}

// An enumerator with a value of its own.
class CanonOp<code doc = [{}]> : CanonOpBase<doc>;

// An enumerator that shares another's value (`NAME = Target` in C++). Used
// for the range sentinels the raiser compares against.
class CanonOpAlias<CanonOpBase target, code doc = [{}]> : CanonOpBase<doc> {
  CanonOpBase AliasOf = target;
}

// `Pseudo` canonicalizes onto `Op`. Because `Pseudo` is an `Instruction`
// record rather than a string, a rename in AMDGPU.td is a TableGen error
// here rather than a silent lookup miss at runtime.
class Canon<Instruction pseudo, CanonOpBase op> {
  Instruction Pseudo = pseudo;
  CanonOpBase Op = op;
}

// MUBUF: four addressing modes (OFFSET/OFFEN/IDXEN/BOTHEN) per base opcode.
multiclass CanonMUBUF4<string base, CanonOpBase op> {
  foreach mode = ["OFFSET", "OFFEN", "IDXEN", "BOTHEN"] in
    def : Canon<!cast<Instruction>(base # "_" # mode), op>;
}

// The gfx12+ VBUFFER fork of the same four addressing modes, introduced to
// separate the buffer descriptor source.
multiclass CanonVBUF4<string base, CanonOpBase op> {
  foreach mode = ["OFFSET", "OFFEN", "IDXEN", "BOTHEN"] in
    def : Canon<!cast<Instruction>(base # "_VBUFFER_" # mode), op>;
}

// SMEM scalar loads: three operand-source forms (IMM / SGPR / SGPR_IMM).
multiclass CanonSMEM3<string base, CanonOpBase op> {
  foreach form = ["IMM", "SGPR", "SGPR_IMM"] in
    def : Canon<!cast<Instruction>(base # "_" # form), op>;
}

//===----------------------------------------------------------------------===//
// Canonical opcodes
//===----------------------------------------------------------------------===//

def COP_S_ENDPGM : CanonOp<[{
  -- SOPP / control flow --
}]>;

def COP_S_NOP : CanonOp;

def COP_S_BRANCH : CanonOp;

def COP_S_CODE_END : CanonOp;

def COP_S_CBRANCH_SCC0 : CanonOp;

def COP_S_CBRANCH_SCC1 : CanonOp;

def COP_S_CBRANCH_VCCZ : CanonOp;

def COP_S_CBRANCH_VCCNZ : CanonOp;

def COP_S_CBRANCH_EXECZ : CanonOp;

def COP_S_CBRANCH_EXECNZ : CanonOp;

def COP_S_WAITCNT : CanonOp;

def COP_S_WAIT_LOADCNT : CanonOp;

def COP_S_WAIT_STORECNT : CanonOp;

def COP_S_WAIT_KMCNT : CanonOp;

def COP_S_WAIT_DSCNT : CanonOp;

def COP_S_WAIT_XCNT : CanonOp;

def COP_S_WAIT_ASYNCCNT : CanonOp<[{
  gfx1250 async-memory wait counters. `S_WAIT_ASYNCCNT` is the
  companion barrier for the `GLOBAL_LOAD_ASYNC_TO_LDS_B*` family
  below (and `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`); `S_WAIT_TENSORCNT`
  is the companion for `TENSOR_LOAD_TO_LDS` / `TENSOR_STORE_FROM_LDS`.
  Both track dependency counters that do not exist on gfx942 (no
  `ASYNCcnt` / `TENSORcnt` hardware) -- the raiser lowers them as
  no-ops on every target:

  * On gfx942 (cross-target): the source async DMA is emulated
  as a synchronous `load`+`store` chain that has already
  completed by the time the wait is reached. IR dataflow from
  the emulated `store` through subsequent LDS reads carries
  the happens-before; the backend re-inserts an `s_waitcnt
  lgkmcnt(0)` before the reader. See the
  `GLOBAL_LOAD_ASYNC_TO_LDS_B*` CanonicalOp doc block for the full
  trade-off argument.
  * On gfx1250 (same-target): like every other wait counter in
  `handle-sopp.cpp`, the raiser relies on LLVM's memory model
  to re-emit the native wait from the IR's load/store
  ordering.  The async intrinsic's
  `IntrInaccessibleMemOrArgMemOnly` annotation prevents
  reorder across the wait site, so the backend re-derives the
  correct `s_wait_asynccnt` / `s_wait_tensorcnt` from that
  scheduling constraint.

  Declared here (rather than inlined as a generic SOPP no-op) so
  that the canonicalisation is explicit and a future
  reviewer touching the async family can grep
  `S_WAIT_ASYNCCNT` and find both the CanonicalOp, its `Canon`
  entry, and the handler's no-op arm in one pass.
}]>;

def COP_S_WAIT_TENSORCNT : CanonOp;

def COP_S_WAIT_LOADCNT_DSCNT : CanonOp;

def COP_S_WAIT_ALU : CanonOp;

def COP_S_CLAUSE : CanonOp;

def COP_S_DELAY_ALU : CanonOp;

def COP_S_SET_GPR_IDX_ON : CanonOp;

def COP_S_SET_GPR_IDX_OFF : CanonOp;

def COP_S_SETVSKIP : CanonOp;

def COP_S_BARRIER : CanonOp<[{
  Barriers. GFX12+ splits s_barrier into signal + wait; earlier ISAs emit a
  single s_barrier. Handlers model signal as a no-op and wait as a full
  LLVM `amdgcn.s.barrier` call.
}]>;

def COP_S_BARRIER_WAIT : CanonOp;

def COP_S_BARRIER_SIGNAL : CanonOp;

def COP_S_SENDMSG : CanonOp<[{
  s_sendmsg / s_sendmsghalt. Only INTERRUPT and DEALLOC_VGPRS message
  IDs are lifted; see handle-sopp.cpp for the dispatch and policy.
}]>;

def COP_S_SENDMSGHALT : CanonOp;

def COP_S_LOAD_B32 : CanonOp<[{
  -- SMEM --
}]>;

def COP_S_LOAD_B64 : CanonOp;

def COP_S_LOAD_B96 : CanonOp;

def COP_S_LOAD_B128 : CanonOp;

def COP_S_LOAD_B256 : CanonOp;

def COP_S_LOAD_B512 : CanonOp;

def COP_S_BUFFER_LOAD_B32 : CanonOp<[{
  Scalar loads through a four-SGPR buffer resource descriptor.
}]>;

def COP_S_BUFFER_LOAD_B64 : CanonOp;

def COP_S_BUFFER_LOAD_B96 : CanonOp;

def COP_S_BUFFER_LOAD_B128 : CanonOp;

def COP_S_BUFFER_LOAD_B256 : CanonOp;

def COP_S_BUFFER_LOAD_B512 : CanonOp;

def COP_S_LOAD_U8 : CanonOp<[{
  gfx12+ scalar narrow loads: fetch 1 or 2 bytes from a uniform address and
  zero/sign-extend into a 32-bit SGPR. Older ISAs have no equivalent; on a
  cross-target lift to gfx942 the backend will lower the narrow `load iN`
  to VMEM (per-lane global_load_{ubyte,sbyte,ushort,sshort}) -- semantically
  correct but uniformity-lossy. See handle-smem.cpp for the design notes.
}]>;

def COP_S_LOAD_I8 : CanonOp;

def COP_S_LOAD_U16 : CanonOp;

def COP_S_LOAD_I16 : CanonOp;

def COP_S_STORE_B32 : CanonOp;

def COP_S_STORE_B64 : CanonOp;

def COP_S_STORE_B128 : CanonOp;

def COP_S_CMP_EQ_U32 : CanonOp<[{
  -- SOPC --
}]>;

def COP_S_CMP_LG_U32 : CanonOp;

def COP_S_CMP_GT_U32 : CanonOp;

def COP_S_CMP_GE_U32 : CanonOp;

def COP_S_CMP_LT_U32 : CanonOp;

def COP_S_CMP_LE_U32 : CanonOp;

def COP_S_CMP_EQ_U64 : CanonOp<[{
  gfx8+ 64-bit unsigned scalar compares (SOPC_CMP_64). Only EQ and
  LG (not equal) are defined in SOPInstructions.td; there are no
  ordered/strict 64-bit SOPC compares on any AMDGPU generation
  because the .td record `SOPC_CMP_64` is reserved for these two.
}]>;

def COP_S_CMP_LG_U64 : CanonOp;

def COP_S_CMP_EQ_I32 : CanonOp;

def COP_S_CMP_LG_I32 : CanonOp;

def COP_S_CMP_GT_I32 : CanonOp;

def COP_S_CMP_GE_I32 : CanonOp;

def COP_S_CMP_LT_I32 : CanonOp;

def COP_S_CMP_LE_I32 : CanonOp;

def COP_S_CMP_EQ_F32 : CanonOp;

def COP_S_CMP_LG_F32 : CanonOp;

def COP_S_CMP_GT_F32 : CanonOp;

def COP_S_CMP_GE_F32 : CanonOp;

def COP_S_CMP_LT_F32 : CanonOp;

def COP_S_CMP_LE_F32 : CanonOp;

def COP_S_CMP_NEQ_F32 : CanonOp;

def COP_S_CMP_NGT_F32 : CanonOp;

def COP_S_CMP_NGE_F32 : CanonOp;

def COP_S_CMP_NLT_F32 : CanonOp;

def COP_S_CMP_NLE_F32 : CanonOp;

def COP_S_CMP_NLG_F32 : CanonOp;

def COP_S_CMP_O_F32 : CanonOp;

def COP_S_CMP_U_F32 : CanonOp;

def COP_S_CMP_EQ_F16 : CanonOp;

def COP_S_CMP_LG_F16 : CanonOp;

def COP_S_CMP_GT_F16 : CanonOp;

def COP_S_CMP_GE_F16 : CanonOp;

def COP_S_CMP_LT_F16 : CanonOp;

def COP_S_CMP_LE_F16 : CanonOp;

def COP_S_CMP_NEQ_F16 : CanonOp;

def COP_S_CMP_NGT_F16 : CanonOp;

def COP_S_CMP_NGE_F16 : CanonOp;

def COP_S_CMP_NLT_F16 : CanonOp;

def COP_S_CMP_NLE_F16 : CanonOp;

def COP_S_CMP_NLG_F16 : CanonOp;

def COP_S_CMP_O_F16 : CanonOp;

def COP_S_CMP_U_F16 : CanonOp;

def COP_S_MOVK_I32 : CanonOp<[{
  -- SOPK --
}]>;

def COP_S_ADDK_I32 : CanonOp;

def COP_S_MULK_I32 : CanonOp;

def COP_S_CMPK_GE_I32 : CanonOp;

def COP_S_CMPK_GT_I32 : CanonOp;

def COP_S_CMPK_LE_I32 : CanonOp;

def COP_S_CMPK_LT_I32 : CanonOp;

def COP_S_CMPK_GE_U32 : CanonOp;

def COP_S_CMPK_GT_U32 : CanonOp;

def COP_S_CMPK_LE_U32 : CanonOp;

def COP_S_CMPK_LT_U32 : CanonOp;

def COP_S_CMPK_EQ_I32 : CanonOp;

def COP_S_CMPK_EQ_U32 : CanonOp;

def COP_S_CMPK_LG_I32 : CanonOp;

def COP_S_CMPK_LG_U32 : CanonOp;

def COP_S_GETREG_B32 : CanonOp;

def COP_S_SETREG_B32 : CanonOp;

def COP_S_SETREG_IMM32_B32 : CanonOp;

def COP_S_MOV_B32 : CanonOp<[{
  -- SOP1 --
}]>;

def COP_S_MOV_B64 : CanonOp;

def COP_S_NOT_B32 : CanonOp;

def COP_S_NOT_B64 : CanonOp;

def COP_S_BREV_B32 : CanonOp;

def COP_S_FF1_I32_B32 : CanonOp;

def COP_S_FF1_I32_B64 : CanonOp;

def COP_S_BCNT1_I32_B32 : CanonOp<[{
  s_bcnt1_i32_b{32,64}: scalar population count
  (SOPInstructions.td:269-274). Both lower to llvm.ctpop on the
  appropriately sized source; the B64 form truncates the ctpop result
  to i32 for the destination SGPR. Both write SCC = (D.u != 0),
  derived automatically by the raiser from Hr.SccResult
  (raiser.cpp:1202) since SOPInstructions.td wraps the defs in
  `let Defs = [SCC]`.
}]>;

def COP_S_BCNT1_I32_B64 : CanonOp;

def COP_S_FF0_I32_B32 : CanonOp<[{
  s_ff0_i32_b{32,64}: find first 0 bit (lowest position), returning
  -1 when the source is all-ones. SOPInstructions.td:278-279 (no
  LLVM ISel pattern is provided, so the instruction is only emitted
  by hand-written asm / inline-asm -- but the corpus contains it).
  Lowers to `cttz(~src, is_zero_poison=false)` with a `cmov` to -1
  on the all-ones input path, mirroring the V_FFBL_B32 / V_FFBH_U32
  shape (the AMDGPU instruction returns 0xFFFFFFFF in the no-bit
  case rather than the LLVM intrinsic's bitwidth-wide return).
}]>;

def COP_S_FF0_I32_B64 : CanonOp;

def COP_S_FLBIT_I32_B32 : CanonOp;

def COP_S_FLBIT_I32_B64 : CanonOp;

def COP_S_FLBIT_I32 : CanonOp<[{
  s_flbit_i32 / s_flbit_i32_i64: signed find-leading-bit-not-equal-to-
  sign-bit. Lowers to llvm.amdgcn.sffbh, the dedicated AMDGPU
  intrinsic that selects directly back to v_ffbh_i32_e32 (or its
  i64-split lowering for the 64-bit variant). See
  SOPInstructions.td:296-298 / VOP1Instructions.td:373.
}]>;

def COP_S_FLBIT_I32_I64 : CanonOp;

def COP_S_SEXT_I32_I8 : CanonOp;

def COP_S_SEXT_I32_I16 : CanonOp;

def COP_S_CVT_F16_F32 : CanonOp;

def COP_S_CVT_F32_F16 : CanonOp;

def COP_S_CVT_HI_F32_F16 : CanonOp;

def COP_S_CVT_F32_U32 : CanonOp;

def COP_S_CVT_F32_I32 : CanonOp;

def COP_S_CVT_U32_F32 : CanonOp;

def COP_S_CVT_I32_F32 : CanonOp;

def COP_S_CEIL_F32 : CanonOp<[{
  gfx11+ scalar F32-to-F32 integral rounding family. The AMD ISA manual
  defines each as a 32-bit SGPR F32 input and F32 output; S_TRUNC_F32 in
  particular stores the integer part using round-toward-zero semantics back
  in floating-point format, not as an integer conversion.
}]>;

def COP_S_FLOOR_F32 : CanonOp;

def COP_S_TRUNC_F32 : CanonOp;

def COP_S_RNDNE_F32 : CanonOp;

def COP_S_AND_SAVEEXEC_B32 : CanonOp;

def COP_S_OR_SAVEEXEC_B32 : CanonOp;

def COP_S_XOR_SAVEEXEC_B32 : CanonOp;

def COP_S_ANDN2_SAVEEXEC_B32 : CanonOp;

def COP_S_ORN2_SAVEEXEC_B32 : CanonOp;

def COP_S_GETPC_B64 : CanonOp;

def COP_S_SET_PC_I64 : CanonOp<[{
  SOP1 indirect set-PC. gfx1250 asm rename for `S_SETPC_B64`
  (SOPInstructions.td:323 declares `isBranch + isIndirectBranch`,
  line 2208 renames the asm string to `s_set_pc_i64`). The source
  SGPR pair holds an absolute 64-bit PC value. In our IR-on-LLVM
  setting we model three principled lowerings (see setpc_analysis.{hpp,
  cpp} for the static analysis that classifies each site):
  DirectA      -- statically resolvable intra-kernel branch (the
  source SGPR pair was produced by a local
  `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32`
  chain). Lowers to `br label %BB_target` since
  the target is a known intra-function label.
  IndirectB    -- subroutine return via an SGPR pair stashed at the
  call site (the canonical s[30:31] return-PC
  idiom). Lowers to a switch (via
  `emitEnumeratedDispatch` in handle-sop1.cpp)
  enumerating the resolved return targets and
  terminating in a trap default. The corresponding call-site
  `s_get_pc_i64 + s_add*` chains are rewritten by
  the raiser to write the plain i64 marker
  `resolvedReturnAddr` (the source-MC byte offset
  of the intended return BB) into the ret-pair
  (via a post-handler hook in raiser.cpp), so
  the switch cases compare ordinary integer markers. See
  `emitEnumeratedDispatch`'s rationale block for why the
  switch is lowered before AMDGPU codegen (LLVM's
  FixIrreducible pass only handles br-flavoured
  predecessors of an irreducible cycle header) and why an
  integer marker rather than
  `ptrtoint(blockaddress)` (AMDGPU ISel has no
  pattern to materialise a `BlockAddress` as an
  i64 register value).
  DispatchSet  -- multi-target dispatch via inter-block PC-chain
  dataflow: each predecessor block writes a
  different chain target into the same SGPR pair,
  then a join block consumes it through
  `s_set_pc_i64`. The dataflow in setpc_analysis
  enumerates the bounded set of targets reaching
  the use site through distinct CFG paths. Lowers
  to the same switch enumerated dispatch as
  IndirectB. Same chain-terminator hook as
  IndirectB writes the per-predecessor i64 marker
  (the callee's source-MC byte offset) on each
  contributing predecessor path.
  Sites the analysis cannot resolve (incomplete dataflow,
  unbounded fan-in past kMaxDispatchTargets, or pair killed by an
  unmodelled write before the use site) refuse loudly via
  RaiseFailure::unsupportedInstructionForm -- never silently emit a stub.
}]>;

def COP_S_SWAP_PC_I64 : CanonOp<[{
  SOP1 branch-and-link. gfx1250 asm rename for `S_SWAPPC_B64`
  (SOPInstructions.td:336 declares `isCall = 1`, line 2311 renames
  the asm string to `s_swap_pc_i64`). Operands:
  sdst = sX:X+1 receives the return PC (i.e. the absolute kernel
  offset of the instruction immediately following the
  swap, swap.offset + swap.size).
  ssrc = sY:Y+1 holds the absolute call target PC.
  PC <- ssrc; sdst <- (return-PC)  (atomically)

  Three principled raisings, mirroring S_SET_PC_I64:
  DirectA      -- call target ssrc was produced by a local
  `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32`
  chain that resolves intra-block. Lowering writes
  the return-address marker (the plain i64
  source-MC byte offset of swap.offset+swap.size)
  into sdst and emits `br label %BB_callee`.
  DispatchSet  -- call target reached via inter-block PC-chain
  dataflow (the tensilelite "activation function
  dispatcher" shape: each predecessor block
  computes a distinct callee target into the same
  pair via its own getpc+add chain, then a join
  block executes `s_swap_pc_i64`). Lowering writes
  the return-address marker into sdst as in
  DirectA, then emits a switch (via
  `emitEnumeratedDispatch` in handle-sop1.cpp) over the
  enumerated callee targets, terminating in a trap
  default. The chain-terminator hook in raiser.cpp
  rewrites ssrc to hold the callee's i64 marker
  (source-MC byte offset) on every contributing
  predecessor path. See `emitEnumeratedDispatch`'s
  rationale block for why the switch is lowered before
  AMDGPU codegen (FixIrreducible compatibility under
  irreducible CFGs -- the dominant shape this pattern
  produces) and why an integer marker rather than
  `ptrtoint(blockaddress)`
  (AMDGPU ISel cannot materialise a
  `BlockAddress` as an i64).
  Unresolvable -- call target cannot be statically enumerated
  (incomplete dataflow, fan-in past
  kMaxDispatchTargets, or runtime-derived value).
  Refuse loudly via RaiseFailure::unsupportedInstructionForm
  -- never emit a stub branch.

  The analysis never produces IndirectB for a swap_pc site (a
  swap_pc's source pair is the call target, not a return slot;
  IndirectB describes the return-side use of such a pair).

  Independent of the call-target classification, the analysis
  registers a synthetic chain-terminator at the swap site itself
  (key = swap.offset, value = {sdst-low-reg, swap.offset+swap.size})
  so any downstream IndirectB `s_set_pc_i64` reading sdst
  enumerates the swap's return offset as one of its switch targets.
}]>;

def COP_S_ADD_PC_I64 : CanonOp<[{
  SOP1 gfx1250 PC-relative branch; signed i64 offset, immediate form only.
}]>;

def COP_S_ABS_I32 : CanonOp;

def COP_S_SET_VGPR_MSB : CanonOp;

def COP_S_BITSET0_B32 : CanonOp<[{
  Read-modify-write bit set/clear on an SGPR. Tied src keeps the
  un-touched bits of the destination register alive across the op.
  B64 variants index into 64 bits (bit index is still an SReg_32).
}]>;

def COP_S_BITSET1_B32 : CanonOp;

def COP_S_BITSET0_B64 : CanonOp;

def COP_S_BITSET1_B64 : CanonOp;

def COP_S_BITCMP0_B32 : CanonOp<[{
  SOPC bit-test family (SOPInstructions.td:1411-1414; gfx6+ on every
  AMDGPU generation, so fully cross-target viable).  Tests a single
  bit of src0 selected by src1 (src1's lower 5 bits for _B32, lower
  6 bits for _B64) and writes the result into SCC:
  S_BITCMP0_B32  SCC = (src0 & (1u << (src1 & 0x1F))) == 0
  S_BITCMP1_B32  SCC = (src0 & (1u << (src1 & 0x1F))) != 0
  S_BITCMP0_B64  SCC = (src0 & (1ull << (src1 & 0x3F))) == 0
  S_BITCMP1_B64  SCC = (src0 & (1ull << (src1 & 0x3F))) != 0
  For _B64, src0 is a 64-bit SGPR pair; src1 remains a 32-bit SReg
  whose high 26/27 bits are ignored by the hardware (we apply the
  mask in IR to preserve that invariant exactly instead of relying
  on undef-width behaviour).  The handler lives in handle-sopc.cpp
  next to the SOPC compares it mirrors.
}]>;

def COP_S_BITCMP1_B32 : CanonOp;

def COP_S_BITCMP0_B64 : CanonOp;

def COP_S_BITCMP1_B64 : CanonOp;

def COP_S_CMOV_B32 : CanonOp<[{
  Conditional move on SCC. `if (SCC) sdst = src; else sdst stays
  unchanged.` The dst-on-SCC=0 read-modify is NOT modeled by LLVM
  as a tied sdst_in operand on the MCInst (SOP1_32/SOP1_64 just
  declares `(outs sdst), (ins src0)`), so the handler must
  explicitly read the prior dst value via
  `ctx.Regs.readReg{32,64}(op.dst())`. SCC is read but not
  written.
}]>;

def COP_S_CMOV_B64 : CanonOp;

def COP_S_ADD_U32 : CanonOp<[{
  -- SOP2 --
  `S_ADD_U64` used to live here as a second CanonicalOp alongside the
  gfx12-renamed `S_ADD_NC_U64` below, created by the same commit
  that first added the `s_add_u64` opcode-map row.  That row was later
  retargeted at `CanonicalOp::S_ADD_NC_U64` (matching S_SUB_U64 ->
  S_SUB_NC_U64), but the old `CanonicalOp::S_ADD_U64` enum entry + a
  defensive `||` branch in `handle-sop2.cpp` + a stray duplicate row
  were left behind.  Back when the table was flattened at runtime the
  duplicate was resolved keeps-first, so lifts silently routed through
  the stale enum value and the handler's `||` disjunct masked the
  difference.  hotswap-tblgen now rejects a duplicated `Canon` row
  outright, so the same mistake cannot ship again.
  `CanonicalOp::S_ADD_NC_U64` below is the ONLY CanonicalOp for LLVM's
  `S_ADD_U64` pseudo.
}]>;

def COP_S_ADDC_U32 : CanonOp;

def COP_S_SUB_U32 : CanonOp;

def COP_S_SUBB_U32 : CanonOp;

def COP_S_AND_B32 : CanonOp;

def COP_S_AND_B64 : CanonOp;

def COP_S_OR_B32 : CanonOp;

def COP_S_OR_B64 : CanonOp;

def COP_S_XOR_B32 : CanonOp;

def COP_S_XOR_B64 : CanonOp;

def COP_S_ANDN2_B32 : CanonOp;

def COP_S_ANDN2_B64 : CanonOp;

def COP_S_ORN2_B32 : CanonOp;

def COP_S_ORN2_B64 : CanonOp;

def COP_S_NAND_B32 : CanonOp<[{
  SOP2 negated bitops (gfx7+). SOPInstructions.td:789-803 -- each
  computes `dst = ~(src0 OP src1)` and sets SCC = (result != 0). These
  are produced heavily by triton/tensilelite when constant-folding
  bitfield masks (e.g. `s_nand_b32 sX, sY, 0xffff` to clear the low
  16 bits). All can target EXEC, so they must be marked
  routesExecThroughStoreExec.
}]>;

def COP_S_NAND_B64 : CanonOp;

def COP_S_NOR_B32 : CanonOp;

def COP_S_NOR_B64 : CanonOp;

def COP_S_XNOR_B32 : CanonOp;

def COP_S_XNOR_B64 : CanonOp;

def COP_S_ABSDIFF_I32 : CanonOp<[{
  SOP2 absolute-difference (gfx7+). SOPInstructions.td:886-888 --
  `dst = |src0 - src1|` on signed i32, SCC = (result != 0). Lower
  through llvm.abs.i32 with is_int_min_poison=false: hardware wraps
  for INT_MIN (the only value whose negation equals itself), so we
  mustn't poison there. Heavily used by tensilelite for stride math.
}]>;

def COP_S_LSHL_B32 : CanonOp;

def COP_S_LSHL_B64 : CanonOp;

def COP_S_LSHR_B32 : CanonOp;

def COP_S_LSHR_B64 : CanonOp;

def COP_S_ASHR_I32 : CanonOp;

def COP_S_ASHR_I64 : CanonOp;

def COP_S_MUL_I32 : CanonOp;

def COP_S_MUL_HI_U32 : CanonOp;

def COP_S_MUL_HI_I32 : CanonOp;

def COP_S_MUL_U64 : CanonOp;

def COP_S_MUL_F32 : CanonOp;

def COP_S_ADD_F32 : CanonOp;

def COP_S_SUB_F32 : CanonOp;

def COP_S_FMAC_F32 : CanonOp<[{
  gfx11+ scalar fused multiply-accumulate. SOP2 encodes only two explicit
  sources; OPF_DACCUM ties the old destination value as the third operand:
  sdst.f32 = fma(ssrc0.f32, ssrc1.f32, old sdst.f32)
}]>;

def COP_S_FMAAK_F32 : CanonOp<[{
  Scalar literal fused multiply-add forms. OPF_IMPLIED_LITERAL supplies the
  third f32 operand from the instruction literal slot:
  s_fmaak_f32: sdst.f32 = fma(ssrc0.f32, ssrc1.f32, literal.f32)
  s_fmamk_f32: sdst.f32 = fma(ssrc0.f32, literal.f32, ssrc1.f32)
}]>;

def COP_S_FMAMK_F32 : CanonOp;

def COP_S_MAX_NUM_F32 : CanonOp<[{
  Scalar IEEE-754-2019 maximumNumber/minimumNumber. LLVM's canonical pseudo
  is `S_{MAX,MIN}_F32`; gfx12+ manuals name the real mnemonics
  `s_{max,min}_num_f32` and keep `s_{max,min}_f32` as compatibility aliases.
  Semantics favor a numeric operand over NaN (including signaling NaN after
  raising invalid) and order signed zeros (+0 > -0 for max, -0 < +0 for min),
  matching LLVM's `maximumnum` / `minimumnum` intrinsic contract without
  fast-math flags.
}]>;

def COP_S_MIN_NUM_F32 : CanonOp;

def COP_S_MAXIMUM_F16 : CanonOp<[{
  Scalar IEEE-754-2019 NaN-propagating f16 maximum/minimum (gfx12+). Low 16
  bits of the 32-bit SGPRs hold the f16 operand. Map to `llvm.maximum.f16` /
  `llvm.minimum.f16`, distinct from the non-propagating NUM family above.
}]>;

def COP_S_MINIMUM_F16 : CanonOp;

def COP_S_MAXIMUM_F32 : CanonOp<[{
  Scalar IEEE-754-2019 NaN-propagating f32 maximum/minimum (gfx12+). Map to
  `llvm.maximum.f32` / `llvm.minimum.f32`, distinct from the NUM family.
}]>;

def COP_S_MINIMUM_F32 : CanonOp;

def COP_S_BFE_U32 : CanonOp;

def COP_S_BFE_I32 : CanonOp;

def COP_S_BFE_I64 : CanonOp;

def COP_S_BFM_B32 : CanonOp;

def COP_S_BFM_B64 : CanonOp;

def COP_S_CSELECT_B32 : CanonOp;

def COP_S_CSELECT_B64 : CanonOp;

def COP_S_MIN_I32 : CanonOp;

def COP_S_MIN_U32 : CanonOp;

def COP_S_MAX_I32 : CanonOp;

def COP_S_MAX_U32 : CanonOp;

def COP_S_PACK_LL_B32_B16 : CanonOp;

def COP_S_PACK_LH_B32_B16 : CanonOp;

def COP_S_LSHL1_ADD_U32 : CanonOp;

def COP_S_LSHL2_ADD_U32 : CanonOp;

def COP_S_LSHL3_ADD_U32 : CanonOp;

def COP_S_LSHL4_ADD_U32 : CanonOp;

def COP_S_ADD_NC_U64 : CanonOp;

def COP_S_SUB_NC_U64 : CanonOp;

def COP_V_MOV_B32 : CanonOp<[{
  -- VOP1 --
}]>;

def COP_V_MOV_B64 : CanonOp;

def COP_V_MOV_B16 : CanonOp;

def COP_V_NOP : CanonOp;

def COP_V_NOT_B32 : CanonOp;

def COP_V_BFREV_B32 : CanonOp;

def COP_V_SWAP_B32 : CanonOp;

def COP_V_MOVRELD_B32 : CanonOp<[{
  Register-relative moves. The relative index is `M0` (uniform across
  lanes), so these carry no cross-lane component and are wave-size-
  agnostic. V_MOVRELD writes VGPR[base(vdst)+M0]=vsrc (M0-relative dst,
  tied vdst_in); V_MOVRELS reads vdst=VGPR[base(vsrc)+M0] (relative
  src); V_MOVRELSD is both relative. See handle-valu-small-ops.cpp.
}]>;

def COP_V_MOVRELS_B32 : CanonOp;

def COP_V_MOVRELSD_B32 : CanonOp;

def COP_V_CVT_F32_I32 : CanonOp;

def COP_V_CVT_F32_U32 : CanonOp;

def COP_V_CVT_I32_F32 : CanonOp;

def COP_V_CVT_U32_F32 : CanonOp;

def COP_V_CVT_U32_U16 : CanonOp;

def COP_V_CVT_F16_F32 : CanonOp;

def COP_V_CVT_F32_F16 : CanonOp;

def COP_V_CVT_F32_BF16 : CanonOp;

def COP_V_CVT_F32_UBYTE0 : CanonOp;

def COP_V_CVT_F32_UBYTE1 : CanonOp;

def COP_V_CVT_F32_UBYTE2 : CanonOp;

def COP_V_CVT_F32_UBYTE3 : CanonOp;

def COP_V_CVT_F32_F64 : CanonOp;

def COP_V_CVT_F64_F32 : CanonOp;

def COP_V_CVT_F64_U32 : CanonOp;

def COP_V_CVT_F64_I32 : CanonOp;

def COP_V_CVT_U32_F64 : CanonOp;

def COP_V_CVT_I32_F64 : CanonOp;

def COP_V_RCP_IFLAG_F32 : CanonOp;

def COP_V_RCP_F32 : CanonOp;

def COP_V_RSQ_F32 : CanonOp;

def COP_V_SQRT_F32 : CanonOp;

def COP_V_EXP_F32 : CanonOp;

def COP_V_LOG_F32 : CanonOp;

def COP_V_SIN_F32 : CanonOp<[{
  VOP1/VOP3 F32 trigonometric TRANS ops. The ISA defines these as
  sin/cos(src * 2*pi), not generic radian-domain sin/cos; lower through
  the AMDGPU intrinsics that select the hardware TRANS instructions.
}]>;

def COP_V_COS_F32 : CanonOp;

def COP_V_FREXP_EXP_I32_F32 : CanonOp;

def COP_V_FREXP_EXP_I32_F64 : CanonOp;

def COP_V_TANH_F32 : CanonOp<[{
  Targets with native tanh support lower this through `llvm.amdgcn.tanh.*`;
  other targets use OCML when a matching OCML entry point exists.
}]>;

def COP_V_COS_BF16 : CanonOp<[{
  bf16 single-source transcendentals; lifted through f32 intrinsics
  (v_tanh_bf16 through __ocml_tanh_f32).
}]>;

def COP_V_EXP_BF16 : CanonOp;

def COP_V_LOG_BF16 : CanonOp;

def COP_V_RCP_BF16 : CanonOp;

def COP_V_RSQ_BF16 : CanonOp;

def COP_V_SIN_BF16 : CanonOp;

def COP_V_SQRT_BF16 : CanonOp;

def COP_V_TANH_BF16 : CanonOp;

def COP_V_S_EXP_F32 : CanonOp<[{
  gfx12+ VOP3 pseudo-scalar f32 transcendentals: scalar input and scalar
  output variants of the corresponding VOP1 special-function instructions.
  The default clamp=0/omod=0 forms lower through AMDGPU hardware intrinsics;
  non-default output modifiers are refused until modeled exactly.
}]>;

def COP_V_S_LOG_F32 : CanonOp;

def COP_V_S_RCP_F32 : CanonOp;

def COP_V_S_RSQ_F32 : CanonOp;

def COP_V_S_SQRT_F32 : CanonOp;

def COP_V_LDEXP_F32 : CanonOp;

def COP_V_FLOOR_F32 : CanonOp;

def COP_V_CEIL_F32 : CanonOp;

def COP_V_TRUNC_F32 : CanonOp;

def COP_V_RNDNE_F32 : CanonOp;

def COP_V_FRACT_F32 : CanonOp;

def COP_V_FLOOR_F64 : CanonOp;

def COP_V_CEIL_F64 : CanonOp;

def COP_V_TRUNC_F64 : CanonOp;

def COP_V_RNDNE_F64 : CanonOp;

def COP_V_FRACT_F64 : CanonOp;

def COP_V_FREXP_MANT_F32 : CanonOp;

def COP_V_FREXP_MANT_F64 : CanonOp;

def COP_V_READFIRSTLANE_B32 : CanonOp;

def COP_V_CVT_PK_F32_FP8 : CanonOp<[{
  VOP1 packed FP8/BF8 -> 2x F32 expansion (VOP1Instructions.td:652-
  653, profile VOPProfileCVT_PK_F32_F8). Reads 16 bits of the i32
  src -- the low half (bytes 0,1) when op_sel:[0,*] / SDWA WORD_0,
  the high half (bytes 2,3) when op_sel:[1,*] / SDWA WORD_1 -- and
  expands the two FP8/BF8 lanes into a v2f32 written to a VGPR
  pair starting at vdst. FP8 is the OCP E4M3FN format; BF8 is the
  OCP E5M2 format. Lowering selects the matching
  `llvm.amdgcn.cvt.pk.f32.{fp8,bf8}(i32 src, i1 word_sel)`
  intrinsic and bitcasts the v2f32 result to i64 before
  writeReg64. The op_sel-based word selector is parsed from the
  disassembly text exactly as in V_ADD_NC_U16 / V_FMA_MIX_F32 (no
  first-class "modifier" channel exists in our OperandView yet);
  unparseable / out-of-range selectors fall through to word_sel=0
  -- never silently corrupted, the parser invariant is the same as
  for the other op_sel handlers. The reverse direction
  (V_CVT_PK_FP8_F32 / V_CVT_PK_BF8_F32) lives in the VOP3 block
  below; this is the read-side companion.
}]>;

def COP_V_CVT_PK_F32_BF8 : CanonOp;

def COP_V_CVT_F32_FP8 : CanonOp<[{
  VOP1 single-lane FP8/BF8 -> F32 expansion (VOP1Instructions.td:650-
  651, profile VOPProfileCVT_F32_F8). Reads ONE 8-bit lane of the
  i32 src -- selected by SDWA src0_sel / e64 op_sel byte_sel -- and
  produces an f32. The SDWA encoding can pick any of the four bytes
  (0..3); the e64 encoding's default (no op_sel printed) is byte 0
  and is the only shape the gfx1250 corpus emits today (the LLVM
  isel pattern in VOP1Instructions.td:670-680 maps non-zero
  byte_sel through the SDWA pseudo, which we have not yet wired --
  adding it would only widen this handler, not change its shape).
  Lowering selects `llvm.amdgcn.cvt.f32.{fp8,bf8}(i32 src, i32
  byte_sel)` and writeReg32 the result. SDWA / op_sel-bearing
  encodings refuse loudly via RaiseFailure::unsupportedInstructionForm so a
  future corpus drift surfaces immediately rather than silently
  collapsing to byte 0.
}]>;

def COP_V_CVT_F32_BF8 : CanonOp;

def COP_V_CVT_SCALE_PK8_BF16_FP4 : CanonOp<[{
  VOP3 scaled packed-8 FP4 -> BF16 conversion (gfx1250 only,
  VOP3Instructions.td:1788; LLVM opcode V_CVT_SCALE_PK8_BF16_FP4_e64,
  real form `..._gfx1250`).  Reads 1 VGPR of packed 8xFP4 (4 bits
  each, 32 bits total) plus an E8M0-encoded scale value and a
  byte-granularity `scale_sel` immediate, and writes an 8xBF16
  result across 4 consecutive VGPRs.  Lowers to
  `<8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 src,
  i32 scale,
  i32 immarg sel)`
  which is declared inside the gfx1250-only block of
  IntrinsicsAMDGPU.td (AMDGPUCvtScaleIntrinsic w/ isGFX125xOnly) --
  the handler therefore gates same-target lift on
  `ctx.targetIsa.hasTensorOps` (FeatureGFX1250Insts) and refuses
  cross-target lift to gfx942 loudly (no MX-FP4 scaling unit on
  CDNA3; a manual per-nibble dequant expansion would be a separate
  design).

  Sibling variants (V_CVT_SCALE_PK8_{F16,F32}_{FP4,FP8,BF8} /
  V_CVT_SCALE_PK8_BF16_{FP8,BF8}) share the same operand shape and
  the same same-target-only constraint.  Only the BF16_FP4 form is
  exercised by the current kerneldex corpus (scope_discovery
  `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1`); adding a sibling
  is a two-line change (new CanonicalOp + new entry in the handler's
  variant-classifier table) and intentionally deferred until a
  corpus kernel exercises it.
}]>;

def COP_V_FFBH_U32 : CanonOp<[{
  VOP1 find-first-bit family (gfx7+, VOP1Instructions.td:371-373).
  V_FFBH_U32  -> AMDGPUffbh_u32 = ctlz_zero_undef but returns -1 on
  input 0; lower with llvm.ctlz(x, false) -- LLVM
  returns the bitwidth (32) for input 0, so we cmov
  to -1 explicitly to match hardware.
  V_FFBL_B32  -> AMDGPUffbl_b32 = cttz_zero_undef but returns -1 on
  input 0; same pattern with llvm.cttz.
  V_FFBH_I32  -> AMDGPUffbh_i32 = position of highest non-sign bit;
  returns -1 for input 0 or -1 (uniform sign). Lower
  via the dedicated llvm.amdgcn.sffbh intrinsic which
  selects directly back to v_ffbh_i32_e32.
}]>;

def COP_V_FFBL_B32 : CanonOp;

def COP_V_FFBH_I32 : CanonOp;

def COP_V_PRNG_B32 : CanonOp;

def COP_V_ADD_F32 : CanonOp<[{
  -- VOP2 / VOP3 --
}]>;

def COP_V_SUB_F32 : CanonOp;

def COP_V_SUBREV_F32 : CanonOp;

def COP_V_MUL_F32 : CanonOp;

def COP_V_MUL_LEGACY_F32 : CanonOp;

def COP_V_FMAC_F32 : CanonOp;

def COP_V_FMA_F32 : CanonOp;

def COP_V_FMAMK_F32 : CanonOp;

def COP_V_FMAAK_F32 : CanonOp;

def COP_V_ADD_NC_U32 : CanonOp;

def COP_V_BCNT_U32_B32 : CanonOp;

def COP_V_SUB_NC_U32 : CanonOp;

def COP_V_SUBREV_NC_U32 : CanonOp;

def COP_V_ADD_CO_U32 : CanonOp;

def COP_V_ADD_CO_CI_U32 : CanonOp;

def COP_V_SUB_CO_U32 : CanonOp;

def COP_V_SUBREV_CO_U32 : CanonOp;

def COP_V_SUB_CO_CI_U32 : CanonOp;

def COP_V_SUBREV_CO_CI_U32 : CanonOp;

def COP_V_AND_B32 : CanonOp;

def COP_V_OR_B32 : CanonOp;

def COP_V_XOR_B32 : CanonOp;

def COP_V_XNOR_B32 : CanonOp;

def COP_V_LSHLREV_B32 : CanonOp;

def COP_V_LSHRREV_B32 : CanonOp;

def COP_V_ASHRREV_I32 : CanonOp;

def COP_V_CNDMASK_B32 : CanonOp;

def COP_V_CNDMASK_B16 : CanonOp<[{
  gfx11+ true16 16-bit conditional select (VOP3-only). op_sel selects the
  src0/src1/dst halves; the wave-mask condition is read exactly as for
  V_CNDMASK_B32 via raiseCndmaskWaveCondition().
}]>;

def COP_V_AND_B16 : CanonOp<[{
  gfx11+ true16 16-bit bitwise ops (VOP3 true16 half-select on
  src0/src1/dst). V_NOT_B16 is single-source.
}]>;

def COP_V_OR_B16 : CanonOp;

def COP_V_XOR_B16 : CanonOp;

def COP_V_NOT_B16 : CanonOp;

def COP_V_MUL_LO_U32 : CanonOp;

def COP_V_MUL_HI_U32 : CanonOp;

def COP_V_MUL_HI_I32 : CanonOp;

def COP_V_MUL_I32_I24 : CanonOp;

def COP_V_MUL_U32_U24 : CanonOp;

def COP_V_MUL_HI_U32_U24 : CanonOp;

def COP_V_MUL_HI_I32_I24 : CanonOp;

def COP_V_MAD_I32_I24 : CanonOp;

def COP_V_MAD_U32_U24 : CanonOp;

def COP_V_MAD_U32 : CanonOp;

def COP_V_ADD3_U32 : CanonOp;

def COP_V_LSHL_ADD_U32 : CanonOp;

def COP_V_ADD_LSHL_U32 : CanonOp;

def COP_V_LSHL_OR_B32 : CanonOp;

def COP_V_AND_OR_B32 : CanonOp;

def COP_V_OR3_B32 : CanonOp;

def COP_V_XAD_U32 : CanonOp;

def COP_V_ALIGNBIT_B32 : CanonOp<[{
  VOP3 funnel-shift right: dst = ((src0:src1) >> src2[4:0])[31:0].
  .td uses the SDAG `fshr` node directly (VOP3Instructions.td:222),
  which maps to `llvm.fshr.i32` in IR. src2 is masked to 5 bits
  by hardware before the shift.
}]>;

def COP_V_XOR3_B32 : CanonOp<[{
  VOP3 ternary xor -- gfx10+ only (VOP3Instructions.td:1348),
  .td has no SDAG `umin3`-style node, the iselect pattern at
  line 1350 directly matches `(xor (xor a, b), c)`. Lift is the
  same shape as V_OR3_B32 above.
}]>;

def COP_V_ADD_NC_U16 : CanonOp<[{
  VOP3 16-bit no-carry add/sub -- true16 VOP3Inst_t16 forms. Signed
  variants use LLVM's older V_{ADD,SUB}_I16 pseudos; gfx10+ real
  mnemonics spell those no-carry semantics as v_{add,sub}_nc_i16.
  Op_sel routes 16-bit halves of src0/src1 (lo or hi) and
  selects which half of the 32-bit dst register receives the
  result; the unselected half of dst is preserved per the
  RDNA3+ ISA. The handler must read the prior dst value when
  dst op_sel is set so the preserved half survives the
  read-modify-write.
}]>;

def COP_V_SUB_NC_U16 : CanonOp;

def COP_V_ADD_NC_I16 : CanonOp;

def COP_V_SUB_NC_I16 : CanonOp;

def COP_V_MAD_U16 : CanonOp;

def COP_V_ADD_MIN_U32 : CanonOp<[{
  gfx1250 VOP3 add-then-min/max: (s/u)(min/max)((s/u)addsat(src0, src1),
  src2).
}]>;

def COP_V_ADD_MAX_U32 : CanonOp;

def COP_V_ADD_MIN_I32 : CanonOp;

def COP_V_ADD_MAX_I32 : CanonOp;

def COP_V_BFE_U32 : CanonOp;

def COP_V_BFE_I32 : CanonOp;

def COP_V_BFI_B32 : CanonOp;

def COP_V_PERM_B32 : CanonOp;

def COP_V_MBCNT_LO_U32_B32 : CanonOp;

def COP_V_MBCNT_HI_U32_B32 : CanonOp;

def COP_V_READLANE_B32 : CanonOp;

def COP_V_WRITELANE_B32 : CanonOp;

def COP_V_MED3_NUM_F32 : CanonOp<[{
  VOP3 ternary maximumNumber/minimumNumber
}]>;

def COP_V_MAX3_NUM_F32 : CanonOp;

def COP_V_MIN3_NUM_F32 : CanonOp;

def COP_V_MINMAX_NUM_F32 : CanonOp<[{
  VOP3 IEEE-2019 ternary clamp `minnum(maxnum(s0, s1), s2)`.
  gfx12 renamed gfx11's V_MINMAX_F32 (.td:1485, opcode 0x25f)
  to V_MINMAX_NUM_F32 (.td:1696, opcode 0x268) when the .NUM
  suffix was introduced to disambiguate from the IEEE-754
  2019 V_MINIMUMMAXIMUM_F32 (NaN-propagating, opcode 0x26c).
  hotswap-tblgen collapses both real names onto this CanonicalOp.
}]>;

def COP_V_MAXMIN_NUM_F32 : CanonOp<[{
  VOP3 ternary clamp `.NUM` dual:
  V_MAXMIN_NUM_F32: minnum(maxnum(s0, s1), s2).
}]>;

def COP_V_MAX3_U32 : CanonOp<[{
  VOP3 integer 3-way max/min/median and ternary min/max.
}]>;

def COP_V_MIN3_U32 : CanonOp;

def COP_V_MAX3_I32 : CanonOp;

def COP_V_MIN3_I32 : CanonOp;

def COP_V_MAX3_I16 : CanonOp;

def COP_V_MED3_I32 : CanonOp<[{
  VOP3 signed-integer median-of-three. Hardware semantic
  (VOP3Instructions.td:1796 via AMDGPUsmed3 SDAG node):
  med3_i32(a, b, c) = smax(smin(a, b), smin(smax(a, b), c))
  i.e. the middle of three signed i32 values. We lift it as a
  pair of `llvm.smin`/`llvm.smax` intrinsics (matching the
  `handle-vopd.cpp` style that already uses these intrinsics for
  VOPD smin/smax/umin/umax pairs). The backend's
  `AMDGPUISelDAGToDAG`/`AMDGPUISelLowering` pattern-matches the
  `smax(smin(...), smin(smax(...), ...))` shape back to
  V_MED3_I32, so the round-trip is structure-preserving and the
  generated assembly recovers the original instruction without
  codegen quality loss.
}]>;

def COP_V_MINMAX_I32 : CanonOp<[{
  Ternary nested min/max family.
}]>;

def COP_V_MAXMIN_I32 : CanonOp;

def COP_V_MINMAX_U32 : CanonOp;

def COP_V_MAXMIN_U32 : CanonOp;

def COP_V_MAX_NUM_F32 : CanonOp<[{
  IEEE-754-2019 maximumNumber/minimumNumber: numeric operand preferred over
  NaN.
}]>;

def COP_V_MIN_NUM_F32 : CanonOp;

def COP_V_MAXIMUM_F32 : CanonOp<[{
  IEEE-754 2019 maximum/minimum: propagate NaN (distinct from maxnum/minnum).
}]>;

def COP_V_MINIMUM_F32 : CanonOp;

def COP_V_MAXIMUM3_F32 : CanonOp<[{
  IEEE-754 2019 ternary maximum/minimum: NaN-propagating 3-source
  reduction. gfx11+ (`HasMinimum3Maximum3F32`; VOP3 opcodes 0x22d/0x22e).
}]>;

def COP_V_MINIMUM3_F32 : CanonOp;

def COP_V_MAXIMUMMINIMUM_F32 : CanonOp<[{
  IEEE-754 2019 ternary clamp pair, distinct from V_MINMAX_NUM_F32's
  NaN-pruning `.NUM` semantics:
  V_MAXIMUMMINIMUM_F32: minimum(maximum(s0, s1), s2)
  V_MINIMUMMAXIMUM_F32: maximum(minimum(s0, s1), s2)
  Both support source modifiers through OpResolver::srcF. Non-default VOP3
  output modifiers (clamp / omod) are refused until modeled exactly.
}]>;

def COP_V_MINIMUMMAXIMUM_F32 : CanonOp;

def COP_V_DIV_FIXUP_F32 : CanonOp;

def COP_V_DIV_FMAS_F32 : CanonOp;

def COP_V_DIV_SCALE_F32 : CanonOp;

def COP_V_DIV_FIXUP_F64 : CanonOp;

def COP_V_DIV_FMAS_F64 : CanonOp;

def COP_V_DIV_SCALE_F64 : CanonOp;

def COP_V_DIV_FIXUP_F16 : CanonOp<[{
  f16 division fixup (VOP3, true16 half-select like V_FMA_F16). Lifts to
  llvm.amdgcn.div.fixup on f16.
}]>;

def COP_V_FMA_MIX_F32 : CanonOp<[{
  Mixed-precision FMA, VOP3P (VOP3PInstructions.td:109). Both
  variants take three sources and reduce to
  fma(cvt_f32(src0_part), cvt_f32(src1_part), cvt_f32(src2_part))
  where `*_part` is selected by the per-source op_sel / op_sel_hi
  modifiers:
  op_sel_hi[i]==0  -> source i is the full f32 VGPR
  op_sel_hi[i]==1  -> source i is the 16-bit lo (op_sel[i]==0) or
  hi (op_sel[i]==1) half, interpreted as the
  mnemonic's narrow type
  V_FMA_MIX_F32       : narrow type = f16  (all gfx targets)
  V_FMA_MIX_F32_BF16  : narrow type = bf16 (gfx9.5+/gfx1250; the bf16
  narrow half -> f32 extension is
  cross-target-universal via `fpext bfloat to
  float`, so no refusal is needed on gfx942)
  Both CanonicalOps share the op_sel/op_sel_hi parser and write-back shape
  in handle-valu-vop3p.cpp; only the narrow element type differs.
}]>;

def COP_V_FMA_MIX_F32_BF16 : CanonOp;

def COP_V_FMA_MIXLO_F16 : CanonOp<[{
  VOP3P narrow-half destination mixed FMA. Compute
  fptrunc_narrow(fma(cvt_f32(src0_part),
  cvt_f32(src1_part),
  cvt_f32(src2_part)))
  and write the rounded narrow result into either the low or high 16 bits
  of `vdst`. The other half is the tied-output input (`vdst_in`) and must
  be preserved explicitly in IR; this is not a plain 32-bit f32 write.
  V_FMA_MIX{LO,HI}_F16  : narrow result/source type = f16
  V_FMA_MIX{LO,HI}_BF16 : narrow result/source type = bf16 (gfx1250)
}]>;

def COP_V_FMA_MIXHI_F16 : CanonOp;

def COP_V_FMA_MIXLO_BF16 : CanonOp;

def COP_V_FMA_MIXHI_BF16 : CanonOp;

def COP_V_ADD_F16 : CanonOp;

def COP_V_MUL_F16 : CanonOp;

def COP_V_SUB_F16 : CanonOp;

def COP_V_SUBREV_F16 : CanonOp;

def COP_V_MAC_F16 : CanonOp;

def COP_V_FMAC_F16 : CanonOp;

def COP_V_MADMK_F16 : CanonOp<[{
  VOP2 F16 multiply-add-with-literal pseudos (mirror of
  V_FMAMK_F32 / V_FMAAK_F32 for the f16 lane). Defined in
  VOP2Instructions.td:1206-1210 -- both take a 16-bit constant K
  alongside two F16 sources and lower to llvm.fma.f16:
  v_madmk_f16 dst, src0, K, src2 -> dst = src0 * K + src2
  v_madak_f16 dst, src0, src1, K -> dst = src0 * src1 + K
  Note: hardware uses the legacy "mad" name, but the lowered
  semantics are fused-multiply-add (no rounding of the intermediate
  product), matching the F32 FMAMK/FMAAK convention.
}]>;

def COP_V_MADAK_F16 : CanonOp;

def COP_V_FMA_F16 : CanonOp<[{
  VOP3 F16 fused multiply-add: dst = fma(src0, src1, src2). The gfx9+
  pseudos (V_FMA_F16_gfx9_e64 and its t16/fake16 siblings) expose
  per-source op_sel for half selection within the 32-bit VGPR, plus the
  usual VOP3 neg/abs source modifiers and a destination op_sel for
  half-write placement. Lowered to llvm.fma.f16; semantics match
  VOP3Instructions.td:453,2312,2601 (any_fma SDNode).
}]>;

def COP_V_MAX_NUM_F16 : CanonOp<[{
  VOP2 maximumNumber/minimumNumber
}]>;

def COP_V_MIN_NUM_F16 : CanonOp;

def COP_V_MINMAX_NUM_F16 : CanonOp<[{
  F16 ternary clamp `.NUM` pair. Like the IEEE f16 forms below, these
  preserve the unselected destination half and honor source/dst op_sel.
}]>;

def COP_V_MAXMIN_NUM_F16 : CanonOp;

def COP_V_MAXIMUM_F16 : CanonOp<[{
  IEEE-754 2019 f16 maximum/minimum: propagate NaN (`llvm.maximum` /
  `llvm.minimum`)
}]>;

def COP_V_MINIMUM_F16 : CanonOp;

def COP_V_MAXIMUM3_F16 : CanonOp<[{
  IEEE-754 2019 f16 ternary reductions and clamp pair. These handlers
  honor source/destination op_sel and preserve the unselected destination
  half.
}]>;

def COP_V_MINIMUM3_F16 : CanonOp;

def COP_V_MAXIMUMMINIMUM_F16 : CanonOp;

def COP_V_MINIMUMMAXIMUM_F16 : CanonOp;

def COP_V_LDEXP_F16 : CanonOp;

def COP_V_FLOOR_F16 : CanonOp;

def COP_V_TANH_F16 : CanonOp;

def COP_V_CEIL_F16 : CanonOp<[{
  f16 unary rounding + reciprocal (true16 op_sel half-select like
  V_TANH_F16). ceil/trunc/rndne lower to llvm.{ceil,trunc,roundeven}.f16; rcp
  to llvm.amdgcn.rcp.f16 (native v_rcp_f16 on the target).
}]>;

def COP_V_TRUNC_F16 : CanonOp;

def COP_V_RNDNE_F16 : CanonOp;

def COP_V_RCP_F16 : CanonOp;

def COP_V_CVT_F16_U16 : CanonOp;

def COP_V_CVT_F16_I16 : CanonOp;

def COP_V_CVT_U16_F16 : CanonOp;

def COP_V_CVT_I16_F16 : CanonOp;

def COP_V_ASHRREV_I16 : CanonOp;

def COP_V_LSHRREV_B16 : CanonOp;

def COP_V_LSHLREV_B16 : CanonOp;

def COP_V_MAX_U16 : CanonOp;

def COP_V_MIN_U16 : CanonOp;

def COP_V_MAX_I16 : CanonOp;

def COP_V_MIN_I16 : CanonOp;

def COP_V_ADD_U16 : CanonOp<[{
  16-bit integer arith (gfx8+, VOP2Instructions.td). Plain i16
  add/sub/subrev with wrapping overflow (no carry-out -- distinct
  from the rarely-used v_add_co_u16). v_mul_lo_u16 returns the low
  16 bits of the multiply, naturally produced by `mul i16`.
}]>;

def COP_V_SUB_U16 : CanonOp;

def COP_V_SUBREV_U16 : CanonOp;

def COP_V_MUL_LO_U16 : CanonOp;

def COP_V_DOT2C_I32_I16 : CanonOp;

def COP_V_DOT4C_I32_I8 : CanonOp;

def COP_V_DOT8C_I32_I4 : CanonOp;

def COP_V_DOT4_I32_IU8 : CanonOp;

def COP_V_PK_FMAC_F16 : CanonOp;

def COP_V_PACK_B32_F16 : CanonOp;

def COP_V_CVT_PK_BF16_F32 : CanonOp;

def COP_V_CVT_PK_BF8_F32 : CanonOp;

def COP_V_CVT_PK_FP8_F32 : CanonOp;

def COP_V_CVT_PKRTZ_F16_F32 : CanonOp;

def COP_V_CVT_PK_F16_F32 : CanonOp;

def COP_V_CVT_SCALEF32_PK_FP4_F32 : CanonOp;

def COP_V_CVT_SCALEF32_PK8_FP8_F32 : CanonOp<[{
  gfx1250 packed-8 scaled FP8 conversion: 8x f32 in a Wave32 vector ->
  8x packed FP8 (2 dwords) with a scalar f32 scale multiplier.
  Cross-target lowering uses 4 chained `cvt_pk_fp8_f32` calls after a
  vector multiply by the broadcast scale on targets with FP8 conversion
  support.
}]>;

def COP_V_BFM_B32 : CanonOp;

def COP_V_ADD_F64 : CanonOp<[{
  -- VOP2/VOP3 FP64 --
}]>;

def COP_V_MUL_F64 : CanonOp;

def COP_V_FMA_F64 : CanonOp;

def COP_V_FMAC_F64 : CanonOp;

def COP_V_FMAMK_F64 : CanonOp<[{
  gfx1250/gfx13 mandatory-literal FMA forms (VOP2-only, no e64). 64-bit K
  immediate carried in the encoding; operand layout mirrors the F32 forms:
  v_fmamk_f64 vd, src0, K, src1 -> vd = fma(src0, K, src1)
  v_fmaak_f64 vd, src0, src1, K -> vd = fma(src0, src1, K)
}]>;

def COP_V_FMAAK_F64 : CanonOp;

def COP_V_MAX_NUM_F64 : CanonOp<[{
  gfx12+ IEEE-754 maximumNumber/minimumNumber f64. LLVM names the pseudos
  `V_{MAX,MIN}_NUM_F64`; assembler mnemonics are `v_{max,min}_f64` on
  gfx12/gfx1250. Semantics prefer a numeric operand over NaN, matching
  LLVM's `maxnum` / `minnum` intrinsic contract.
}]>;

def COP_V_MIN_NUM_F64 : CanonOp;

def COP_V_MAXIMUM_F64 : CanonOp;

def COP_V_MINIMUM_F64 : CanonOp;

def COP_V_RCP_F64 : CanonOp<[{
  VOP1 FP64. v_rcp_f64 is a TRANS-class transcendental (see
  VOP1Instructions.td: `let TRANS = 1, SchedRW = [WriteTrans64]`),
  not a true reciprocal -- hardware returns a ~26-bit accurate
  approximation that the LLVM `int_amdgcn_rcp` intrinsic models
  exactly. We deliberately lift to that intrinsic rather than to a
  generic `fdiv 1.0, x` because (a) gfx942 isels the intrinsic
  straight back to v_rcp_f64 (no Newton-Raphson refinement is
  emitted), and (b) `fdiv` would lower to a software divide
  sequence on gfx942 unless `arcp`/fast-math flags are set, which
  would be a silent semantics change versus the source op.
}]>;

def COP_V_RSQ_F64 : CanonOp<[{
  VOP1 FP64 TRANS-class rsqrt; lifts to llvm.amdgcn.rsq.f64, the
  approximation the hardware op computes (same rationale as V_RCP_F64).
}]>;

def COP_V_SQRT_F64 : CanonOp<[{
  VOP1 FP64 sqrt; lifts to llvm.amdgcn.sqrt.f64, the hardware op's
  approximation (same rationale as V_RCP_F64, not a generic llvm.sqrt).
}]>;

def COP_V_LDEXP_F64 : CanonOp<[{
  VOP3 FP64 ldexp: F64 src0 * 2^(I32 src1). Lifted to the generic
  `llvm.ldexp.f64.i32` intrinsic; the AMDGPU backend lowers this back
  to v_ldexp_f64 on targets that have the op natively. No e32 form
  exists -- the pseudo is VOP3-only -- so output modifiers are
  required to be present in the MC operand table and must be default.
}]>;

def COP_V_TRIG_PREOP_F64 : CanonOp<[{
  VOP3 FP64 trig pre-op: look up 2/pi around src0 with segment select
  src1[4:0]. Lifted to llvm.amdgcn.trig.preop.f64.
}]>;

def COP_V_MAX_U32 : CanonOp;

def COP_V_MIN_U32 : CanonOp;

def COP_V_MAX_I32 : CanonOp;

def COP_V_MIN_I32 : CanonOp;

def COP_V_PERMLANE16_B32 : CanonOp;

def COP_V_PERMLANEX16_B32 : CanonOp;

def COP_V_PERMLANE64_B32 : CanonOp;

def COP_V_PERMLANE16_SWAP_B32 : CanonOp;

def COP_V_PERMLANE32_SWAP_B32 : CanonOp;

def COP_V_CMP : CanonOp<[{
  -- VOPC (V_CMP_* and V_CMPX_*) --

  All ~100 V_CMP_*_{U,I,F}{16,32,64} and V_CMPX_*_{U,I,F}{16,32} pseudos
  collapse onto these two CanonicalOps; the actual {predicate, element type,
  width} triple is looked up from `VCmpMeta` keyed on the MC opcode.
  `V_CMP`   writes an SGPR pair (or VCC, depending on the encoding).
  `V_CMPX`  additionally ANDs the compare result into EXEC.
}]>;

def COP_V_CMPX : CanonOp;

def COP_V_PK_ADD_F32 : CanonOp<[{
  -- VOP3P --
}]>;

def COP_V_PK_MUL_F32 : CanonOp;

def COP_V_PK_FMA_F32 : CanonOp;

def COP_V_PK_MAX_F32 : CanonOp;

def COP_V_PK_MIN_F32 : CanonOp;

def COP_V_PK_MOV_B32 : CanonOp;

def COP_V_PK_FMA_F16 : CanonOp<[{
  VOP3P packed fused FMA (VOP3PInstructions.td
  `V_PK_FMA_F16`, profile VOP_V2F16_V2F16_V2F16_V2F16):
  dst = fma(src0_lane, src1_lane, src2_lane) per `<2 x half>` lane.
  The source lane for each output lane is selected by the packed
  `srcN_modifiers` bits: OP_SEL_0 selects the source high half for the
  low output lane, OP_SEL_1 selects the source high half for the high
  output lane, and NEG / NEG_HI negate the low / high source values before
  the fused operation. The VOP3P clamp bit clamps the packed f16 result to
  [0, 1] after the fused operation.
}]>;

def COP_V_PK_ADD_F16 : CanonOp<[{
  VOP3P packed f16 add (VOP3PInstructions.td
  `V_PK_ADD_F16`, profile VOP_V2F16_V2F16_V2F16):
  dst = src0_lane + src1_lane per `<2 x half>` lane.
  Source lane selection (OP_SEL_0 / OP_SEL_1) and per-lane negation
  (NEG / NEG_HI) follow the same packed `srcN_modifiers` contract as
  V_PK_FMA_F16. The VOP3P clamp bit clamps the packed f16 result to
  [0, 1] after the add.
}]>;

def COP_V_PK_MUL_F16 : CanonOp<[{
  VOP3P packed f16 multiply (VOP3PInstructions.td
  `V_PK_MUL_F16`, profile VOP_V2F16_V2F16_V2F16):
  dst = src0_lane * src1_lane per `<2 x half>` lane.
  Source lane selection (OP_SEL_0 / OP_SEL_1) and per-lane negation
  (NEG / NEG_HI) follow the same packed `srcN_modifiers` contract as
  V_PK_FMA_F16. The VOP3P clamp bit clamps the packed f16 result to
  [0, 1] after the multiply.
}]>;

def COP_V_PK_ADD_BF16 : CanonOp<[{
  gfx1250 VOP3P packed BF16 arithmetic (VOP3PInstructions.td
  `V_PK_{ADD,MUL,MIN_NUM,MAX_NUM,FMA}_BF16`):
  V_PK_ADD_BF16:     dst = src0_lane + src1_lane
  V_PK_MUL_BF16:     dst = src0_lane * src1_lane
  V_PK_MIN_NUM_BF16: dst = minimumNumber(src0_lane, src1_lane)
  V_PK_MAX_NUM_BF16: dst = maximumNumber(src0_lane, src1_lane)
  V_PK_FMA_BF16:     dst = fma(src0_lane, src1_lane, src2_lane)
  per `<2 x bfloat>` lane. Source lane selection and per-lane negation use
  the same packed `srcN_modifiers` contract as V_PK_FMA_F16.
}]>;

def COP_V_PK_MUL_BF16 : CanonOp;

def COP_V_PK_MIN_NUM_BF16 : CanonOp;

def COP_V_PK_MAX_NUM_BF16 : CanonOp;

def COP_V_PK_FMA_BF16 : CanonOp;

def COP_V_PK_MAD_U16 : CanonOp<[{
  VOP3P packed-pair `<2 x i16>` int ops (gfx9+, available on both
  gfx942 and gfx1250 -- same MC encoding family). Operand profile is
  VOP_V2I16_V2I16_V2I16: 32-bit dst / 32-bit src0 / 32-bit src1, each
  bitcast to `<2 x i16>` before the op and back to i32 for the
  write-back. Inline literals encode a packed `<2 x i16>` directly
  (lo i16 = bits[15:0], hi i16 = bits[31:16]) -- NO broadcast
  analogue to the V_PK_F32 32-bit-element family, because the
  literal width matches the operand width here.

  V_PK_MAD_U16:     dst = src0 * src1 + src2         (lane-wise modular
  i16 multiply-add; ternary, VOP_V2I16_V2I16_V2I16_V2I16).
  V_PK_ADD_U16:     dst = src0 + src1                (lane-wise i16 add)
  V_PK_SUB_I16:     dst = src0 - src1                (lane-wise i16 sub)
  V_PK_LSHLREV_B16: dst = src1 << (src0 & 15)        (clshl_rev_16
  SDAG: shift count is src0, value is src1, low 4
  bits of the count select the shift amount per
  AMDGPU's hardware-clamp-to-element-width).
  V_PK_ASHRREV_I16: dst = src1 >>_arith (src0 & 15)  (cashr_rev_16
  SDAG: shift count is src0, value is src1, low 4
  bits of the count select the shift amount per
  AMDGPU's hardware-clamp-to-element-width with
  arithmetic (sign-extending) right-shift semantics).
  V_PK_MUL_LO_U16:  dst = (src0 * src1) & 0xFFFF     (lane-wise low
  16 bits of the integer product; "lo" means the
  low half of the 32-bit multiply, so signed vs
  unsigned does not change the result -- LLVM `mul`
  without nuw/nsw matches the AMDGPU modular
  multiply semantics directly).
  V_PK_MAX_I16:     dst = smax(src0, src1)           (lane-wise signed
  i16 max).
  V_PK_MAX3_I16:    dst = smax(smax(src0, src1), src2)  (lane-wise signed
  i16 3-way max; ternary).

  op_sel / op_sel_hi modifiers select which i16 of each source feeds
  each output lane (defaults: op_sel=[0,0,0], op_sel_hi=[1,1,1] --
  natural lo->lo, hi->hi packing).
}]>;

def COP_V_PK_ADD_U16 : CanonOp;

def COP_V_PK_SUB_I16 : CanonOp;

def COP_V_PK_LSHLREV_B16 : CanonOp;

def COP_V_PK_LSHRREV_B16 : CanonOp;

def COP_V_PK_ASHRREV_I16 : CanonOp;

def COP_V_PK_MUL_LO_U16 : CanonOp;

def COP_V_PK_MAX_I16 : CanonOp;

def COP_V_PK_MAX3_I16 : CanonOp;

def COP_V_BITOP3_B32 : CanonOp;

def COP_V_BITOP3_B16 : CanonOp;

def COP_V_ADD_I32 : CanonOp<[{
  GFX9 VOP3-only v_add/sub_i32 -- plain add/sub when clamp=0,
  saddsat/ssubsat when clamp=1.
}]>;

def COP_V_SUB_I32 : CanonOp;

def COP_V_LSHLREV_B64 : CanonOp<[{
  -- 64-bit vector ops --
}]>;

def COP_V_LSHRREV_B64 : CanonOp<[{
  gfx8+ VOP3 64-bit shifts. Same operand shape as V_LSHLREV_B64
  (i64 dst, i32 shamt, i64 src1, reversed-operand convention:
  `dst = src1 >> shamt`). Lower to LLVM `lshr` (logical right) and
  `ashr` (arithmetic right) on the i64 src1, with the i32 shamt
  zext'd to i64 -- the AMDGPU hardware masks the count to 6 bits so
  the LLVM behaviour matches as long as we feed a valid i32 (LLVM
  shifts >= bitwidth are poison, the hardware masks; we don't paper
  over the difference because corpus shifts always carry a finite
  immediate or a producer that already masks).
}]>;

def COP_V_ASHRREV_I64 : CanonOp;

def COP_V_LSHL_ADD_U64 : CanonOp;

def COP_V_ADD_NC_U64 : CanonOp;

def COP_V_SUB_NC_U64 : CanonOp;

def COP_V_MAX_I64 : CanonOp<[{
  gfx1250 VOP3 64-bit integer min/max.  These are pure per-lane
  compare-and-select operations: signed forms use i64 ordering, unsigned
  forms use u64 ordering.  They do not consult MODE and have no NaN,
  signed-zero, denorm/FTZ, or rounding behaviour.
}]>;

def COP_V_MAX_U64 : CanonOp;

def COP_V_MIN_I64 : CanonOp;

def COP_V_MIN_U64 : CanonOp;

def COP_V_MUL_U64 : CanonOp<[{
  gfx1250 VOP2 64-bit unsigned multiply (low 64 bits of s0 * s1).
}]>;

def COP_V_MAD_U64_U32 : CanonOp;

def COP_V_MAD_CO_U64_U32 : CanonOp;

def COP_V_MAD_NC_U64_U32 : CanonOp<[{
  gfx1250 no-carry 64-bit multiply-add VOP3 opcodes
  (VOP3Only_Realtriple_gfx1250, VOP3Instructions.td:2129 / 2130: encodings
  0x2fa / 0x2fb).  Both widen two 32-bit sources into a 64-bit accumulator:
  V_MAD_NC_U64_U32: D.u64 = zext(S0.u32)*zext(S1.u32) + S2.u64
  V_MAD_NC_I64_I32: D.i64 = sext(S0.i32)*sext(S1.i32) + S2.i64
  Neither produces a carry/overflow output (hence the "nc" suffix).  The
  backend's AMDGPUISelDAGToDAG.cpp::SelectMad64_32 pattern-matches the
  canonical `add(mul(zext/sext s0, zext/sext s1), s2_i64)` IR we emit
  back into v_mad_(nc|co|_i64_i32) on whichever target the raise writes
  to -- identical to how V_MAD_U64_U32 lowers today (see handle-valu.cpp
  v_mad_u64_u32 arm and the "LLVM no longer exposes a distinct
  carry-out variant" note on that CanonicalOp for historical context).
}]>;

def COP_V_MAD_NC_I64_I32 : CanonOp;

def COP_FLAT_LOAD_UBYTE : CanonOp<[{
  -- FLAT / GLOBAL / SCRATCH memory --
}]>;

def COP_FLAT_LOAD_SBYTE : CanonOp;

def COP_FLAT_LOAD_USHORT : CanonOp;

def COP_FLAT_LOAD_SSHORT : CanonOp;

def COP_FLAT_LOAD_DWORD : CanonOp;

def COP_FLAT_LOAD_DWORDX2 : CanonOp;

def COP_FLAT_LOAD_DWORDX3 : CanonOp;

def COP_FLAT_LOAD_DWORDX4 : CanonOp;

def COP_FLAT_STORE_BYTE : CanonOp;

def COP_FLAT_STORE_BYTE_D16_HI : CanonOp;

def COP_FLAT_STORE_SHORT : CanonOp;

def COP_FLAT_STORE_SHORT_D16_HI : CanonOp;

def COP_FLAT_STORE_DWORD : CanonOp;

def COP_FLAT_STORE_DWORDX2 : CanonOp;

def COP_FLAT_STORE_DWORDX3 : CanonOp;

def COP_FLAT_STORE_DWORDX4 : CanonOp;

def COP_GLOBAL_LOAD_UBYTE : CanonOp;

def COP_GLOBAL_LOAD_SBYTE : CanonOp;

def COP_GLOBAL_LOAD_USHORT : CanonOp;

def COP_GLOBAL_LOAD_SSHORT : CanonOp;

def COP_GLOBAL_LOAD_SHORT_D16_HI : CanonOp;

def COP_GLOBAL_LOAD_DWORD : CanonOp;

def COP_GLOBAL_LOAD_DWORDX2 : CanonOp;

def COP_GLOBAL_LOAD_DWORDX3 : CanonOp;

def COP_GLOBAL_LOAD_DWORDX4 : CanonOp;

def COP_GLOBAL_STORE_BYTE : CanonOp;

def COP_GLOBAL_STORE_BYTE_D16_HI : CanonOp;

def COP_GLOBAL_STORE_SHORT : CanonOp;

def COP_GLOBAL_STORE_SHORT_D16_HI : CanonOp;

def COP_GLOBAL_STORE_DWORD : CanonOp;

def COP_GLOBAL_STORE_DWORDX2 : CanonOp;

def COP_GLOBAL_STORE_DWORDX3 : CanonOp;

def COP_GLOBAL_STORE_DWORDX4 : CanonOp;

def COP_GLOBAL_WB : CanonOp;

def COP_GLOBAL_INV : CanonOp;

def COP_SCRATCH_LOAD_DWORD : CanonOp;

def COP_SCRATCH_LOAD_DWORDX2 : CanonOp;

def COP_SCRATCH_LOAD_DWORDX3 : CanonOp;

def COP_SCRATCH_LOAD_DWORDX4 : CanonOp;

def COP_SCRATCH_STORE_DWORD : CanonOp;

def COP_SCRATCH_STORE_DWORDX2 : CanonOp;

def COP_SCRATCH_STORE_DWORDX3 : CanonOp;

def COP_SCRATCH_STORE_DWORDX4 : CanonOp;

def COP_SCRATCH_LOAD_UBYTE : CanonOp<[{
  Sub-dword scratch (private) accesses. Loads zero/sign-extend the sub-dword
  value into the 32-bit VGPR; stores truncate the low byte/short. Mirror the
  GLOBAL_{LOAD,STORE}_{U,S}BYTE / SHORT semantics on the private address
  space.
}]>;

def COP_SCRATCH_LOAD_SBYTE : CanonOp;

def COP_SCRATCH_LOAD_USHORT : CanonOp;

def COP_SCRATCH_LOAD_SSHORT : CanonOp;

def COP_SCRATCH_STORE_BYTE : CanonOp;

def COP_SCRATCH_STORE_SHORT : CanonOp;

def COP_FLAT_ATOMIC_ADD : CanonOp<[{
  -- FLAT atomics --
}]>;

def COP_FLAT_ATOMIC_SUB : CanonOp;

def COP_FLAT_ATOMIC_AND : CanonOp;

def COP_FLAT_ATOMIC_OR : CanonOp;

def COP_FLAT_ATOMIC_XOR : CanonOp;

def COP_FLAT_ATOMIC_SMIN : CanonOp;

def COP_FLAT_ATOMIC_SMAX : CanonOp;

def COP_FLAT_ATOMIC_UMIN : CanonOp;

def COP_FLAT_ATOMIC_UMAX : CanonOp;

def COP_FLAT_ATOMIC_SWAP : CanonOp;

def COP_FLAT_ATOMIC_CMPSWAP : CanonOp;

def COP_FLAT_ATOMIC_ADD_X2 : CanonOp<[{
  64-bit integer RMW family (`*_X2` in LLVM TableGen, printed as
  `flat_atomic_*_{u64,i64,b64}` on gfx11+ / gfx12+).
}]>;

def COP_FLAT_ATOMIC_SUB_X2 : CanonOp;

def COP_FLAT_ATOMIC_AND_X2 : CanonOp;

def COP_FLAT_ATOMIC_OR_X2 : CanonOp;

def COP_FLAT_ATOMIC_XOR_X2 : CanonOp;

def COP_FLAT_ATOMIC_SMIN_X2 : CanonOp;

def COP_FLAT_ATOMIC_SMAX_X2 : CanonOp;

def COP_FLAT_ATOMIC_UMIN_X2 : CanonOp;

def COP_FLAT_ATOMIC_UMAX_X2 : CanonOp;

def COP_FLAT_ATOMIC_SWAP_X2 : CanonOp;

def COP_FLAT_ATOMIC_CMPSWAP_X2 : CanonOp;

def COP_FLAT_ATOMIC_ADD_F32 : CanonOp;

def COP_FLAT_ATOMIC_ADD_F64 : CanonOp;

def COP_FLAT_ATOMIC_MIN_NUM_F64 : CanonOp;

def COP_FLAT_ATOMIC_MAX_NUM_F64 : CanonOp;

def COP_GLOBAL_ATOMIC_ADD : CanonOp<[{
  -- GLOBAL atomics --
}]>;

def COP_GLOBAL_ATOMIC_ADD_X2 : CanonOp<[{
  64-bit unsigned integer add (`GLOBAL_ATOMIC_ADD_X2` in LLVM TableGen,
  printed as `global_atomic_add_u64` on gfx11+ / gfx12+).
}]>;

def COP_GLOBAL_ATOMIC_SUB : CanonOp;

def COP_GLOBAL_ATOMIC_AND : CanonOp;

def COP_GLOBAL_ATOMIC_OR : CanonOp;

def COP_GLOBAL_ATOMIC_XOR : CanonOp;

def COP_GLOBAL_ATOMIC_SMIN : CanonOp;

def COP_GLOBAL_ATOMIC_SMAX : CanonOp;

def COP_GLOBAL_ATOMIC_UMIN : CanonOp;

def COP_GLOBAL_ATOMIC_UMAX : CanonOp;

def COP_GLOBAL_ATOMIC_SWAP : CanonOp;

def COP_GLOBAL_ATOMIC_CMPSWAP : CanonOp;

def COP_GLOBAL_ATOMIC_ADD_F32 : CanonOp;

def COP_GLOBAL_ATOMIC_PK_ADD_BF16 : CanonOp;

def COP_GLOBAL_ATOMIC_PK_ADD_F16 : CanonOp;

def COP_GLOBAL_ATOMIC_ADD_F64 : CanonOp;

def COP_GLOBAL_ATOMIC_MIN_NUM_F64 : CanonOp;

def COP_GLOBAL_ATOMIC_MAX_NUM_F64 : CanonOp;

def COP_S_ATOMIC_SWAP : CanonOp<[{
  -- SMEM atomics --
  gfx8+ scalar-cache atomics.  Lifted to `atomicrmw` IR via handle-smem.cpp;
  the SCOPE/GLC bits fold into AtomicOrdering (monotonic) and whether the
  return-value slot is written back.

  S_ATOMIC_DEC has wrap-at-zero semantics that do NOT match a plain
  `atomicrmw sub` -- the hardware computes
  new = (old == 0 || old > src) ? src : old - 1
  which is exactly LLVM's `AtomicRMWInst::UDecWrap` binop (landed in
  LLVM 19).  The canonical split-k "last workgroup runs the epilogue"
  barrier counter is the overwhelming corpus use (every AITER
  `bf16gemm_*_splitk_clean.co` kernel), keyed on whether the returned
  pre-decrement value equals 1.  Like S_ATOMIC_SWAP this op is classed
  NonCommutative for the Class-3 wave-size obstruction classifier in
  wave-size-obstruction.cpp (a lane-id-derived decrement sequence's
  outcome is replica-order-dependent under modulo-replication).
}]>;

def COP_S_ATOMIC_DEC : CanonOp;

def COP_DS_LOAD_TR16_B128 : CanonOp<[{
  -- DS --
}]>;

def COP_DS_READ_B64_TR_B16 : CanonOp;

def COP_DS_READ_B64_TR_B8 : CanonOp;

def COP_DS_LOAD_TR8_B64 : CanonOp<[{
  gfx1250 spelling of the same 64-bit transposed LDS load that
  gfx950 disassembles as `ds_read_b64_tr_b8`. The hardware
  semantics are identical: each lane reads 64 bits (8 x i8) from
  its LDS base, then the data is transposed across 8-lane groups
  so each lane post-transpose holds 8 i8 values from 8 different
  source lanes at the same intra-group element offset (v2i32
  packed). The two CanonicalOps are kept distinct because they are two
  distinct LLVM MC opcodes (DS_LOAD_TR8_B64 vs DS_READ_B64_TR_B8)
  with separate isel patterns and separate intrinsics
  (`int_amdgcn_ds_load_tr8_b64` gated isGFX1250Plus,
  `int_amdgcn_ds_read_tr8_b64` gated HasGFX950Insts); both lower
  through the same hand-rolled bpermute-based emulation in
  handle-ds.cpp because gfx942 (the transpiler's target ISA) has
  neither isel pattern and no in-tree pre-isel emulation.
}]>;

def COP_DS_READ_B32 : CanonOp;

def COP_DS_READ_B64 : CanonOp;

def COP_DS_READ_B96 : CanonOp<[{
  96-bit (3 x i32) LDS load. LLVM MC opcode `DS_READ_B96`; gfx11+
  (gfx1100/gfx1200/gfx1250) renames the asm spelling to
  `ds_load_b96` (DSInstructions.td:1578 declares
  `defm DS_READ_B96 : DS_Real_gfx11_gfx12_gfx13<0x0fe,
  "ds_load_b96">`). Hardware reads 96 bits from the lane's LDS
  base; the lift is `load <3 x i32>` from addrspace(3). The
  gfx942 backend lowers the 3-dword vector load to either a
  native `ds_read_b96` (gfx9 inherits the `_vi` Real form) or
  splits it into 3x `ds_read_b32` with the appropriate
  increments -- both are correct in-place lowerings.
  Inserted between DS_READ_B64 and DS_READ_B128 deliberately so
  the existing range checks (`sop >= DS_READ_B32 &&
  sop <= DS_READ_I8` for reads, parallel for writes) continue to
  cover it without a special case.
}]>;

def COP_DS_READ_B128 : CanonOp;

def COP_DS_READ2_B32 : CanonOp;

def COP_DS_READ2_B64 : CanonOp;

def COP_DS_READ2ST64_B32 : CanonOp<[{
  gfx11+ stride-64 two-address LDS load forms
  (DSInstructions.td:1529,1542 -- `ds_load_2addr_stride64_b{32,64}`).
  Semantics parallel DS_READ2_B{32,64}, but the per-access byte
  offset is `rawFieldValue * 256` (B32) or `* 512` (B64) instead of
  `* 4` / `* 8`, extending reach with the same 8-bit offset field
  at the cost of a 64-dword stride granularity. Handled jointly
  with the non-ST64 variants in handle-ds.cpp's dedicated
  READ2/WRITE2 block; placed adjacent in the enum so the existing
  `sop >= DS_READ_B32 && sop <= DS_READ_I8` range check continues
  to classify them as DS reads (the dedicated block intercepts
  before the single-offset generic handler ever sees them).
}]>;

def COP_DS_READ2ST64_B64 : CanonOp;

def COP_DS_READ_U16 : CanonOp;

def COP_DS_READ_I16 : CanonOp;

def COP_DS_READ_U8 : CanonOp;

def COP_DS_READ_I8 : CanonOp;

def COP_DS_WRITE_B32 : CanonOp;

def COP_DS_WRITE_B64 : CanonOp;

def COP_DS_WRITE_B96 : CanonOp<[{
  Symmetric write-side for `ds_load_b96`: gfx11+ asm spelling is
  `ds_store_b96` (DSInstructions.td:1576); the LLVM MC opcode
  remains `DS_WRITE_B96`. Lift is `store <3 x i32>` to
  addrspace(3). Inserted between DS_WRITE_B64 and DS_WRITE_B128
  for the same range-check reason as DS_READ_B96 above.
}]>;

def COP_DS_WRITE_B128 : CanonOp;

def COP_DS_WRITE2_B32 : CanonOp;

def COP_DS_WRITE2_B64 : CanonOp;

def COP_DS_WRITE2ST64_B32 : CanonOp<[{
  gfx11+ stride-64 two-address LDS store forms (mirror the
  DS_READ2ST64 block above; see the read-side comment for the
  offset-scaling rationale and enum-placement reasoning).
}]>;

def COP_DS_WRITE2ST64_B64 : CanonOp;

def COP_DS_WRITE_B16 : CanonOp;

def COP_DS_WRITE_B8 : CanonOp;

def COP_DS_WRITE_B16_D16_HI : CanonOp<[{
  D16_HI partial-store family (gfx8+ HasD16LoadStore):
  store the upper 16 bits (B16_D16_HI) or bits [23:16] (B8_D16_HI)
  of the source VGPR to LDS. The "D16_HI" suffix names the
  *source* register half being stored, not a dest-merge -- these
  are write-only and there is no tied dest_in operand. The
  companion D16 reads (DS_READ_U/I8_D16{,_HI}, DS_READ_U16_D16{,_HI})
  are not yet on the worklist; if they surface, add them here as
  a separate set with their own tied-source dest_in handling.
}]>;

def COP_DS_WRITE_B8_D16_HI : CanonOp;

def COP_DS_BPERMUTE_B32 : CanonOp;

def COP_DS_PERMUTE_B32 : CanonOp<[{
  Forward cross-lane permute (PUSH semantics): lane i sends its src
  datum to destination lane addr[i]>>2. Mirror of DS_BPERMUTE_B32
  (PULL) lifted through llvm.amdgcn.ds.permute. Class 2 cross-lane
  (hotswap/docs/wave-size-translation.md sec. 5.3); handled in
  handle-ds.cpp with the same source-wave selector rebase as
  DS_BPERMUTE_B32 under wave32->wave64 cross-widening.
}]>;

def COP_DS_SWIZZLE_B32 : CanonOp<[{
  Class 2 DsSwizzle (hotswap/docs/wave-size-translation.md sec. 6).
  Wave-width-specific cross-lane shuffle. The handler refuses with
  `unsupportedInstructionForm` until the P6 rewrite (lift through
  llvm.amdgcn.ds.swizzle -- see wave-size-translation.md sec. 5.3 row
  P6) lands; the wave-size classifier
  (wave-size-obstruction.cpp) flags it before the handler is even
  dispatched in the cross-wave case.
}]>;

def COP_DS_ADD_F64 : CanonOp<[{
  -- DS atomics --
}]>;

def COP_DS_ADD_U32 : CanonOp;

def COP_BUFFER_LOAD_DWORD : CanonOp<[{
  -- MUBUF --
}]>;

def COP_BUFFER_LOAD_DWORDX2 : CanonOp;

def COP_BUFFER_LOAD_DWORDX3 : CanonOp;

def COP_BUFFER_LOAD_DWORDX4 : CanonOp;

def COP_BUFFER_LOAD_UBYTE : CanonOp;

def COP_BUFFER_LOAD_SBYTE : CanonOp;

def COP_BUFFER_LOAD_USHORT : CanonOp;

def COP_BUFFER_LOAD_SSHORT : CanonOp;

def COP_BUFFER_LOAD_SHORT_D16 : CanonOp;

def COP_BUFFER_LOAD_SHORT_D16_HI : CanonOp;

def COP_BUFFER_LOAD_UBYTE_D16 : CanonOp<[{
  D16 byte variants -- gfx9+ partial-write loads. The 8-bit datum is
  sign- or zero-extended to i16 and merged into the lo (`_D16`) or
  hi (`_D16_HI`) half of the destination VGPR; the other 16 bits
  are preserved (BUFInstructions.td:1155-1169, predicate
  `D16PreservesUnusedBits`). Mnemonic on gfx11+/gfx1250 is
  `buffer_load_d16_u8` / `_d16_i8` / `_d16_hi_u8` / `_d16_hi_i8`.
}]>;

def COP_BUFFER_LOAD_UBYTE_D16_HI : CanonOp;

def COP_BUFFER_LOAD_SBYTE_D16 : CanonOp;

def COP_BUFFER_LOAD_SBYTE_D16_HI : CanonOp;

def COP_BUFFER_LOAD_DWORD_LDS : CanonOp;

def COP_BUFFER_LOAD_DWORDX2_LDS : CanonOp;

def COP_BUFFER_LOAD_DWORDX4_LDS : CanonOp;

def COP_BUFFER_STORE_DWORDX4_LDS : CanonOp;

def COP_BUFFER_STORE_DWORD : CanonOp;

def COP_BUFFER_STORE_DWORDX2 : CanonOp;

def COP_BUFFER_STORE_DWORDX3 : CanonOp;

def COP_BUFFER_STORE_DWORDX4 : CanonOp;

def COP_BUFFER_STORE_BYTE : CanonOp;

def COP_BUFFER_STORE_SHORT : CanonOp;

def COP_BUFFER_ATOMIC_ADD : CanonOp<[{
  -- MUBUF atomics --
  Order is significant: handle-mubuf.cpp dispatches via the range
  check `[BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_PK_ADD_F16]`. New
  BUFFER_ATOMIC_* CanonicalOps must stay inside this range so the range
  check picks them up; entries the handler does not explicitly
  case-match are caught by the switch's default branch with a
  `RaiseFailure::unsupportedInstructionForm("unsupported buffer atomic")`.
}]>;

def COP_BUFFER_ATOMIC_SUB : CanonOp;

def COP_BUFFER_ATOMIC_AND : CanonOp;

def COP_BUFFER_ATOMIC_OR : CanonOp;

def COP_BUFFER_ATOMIC_XOR : CanonOp;

def COP_BUFFER_ATOMIC_SWAP : CanonOp<[{
  Class 3 non-commutative atomics (NonCommutativeAtomic), see
  hotswap/docs/wave-size-translation.md sec. 6.
  The wave-size classifier flags these in the cross-wave case;
  handle-mubuf.cpp models them with raw-buffer atomics so same-wave
  and same-target lifts preserve descriptor-relative addressing.
}]>;

def COP_BUFFER_ATOMIC_CMPSWAP : CanonOp;

def COP_BUFFER_ATOMIC_ADD_F32 : CanonOp;

def COP_BUFFER_ATOMIC_PK_ADD_BF16 : CanonOp;

def COP_BUFFER_ATOMIC_PK_ADD_F16 : CanonOp;

def COP_BUFFER_ATOMIC_ADD_F64 : CanonOp;

def COP_BUFFER_ATOMIC_MIN_F64 : CanonOp<[{
  gfx942 raw `<`/`>` comparator form (ISA manual 12.15.3 op 80/81).
}]>;

def COP_BUFFER_ATOMIC_MAX_F64 : CanonOp;

def COP_BUFFER_ATOMIC_MIN_NUM_F64 : CanonOp<[{
  gfx12 IEEE 754-2019 minimumNumber/maximumNumber form.
}]>;

def COP_BUFFER_ATOMIC_MAX_NUM_F64 : CanonOp;

def COP_MATRIX_OP_BEGIN_SENTINEL : CanonOp {
  let IsSentinel = 1;
}

def COP_V_MFMA_F32_16x16x128_F8F6F4 : CanonOpAlias<COP_MATRIX_OP_BEGIN_SENTINEL, [{
  -- MFMA --
  gfx950 scaled F8F6F4 variants share a per-shape intrinsic but take 9
  src-format sub-variants each; those are collapsed onto these four
  CanonicalOps in the table below.
}]>;

def COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4 : CanonOp;

def COP_V_MFMA_F32_32x32x64_F8F6F4 : CanonOp;

def COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4 : CanonOp;

def COP_V_MFMA_F32_16x16x16_F16 : CanonOp<[{
  F32 <- F16/F32 (gfx908+). Each covers its pseudo's _e64/_vgprcd_/_mac_
  variants via the pseudo-alias stripping hotswap-tblgen performs.
}]>;

def COP_V_MFMA_F32_32x32x8_F16 : CanonOp;

def COP_V_MFMA_F32_16x16x4_F32 : CanonOp;

def COP_V_MFMA_F32_32x32x1_F32 : CanonOp;

def COP_V_MFMA_F32_32x32x2_F32 : CanonOp;

def COP_V_MFMA_F32_4x4x1_F32 : CanonOp;

def COP_V_MFMA_F32_16x16x1_F32 : CanonOp;

def COP_V_MFMA_F32_32x32x4_F16 : CanonOp;

def COP_V_MFMA_F32_16x16x4_F16 : CanonOp;

def COP_V_MFMA_F32_4x4x4_F16 : CanonOp;

def COP_V_MFMA_I32_16x16x32_I8 : CanonOp<[{
  I32 <- I8.
}]>;

def COP_V_MFMA_I32_32x32x16_I8 : CanonOp;

def COP_V_MFMA_I32_32x32x4_I8 : CanonOp;

def COP_V_MFMA_I32_16x16x4_I8 : CanonOp;

def COP_V_MFMA_I32_4x4x4_I8 : CanonOp;

def COP_V_MFMA_F32_16x16x8_XF32 : CanonOp<[{
  F32 <- XF32 (gfx940+).
}]>;

def COP_V_MFMA_F32_32x32x4_XF32 : CanonOp;

def COP_V_MFMA_F32_32x32x2_BF16 : CanonOp<[{
  F32 <- BF16 (gfx908 2-byte variants).
}]>;

def COP_V_MFMA_F32_16x16x2_BF16 : CanonOp;

def COP_V_MFMA_F32_4x4x2_BF16 : CanonOp;

def COP_V_MFMA_F32_16x16x16_BF16_1K : CanonOp<[{
  F32 <- BF16 "1K" shapes (gfx90a+).
}]>;

def COP_V_MFMA_F32_32x32x8_BF16_1K : CanonOp;

def COP_V_MFMA_F32_16x16x32_BF16 : CanonOp<[{
  F32 <- BF16/F16 wide shapes (gfx950).
}]>;

def COP_V_MFMA_F32_32x32x16_BF16 : CanonOp;

def COP_V_MFMA_F32_16x16x32_F16 : CanonOp;

def COP_V_MFMA_F32_16x16x32_FP8_FP8 : CanonOp<[{
  F32 <- FP8/BF8 (gfx940+).
}]>;

def COP_V_MFMA_F32_16x16x32_FP8_BF8 : CanonOp;

def COP_V_MFMA_F32_16x16x32_BF8_FP8 : CanonOp;

def COP_V_MFMA_F32_16x16x32_BF8_BF8 : CanonOp;

def COP_V_MFMA_F32_32x32x16_FP8_FP8 : CanonOp;

def COP_V_MFMA_F32_32x32x16_FP8_BF8 : CanonOp;

def COP_V_MFMA_F32_32x32x16_BF8_FP8 : CanonOp;

def COP_V_MFMA_F32_32x32x16_BF8_BF8 : CanonOp;

def COP_V_WMMA_F32_16x16x32_F16 : CanonOp<[{
  -- WMMA (gfx1250) --
  16x16x32 WMMA with f32 accumulator and 16-bit element types. Both
  share the same per-lane fragment shape (A,B: <16 x t>, C/D:
  <8 x f32>) and same K-decomposition path through the gfx942 MFMA
  lowering -- `emitWMMAtoMFMA` is parameterised on input element
  type and routes to the matching CDNA3 MFMA intrinsic
  (mfma_f32_16x16x16f16 vs mfma_f32_16x16x16bf16_1k).
}]>;

def COP_V_WMMA_F32_16x16x32_BF16 : CanonOp;

def COP_V_WMMA_F32_16x16x4_F32 : CanonOp<[{
  16x16x4 WMMA with f32 accumulator and 32-bit f32 element types
  for both A and B (gfx1250 RDNA4 VOP3P opcode 0x05D). Per-Wave32-
  lane fragment shape is A,B: <2 x f32> (only 4 K-elements split
  across 2 dwords per lane), C/D: <8 x f32>; this is structurally
  distinct from the 16-bit (K=32, A/B = <16 x t>) and 8-bit
  (K=64, A/B = <8 x i32>) families above and so does NOT share
  the `emitWMMAtoMFMA` decomposition (which is parameterised on
  16-/8-bit element packing, not f32). The native intrinsic
  `amdgcn_wmma_f32_16x16x4_f32` is declared inside
  `AMDGPUWMMAIntrinsicsGFX1250` (gated by `isGFX125xOnly` in
  IntrinsicsAMDGPU.td:4113-4114) and is NOT part of the gfx12
  RDNA4-base WMMA family (`AMDGPUWMMAIntrinsicsGFX12`,
  FeatureWMMA{128,256}bInsts), so the same-target lift gates on
  `ISAProfile::hasTensorOps` (FeatureGFX1250Insts) -- matching
  the LLVM intrinsic's actual subtarget gating -- rather than
  `hasWMMA12`. Call shape is `AMDGPUWmmaIntrinsicModsAllReuse`,
  8 args: `(A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b)`.
  Cross-target lift to gfx942 would need a new K=4 MFMA
  decomposition path (gfx942 has `mfma_f32_16x16x4f32`) that no
  kernel in the current corpus exercises, so we refuse loudly
  via `RaiseFailure::unsupportedInstructionForm` to surface the gap
  immediately rather than silently degrade.
}]>;

def COP_V_WMMA_F32_16x16x64_FP8_FP8 : CanonOp<[{
  16x16x64 WMMA with f32 accumulator and 8-bit element types
  (fp8/bf8). The four AB combinations are distinct opcodes (and
  distinct CDNA3 MFMA intrinsics on gfx942) but share the same
  per-lane fragment shape (A,B: <8 x i32> = 32 fp8/bf8 bytes per
  Wave32 lane, C/D: <8 x f32>) and the same gfx942 MFMA decomposition
  path through `emitWMMAtoMFMA`. The K=64 dimension splits into
  2 chained K=32 MFMAs per Wave32 group, mirroring the K=32->2xK=16
  split used for the 16-bit variants. The lane-redistribution math
  is byte-identical between the two K-families (32 bytes per lane
  either way), so the only divergence inside `emitWMMAtoMFMA` is the
  per-MFMA pack type (i64 vs <4 x half|i16>) and the dispatched
  intrinsic ID. See `WMMAInputType` in `wmma-lowering.h` for the
  full enumeration.
}]>;

def COP_V_WMMA_F32_16x16x64_FP8_BF8 : CanonOp;

def COP_V_WMMA_F32_16x16x64_BF8_FP8 : CanonOp;

def COP_V_WMMA_F32_16x16x64_BF8_BF8 : CanonOp;

def COP_V_WMMA_I32_16x16x64_IU8 : CanonOp<[{
  16x16x64 WMMA with i32 accumulator and unsigned/signed 8-bit
  integer inputs (the gfx1250 IU8 variant; the LLVM intrinsic
  uses `iu8` to denote that the per-input sign extension is
  selected at call site through the `neg_lo` modifier rather
  than the opcode itself). Per-Wave32-lane fragment shape is
  identical to the FP8 sibling (A,B: <8 x i32> = 32 packed i8
  bytes per lane, C/D: <8 x i32> for integer accumulator). On
  gfx942 we lower through the same `emitWMMAtoMFMA` helper,
  dispatching the per-MFMA call to `mfma_i32_16x16x32_i8`
  (i64 packed A/B, <4 x i32> accumulator). The handler must
  also use a different native-WMMA12 intrinsic shape on gfx12
  hardware: `AMDGPUWmmaIntrinsicModsABClamp` (8 args including
  a trailing clamp flag), distinct from the 16-bit AllReuse
  and the 8-bit FP8 ModsC shapes.
}]>;

def COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4 : CanonOp<[{
  16x16x128 WMMA with f32 accumulator and per-matrix scale exponents,
  f8f6f4 mantissa-format family (gfx1250 RDNA4 VOP3P opcode 0x033 in
  VOP3PX2 form, pseudo `V_WMMA_SCALE_F32_16X16X128_F8F6F4_*_w32_*`).
  Each kernel encodes one of 9 opcode-suffix mantissa-pair variants
  (`{f4,f6,f8} x {f4,f6,f8}`), but the in-family element format
  (BF8 vs FP8 within f8; BF6 vs FP6 within f6) is selected at runtime
  by the `matrix_a_fmt` / `matrix_b_fmt` named-immediate operands
  (`enum MatrixFMT { FP8=0, BF8=1, FP6=2, BF6=3, FP4=4 }`,
  SIDefines.h:1052-1058). Per-Wave32-lane fragment shape is therefore
  format-dependent: A is `<16 x i32>` for f8 (32 packed bytes/lane),
  `<12 x i32>` for f6 (24 packed bytes/lane), and `<8 x i32>` for f4
  (16 packed bytes/lane); B is independently `<16/12/8 x i32>` per
  its own format. C/D is `<8 x f32>`. We collapse all 18 MC pseudos
  (9 mantissa pairs x `_twoaddr` / `_threeaddr`) onto this single
  CanonicalOp and discriminate at the handler with `getNamedOperandIdx`,
  mirroring the F8F6F4 MFMA collapse rule in the table below.

  The native intrinsic `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4`
  (IntrinsicsAMDGPU.td:4138, class `AMDGPUWmmaScaleIntrinsicModsC
  <llvm_i32_ty>`) takes 14 arguments:
  (i32 matrix_a_fmt, <Nax i32> A, i32 matrix_b_fmt, <Nbx i32> B,
  i16 C_modifiers, <8 x f32> C,
  i32 matrix_a_scale, i32 matrix_a_scale_fmt, i32 scale_src0,
  i32 matrix_b_scale, i32 matrix_b_scale_fmt, i32 scale_src1,
  i1 matrix_a_reuse, i1 matrix_b_reuse)
  and is gated by `isGFX125xOnly` inside `AMDGPUWMMAIntrinsicsGFX1250`
  (IntrinsicsAMDGPU.td:4113). Handler decodes operands by name via
  `AMDGPU::getNamedOperandIdx` (`matrix_a_fmt`, `matrix_b_fmt`,
  `matrix_a_scale`, `matrix_b_scale`, `matrix_a_scale_fmt`,
  `matrix_b_scale_fmt`, `scale_src0`, `scale_src1`, `matrix_a_reuse`,
  `matrix_b_reuse`, `src2_modifiers`) so any future TableGen operand
  reshuffle flows in for free.

  === Same-target gfx1250 -> gfx1250 contract ===

  Lift directly to `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4` with
  overloaded {<8 x f32>, <16 x i32>, <16 x i32>} type arguments
  (the overload widths match the f8 family used by the corpus
  kernels; the matrix_a_fmt / matrix_b_fmt immediates carry the
  BF8 vs FP8 distinction). The call is NOT wrapped in
  `emitUnderExec` because the WMMA intrinsic is `IntrConvergent`
  and operates on the wave's matrix fragment, not per-lane
  divergent values -- wrapping would emit one matrix multiply per
  active lane.

  === Cross-target (gfx942 and earlier) contract ===

  gfx942 has no scaled-WMMA hardware. The closest sibling on gfx942
  is `mfma_scale_f32_16x16x128_f8f6f4` (already mapped via
  `V_MFMA_SCALE_F32_16x16x128_F8F6F4`), but the WMMA-to-MFMA lane
  redistribution for K=128 + per-matrix-fmt selection + the
  matrix_a/b_scale_fmt x scale_src0/src1 exponent application is
  not modelled in `wmma-lowering.cpp` (only K=32 / K=64 fp16/bf16/
  fp8/bf8/iu8 paths exist). Per the user-rules (no silent
  fallbacks) and consistent with the gfx1250-only refusal contract
  applied to `V_WMMA_F32_16x16x4_F32` above, we refuse loudly via
  `RaiseFailure::unsupportedInstructionForm` to surface both the cross-target
  capability gap and the missing scaled-WMMA decomposition path.
}]>;

def COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4 : CanonOp<[{
  gfx1250 scaled-WMMA family with SCALE16. On same-target
  gfx1250 both lift to the matching native scaled-WMMA intrinsic. On
  cross-target gfx942/gfx950, both lower by decomposing K=128 into
  unscaled FP8/BF8 MFMA steps, applying the per-block scale factors
  in software, and running the usual Wave32-to-Wave64 lane redistribution.

  V_WMMA_SCALE16_F32_16x16x128_F8F6F4 covers the full f8f6f4.

  V_WMMA_SCALE16_F32_32x16x128_F4 is the M=32 FP4 shape. Cross-target
  lowering splits the M dimension into two 16x16 SCALE16 passes and
  recombines the partial results.
}]>;

def COP_V_WMMA_SCALE16_F32_32x16x128_F4 : CanonOp;

def COP_MATRIX_OP_END_SENTINEL : CanonOpAlias<COP_V_WMMA_SCALE16_F32_32x16x128_F4> {
  let IsSentinel = 1;
}

def COP_VOPD_GENERIC : CanonOp<[{
  -- VOPD -- (handled via string parsing of fullText, not opcode)
}]>;

def COP_TENSOR_LOAD_TO_LDS : CanonOp<[{
  -- VIMAGE TENSOR (gfx1250-only) --
  Tensor descriptor memory ops driven by the gfx1250 TENSOR cnt unit
  (`MIMGInstructions.td:2049-2113`, `VIMAGE_TENSOR_Pseudo`,
  `let SubtargetPredicate = isGFX125xOnly`). Each opcode encodes
  up to four 128-/256-bit Tensor Descriptors (`D# group 0..3`),
  a `R128A16:$r128` flag, and a `CPol:$cpol` cachepolicy immediate.
  `_d2` is the up-to-2D form (passes NULL for D# group 2/3); `_d4`
  is the up-to-4D form. Both share the same CanonicalOp here because
  their semantic intent is identical and their refusal contract is
  identical too -- the handler `handleVIMAGE` discriminates on
  `di.Mnemonic` only when shape differentiation matters (e.g.,
  a future native-target intrinsic-emit path that fills the
  0-init D# operands for `_d2`).

  gfx942 has no equivalent hardware unit. The handler refuses
  loudly via `RaiseFailure::unsupportedInstructionForm` with a precise
  diagnostic explaining the cross-target gap, in line with the
  user-rules (no silent fallbacks). The matching LLVM intrinsics
  are `int_amdgcn_tensor_load_to_lds` /
  `int_amdgcn_tensor_store_from_lds` (IntrinsicsAMDGPU.td:4213).
}]>;

def COP_TENSOR_STORE_FROM_LDS : CanonOp;

def COP_GLOBAL_LOAD_ASYNC_TO_LDS_B8 : CanonOp<[{
  -- gfx1250 async global -> LDS load --

  FLAT async global-to-LDS load, four widths per the b8 / b32 / b64 /
  b128 family. Each width has both a plain VGPR_64 vaddr form and a
  SADDR (SReg_64 base + VGPR_32 vaddr offset) form, both of which
  collapse to the same CanonicalOp per width; `handleFLAT` discriminates
  shape on `op.nSrcs()` exactly the same way `tensor_load_to_lds`
  discriminates `_d2` vs `_d4`. The pseudo InOperandList is
  documented in `FLATInstructions.td:391-417`
  (`FLAT_Global_Load_LDS_Pseudo<..., IsAsync=1>`):

  plain : (vdst:VGPR_32, vaddr:VGPR_64,             offset, cpol)
  SADDR : (vdst:VGPR_32, saddr:SReg_64, vaddr:VGPR_32, offset, cpol)

  `vdst` here is the per-lane LDS i32 OFFSET (TableGen `vdst` slot
  is in the *input* list because `IsAsync=1` enables `has_vdst`),
  not a written register: each lane uses its own VGPR_32 value as
  the LDS-base address for the burst write. The intrinsics
  `int_amdgcn_global_load_async_to_lds_b{8,32,64,128}`
  (IntrinsicsAMDGPU.td:3939-3946) all share signature
  `AMDGPUAsyncGlobalLoadToLDS` (line 3904) and take the LDS
  pointer as the second operand (`local_ptr_ty`); we materialise
  it via `inttoptr i32 -> ptr addrspace(3)` from the per-lane
  VGPR_32. The width is encoded only in the intrinsic ID -- the
  operand bank is identical across all four widths.

  Separate CanonicalOps per width (rather than a single
  `GLOBAL_LOAD_ASYNC_TO_LDS_BX` discriminated by mnemonic) so the
  CanonicalOp <-> intrinsic mapping is direct and the handler is a small
  switch instead of string parsing -- the canonical opcode table
  collapses each `_gfx1250` real onto its width-specific pseudo.

  === Same-target gfx1250 -> gfx1250 contract ===

  gfx1250 has the asynccnt unit and the native intrinsic; the
  handler emits a direct call inside an `emitUnderExec` diamond
  (per-lane operation: each lane fires its own LDS write, inactive
  lanes do not). `IntrInaccessibleMemOrArgMemOnly` on the
  intrinsic prevents downstream passes from CSEing or reordering
  the asynchronous fetch across other memory sites -- the
  user-visible barrier semantics live in companion
  `s_wait_asynccnt` instructions, not in this op. The intrinsic's
  `offset` immediate corresponds to the FLAT instruction's
  `flat_offset` slot; `cpol` is the gfx12+ cachepolicy bitfield
  (th, scope) carried as the trailing immediate.

  === Cross-target (gfx942 and earlier) contract ===

  The asynccnt unit and `int_amdgcn_global_load_async_to_lds_b*`
  are gfx1250-only (`SubtargetPredicate = isGFX1250Plus` on the
  VFLAT reals, `FeatureGFX1250Insts`). gfx942 has no asynchronous
  global->LDS DMA channel and no equivalent burst path.

  We emit a **synchronous per-lane emulation**: for each active
  lane, `load <T>, ptr addrspace(1) %gptr` followed by
  `store <T>, ptr addrspace(3) %lptr`, width `T` chosen per the
  b8 / b32 / b64 / b128 CanonicalOp. The source ISA pragma
  (`instruction_manual.pdf sec. 13.6.{9,10,11,12}`, verbatim):

  pragma "vector" do
  dsaddr  = LDS_BASE.b32 + VGPR[laneId][VDST.u32] + INST_OFFSET.b32;
  memaddr = ADDR;  // CalcGlobalAddr(VADDR, SADDR, IOFFSET)
  LDS[dsaddr].bN = MEM[memaddr].bN   // (N = 8/32/64/128)
  endpragma

  -- a per-lane global->LDS copy, width-parametric, identical in
  every respect to what the synchronous `load` + `store` pair
  produces *per lane*. `INST_OFFSET` applies to BOTH the LDS
  address and the global address (confirmed by the explicit
  appearance in the `dsaddr` expression and by `CalcGlobalAddr`
  folding `IOFFSET` into `memaddr`); the emulation folds it onto
  both pointers via `i8`-GEP before the `load`/`store`, matching
  the same-target intrinsic's immarg behaviour.

  === Documented semantic trade-off ===

  The async intrinsic carries `IntrInaccessibleMemOrArgMemOnly`
  and the hardware schedules the DMA against a dedicated counter
  (`ASYNCcnt`; `programming_manual.pdf sec. 4.9.9`). Completion is
  signalled via `S_WAIT_ASYNCCNT`. The same aggregate
  observable per-lane LDS state is produced by a synchronous
  `load`+`store` chain AFTER the corresponding `s_wait_asynccnt
  0`; the ONLY information lost in the emulation is the
  **pipelining overlap** between in-flight async DMAs and
  unrelated VMEM / LDS operations in the wave's own stream. On
  gfx1250 the DMA unit can fire while the wave's ALU path runs;
  the synchronous emulation blocks the wave until the global
  `load` retires before the `store` publishes the data to LDS.

  This is a **throughput regression, not a correctness
  regression** -- every lane's final LDS state is bit-identical
  to the async version's state observed after
  `s_wait_asynccnt 0`. Kernels that depend on *observable
  effects under a partially-elapsed asynccnt* (e.g., a
  hand-written pipeliner polling asynccnt state out of the wave's
  instruction stream -- not a pattern LLVM IR can express anyway)
  are NOT in the GPT-OSS corpus and remain explicitly out of
  scope. The GPT-OSS MoE expert-GEMM kernels (`matmul_ogs_*`)
  use the async DMA as a compiler-scheduled prefetch-into-LDS
  whose only user-visible contract is "data lands in LDS before
  the subsequent `s_wait_asynccnt 0` + `ds_read_*` chain reads
  it" -- which the synchronous emulation preserves exactly.

  Companion `S_WAIT_ASYNCCNT` CanonicalOp (declared above) lowers to a
  raiser-level no-op on the cross-target arm: the synchronous
  `load`+`store` pair has already completed by the time the wait
  is reached, so the native wait has nothing to track.  Dataflow
  dependencies from the emulated `store` through subsequent LDS
  reads carry the happens-before the source kernel relied on, and
  the gfx942 backend re-inserts the `s_waitcnt lgkmcnt(0)` before
  the reader.  On the same-target arm, `S_WAIT_ASYNCCNT` is still
  a no-op at the raiser (like every other wait counter, per
  `handle-sopp.cpp`): the intrinsic's
  `IntrInaccessibleMemOrArgMemOnly` annotation prevents reorder
  across the wait site, and the backend re-emits the native
  `s_wait_asynccnt` from that IR-level ordering.

  Rationale for preferring a documented emulation over the
  previous loud refusal: every `matmul_ogs_*` variant in the
  GPT-OSS MoE expert-GEMM surface (4 kernels, runtime-dominant
  in inference) hits this opcode and was blocked end-to-end.
  The `matmul_fp16` / `matmul_fp16_16x16` path -- the shared MFMA
  fragment redistribution surface -- already works; unblocking the
  MoE GEMM was the single highest-impact change. The trade-off
  is scoped and documented, not hidden, and matches the posture
  the sibling TDM axis takes in `sync-translation.md sec. 10` ("TDM
  emulation lowers to synchronous buffer loads").
}]>;

def COP_GLOBAL_LOAD_ASYNC_TO_LDS_B32 : CanonOp;

def COP_GLOBAL_LOAD_ASYNC_TO_LDS_B64 : CanonOp;

def COP_GLOBAL_LOAD_ASYNC_TO_LDS_B128 : CanonOp;

def COP_GLOBAL_LOAD_TR4_B64 : CanonOp;

def COP_GLOBAL_LOAD_TR6_B96 : CanonOp;

def COP_GLOBAL_LOAD_TR8_B64 : CanonOp;

def COP_GLOBAL_LOAD_TR16_B128 : CanonOp;

def COP_GLOBAL_LOAD_TR_FIRST : CanonOpAlias<COP_GLOBAL_LOAD_TR4_B64> {
  let IsSentinel = 1;
}

def COP_GLOBAL_LOAD_TR_LAST : CanonOpAlias<COP_GLOBAL_LOAD_TR16_B128> {
  let IsSentinel = 1;
}

def COP_GLOBAL_PREFETCH_B8 : CanonOp<[{
  -- gfx1250 VMEM prefetch (FLAT, hint-class) --

  FLAT advisory prefetch on a per-lane (divergent) VGPR pointer. The
  family is gated by `FeatureVmemPrefInsts` (AMDGPU.td:283) and is
  currently only enabled inside `FeatureISAVersion12_50_Common`
  (AMDGPU.td:2092), i.e. gfx1250 / RDNA4 -- no earlier ISA exposes a
  matching VMEM-prefetch encoding. The companion
  `int_amdgcn_global_prefetch` intrinsic (IntrinsicsAMDGPU.td:3211)
  takes a global-address-space pointer + an i32 cachepolicy immarg
  (bits[2:0]=th, bits[4:3]=scope) and is annotated with
  `IntrInaccessibleMemOrArgMemOnly + IntrWillReturn + NoCapture +
  IntrNoCallback + IntrNoFree`, so the SDAG models it as a memory
  intrinsic that may overlap the load lattice but never publishes a
  value -- exactly the hint contract the hardware promises.

  Operand layout (FLAT_Prefetch_Pseudo, FLATInstructions.td:525-553):

  plain (3 srcs): vaddr:VGPR_64,            offset, cpol
  SADDR (4 srcs): saddr:SReg_64, vaddr:VGPR_32, offset, cpol

  Note `has_vdst = 0`, so there is no destination and no LDS slot --
  distinct from the GLOBAL_LOAD_ASYNC_TO_LDS_B* family above. Width
  is fixed at 8 bytes (the `_b8` mnemonic refers to the request
  granularity, not data); a single CanonicalOp suffices for both
  addressing-mode variants because `handleFLAT` discriminates on
  `op.nSrcs()` exactly the same way the async-to-LDS family does.

  === Same-target gfx1250 -> gfx1250 contract ===

  Lift directly to `int_amdgcn_global_prefetch(globalPtr, cpol)`.
  The FLAT `flat_offset` is folded onto the address by GEP'ing
  `globalPtr += offset` before the call (the intrinsic itself
  takes no offset operand). The call sits OUTSIDE
  `emitUnderExec` because the intrinsic carries the EXEC mask
  implicitly through `IntrInaccessibleMemOrArgMemOnly` -- wrapping
  it in a per-lane diamond would emit one prefetch per active
  lane, gratuitously inflating the IR for what the hardware
  executes as a single broadcast hint.

  === Cross-target (gfx942 and earlier) contract ===

  `int_amdgcn_global_prefetch` is gated by `HasVmemPrefInsts`
  (FLATInstructions.td:1367) and has no isel coverage on
  gfx942 -- emitting the call would compile-fail downstream.
  The closest sibling, `int_amdgcn_s_prefetch_data`
  (IntrinsicsAMDGPU.td:3188), requires a UNIFORM (SGPR) pointer,
  which we cannot prove for the divergent VGPR address used here
  without divergence analysis. Per the user-rules (no silent
  fallbacks) and consistent with the gfx1250-only refusal contract
  applied to GLOBAL_LOAD_ASYNC_TO_LDS_B* and TENSOR_LOAD_TO_LDS
  above, we refuse loudly via `RaiseFailure::unsupportedInstructionForm`.
  Triton's TDM-pipelined GEMM kernels schedule prefetches at
  specific points in their software pipeline and a silent drop on
  gfx942 would mask both the cross-target capability gap and any
  performance-tuning regression downstream.
}]>;

def COP_FLAT_PREFETCH_B8 : CanonOp<[{
  -- gfx1250 VMEM flat prefetch (FLAT, hint-class) --

  Same operand layout and lift contract as GLOBAL_PREFETCH_B8
  above.
}]>;

def COP_V_ACCVGPR_READ_B32 : CanonOp<[{
  -- AGPR --
}]>;

def COP_V_ACCVGPR_WRITE_B32 : CanonOp;


//===----------------------------------------------------------------------===//
// MC pseudo -> canonical opcode
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------
// SOPP
// ---------------------------------------------------------------------
def : Canon<S_ENDPGM, COP_S_ENDPGM>;
def : Canon<S_NOP, COP_S_NOP>;
def : Canon<S_BRANCH, COP_S_BRANCH>;
def : Canon<S_CODE_END, COP_S_CODE_END>;
def : Canon<S_CBRANCH_SCC0, COP_S_CBRANCH_SCC0>;
def : Canon<S_CBRANCH_SCC1, COP_S_CBRANCH_SCC1>;
def : Canon<S_CBRANCH_VCCZ, COP_S_CBRANCH_VCCZ>;
def : Canon<S_CBRANCH_VCCNZ, COP_S_CBRANCH_VCCNZ>;
def : Canon<S_CBRANCH_EXECZ, COP_S_CBRANCH_EXECZ>;
def : Canon<S_CBRANCH_EXECNZ, COP_S_CBRANCH_EXECNZ>;
def : Canon<S_WAITCNT, COP_S_WAITCNT>;
def : Canon<S_WAIT_LOADCNT, COP_S_WAIT_LOADCNT>;
def : Canon<S_WAIT_STORECNT, COP_S_WAIT_STORECNT>;
def : Canon<S_WAIT_KMCNT, COP_S_WAIT_KMCNT>;
def : Canon<S_WAIT_DSCNT, COP_S_WAIT_DSCNT>;
def : Canon<S_WAIT_XCNT, COP_S_WAIT_XCNT>;
// gfx1250 async-memory wait counters. Canonicalised here (not
// left as `CanonicalOp::Unknown`) so that:
// * the raiser's generic `Unsupported instruction` diagnostic
// doesn't fire on these wait sites when the companion
// async DMA / TENSOR op has been emulated or lifted; and
// * `handle-sopp.cpp`'s generic SOPP no-op arm explicitly
// covers them.
// See the `S_WAIT_ASYNCCNT` / `S_WAIT_TENSORCNT` CanonicalOp doc
// block in `canonical-op.h` for the cross-target correctness
// argument (IR dataflow carries happens-before after the
// synchronous `load`+`store` emulation completes).
def : Canon<S_WAIT_ASYNCCNT, COP_S_WAIT_ASYNCCNT>;
def : Canon<S_WAIT_TENSORCNT, COP_S_WAIT_TENSORCNT>;
def : Canon<S_WAIT_LOADCNT_DSCNT, COP_S_WAIT_LOADCNT_DSCNT>;
def : Canon<S_WAITCNT_DEPCTR, COP_S_WAIT_ALU>;
def : Canon<S_CLAUSE, COP_S_CLAUSE>;
def : Canon<S_DELAY_ALU, COP_S_DELAY_ALU>;
def : Canon<S_SET_GPR_IDX_ON, COP_S_SET_GPR_IDX_ON>;
def : Canon<S_SET_GPR_IDX_OFF, COP_S_SET_GPR_IDX_OFF>;
def : Canon<S_SETVSKIP, COP_S_SETVSKIP>;
// Barriers. GFX < 12 has a single SOPP `S_BARRIER` pseudo; GFX12+ splits
// it into a SOPP wait and a SOP1 signal (both IMM and M0 forms). All
// `_ISFIRST`/`_INIT`/`_JOIN`/`_LEAVE` variants are intentionally left
// unmapped -- the raiser does not model them yet.
def : Canon<S_BARRIER, COP_S_BARRIER>;
def : Canon<S_BARRIER_WAIT, COP_S_BARRIER_WAIT>;
def : Canon<S_BARRIER_SIGNAL_IMM, COP_S_BARRIER_SIGNAL>;
def : Canon<S_BARRIER_SIGNAL_M0, COP_S_BARRIER_SIGNAL>;
// s_sendmsg / s_sendmsghalt.
def : Canon<S_SENDMSG, COP_S_SENDMSG>;
def : Canon<S_SENDMSGHALT, COP_S_SENDMSGHALT>;
// ---------------------------------------------------------------------
// SMEM scalar loads
// ---------------------------------------------------------------------
defm : CanonSMEM3<"S_LOAD_DWORD", COP_S_LOAD_B32>;
defm : CanonSMEM3<"S_LOAD_DWORDX2", COP_S_LOAD_B64>;
defm : CanonSMEM3<"S_LOAD_DWORDX3", COP_S_LOAD_B96>;
defm : CanonSMEM3<"S_LOAD_DWORDX4", COP_S_LOAD_B128>;
defm : CanonSMEM3<"S_LOAD_DWORDX8", COP_S_LOAD_B256>;
defm : CanonSMEM3<"S_LOAD_DWORDX16", COP_S_LOAD_B512>;
defm : CanonSMEM3<"S_BUFFER_LOAD_DWORD", COP_S_BUFFER_LOAD_B32>;
defm : CanonSMEM3<"S_BUFFER_LOAD_DWORDX2", COP_S_BUFFER_LOAD_B64>;
defm : CanonSMEM3<"S_BUFFER_LOAD_DWORDX3", COP_S_BUFFER_LOAD_B96>;
defm : CanonSMEM3<"S_BUFFER_LOAD_DWORDX4", COP_S_BUFFER_LOAD_B128>;
defm : CanonSMEM3<"S_BUFFER_LOAD_DWORDX8", COP_S_BUFFER_LOAD_B256>;
defm : CanonSMEM3<"S_BUFFER_LOAD_DWORDX16", COP_S_BUFFER_LOAD_B512>;
// gfx12+ scalar narrow loads. All four types expose the full
// IMM/SGPR/SGPR_IMM triad so the existing SMEM3 macro applies
// unchanged. Handler in handle-smem.cpp.
defm : CanonSMEM3<"S_LOAD_U8", COP_S_LOAD_U8>;
defm : CanonSMEM3<"S_LOAD_I8", COP_S_LOAD_I8>;
defm : CanonSMEM3<"S_LOAD_U16", COP_S_LOAD_U16>;
defm : CanonSMEM3<"S_LOAD_I16", COP_S_LOAD_I16>;
defm : CanonSMEM3<"S_STORE_DWORD", COP_S_STORE_B32>;
defm : CanonSMEM3<"S_STORE_DWORDX2", COP_S_STORE_B64>;
defm : CanonSMEM3<"S_STORE_DWORDX4", COP_S_STORE_B128>;
// ---------------------------------------------------------------------
// SOPC
// ---------------------------------------------------------------------
def : Canon<S_CMP_EQ_U32, COP_S_CMP_EQ_U32>;
def : Canon<S_CMP_LG_U32, COP_S_CMP_LG_U32>;
def : Canon<S_CMP_GT_U32, COP_S_CMP_GT_U32>;
def : Canon<S_CMP_GE_U32, COP_S_CMP_GE_U32>;
def : Canon<S_CMP_LT_U32, COP_S_CMP_LT_U32>;
def : Canon<S_CMP_LE_U32, COP_S_CMP_LE_U32>;
def : Canon<S_CMP_EQ_U64, COP_S_CMP_EQ_U64>;
def : Canon<S_CMP_LG_U64, COP_S_CMP_LG_U64>;
def : Canon<S_CMP_EQ_I32, COP_S_CMP_EQ_I32>;
def : Canon<S_CMP_LG_I32, COP_S_CMP_LG_I32>;
def : Canon<S_CMP_GT_I32, COP_S_CMP_GT_I32>;
def : Canon<S_CMP_GE_I32, COP_S_CMP_GE_I32>;
def : Canon<S_CMP_LT_I32, COP_S_CMP_LT_I32>;
def : Canon<S_CMP_LE_I32, COP_S_CMP_LE_I32>;
def : Canon<S_CMP_EQ_F32, COP_S_CMP_EQ_F32>;
def : Canon<S_CMP_LG_F32, COP_S_CMP_LG_F32>;
def : Canon<S_CMP_GT_F32, COP_S_CMP_GT_F32>;
def : Canon<S_CMP_GE_F32, COP_S_CMP_GE_F32>;
def : Canon<S_CMP_LT_F32, COP_S_CMP_LT_F32>;
def : Canon<S_CMP_LE_F32, COP_S_CMP_LE_F32>;
def : Canon<S_CMP_NEQ_F32, COP_S_CMP_NEQ_F32>;
def : Canon<S_CMP_NGT_F32, COP_S_CMP_NGT_F32>;
def : Canon<S_CMP_NGE_F32, COP_S_CMP_NGE_F32>;
def : Canon<S_CMP_NLT_F32, COP_S_CMP_NLT_F32>;
def : Canon<S_CMP_NLE_F32, COP_S_CMP_NLE_F32>;
def : Canon<S_CMP_NLG_F32, COP_S_CMP_NLG_F32>;
def : Canon<S_CMP_O_F32, COP_S_CMP_O_F32>;
def : Canon<S_CMP_U_F32, COP_S_CMP_U_F32>;
def : Canon<S_CMP_EQ_F16, COP_S_CMP_EQ_F16>;
def : Canon<S_CMP_LG_F16, COP_S_CMP_LG_F16>;
def : Canon<S_CMP_GT_F16, COP_S_CMP_GT_F16>;
def : Canon<S_CMP_GE_F16, COP_S_CMP_GE_F16>;
def : Canon<S_CMP_LT_F16, COP_S_CMP_LT_F16>;
def : Canon<S_CMP_LE_F16, COP_S_CMP_LE_F16>;
def : Canon<S_CMP_NEQ_F16, COP_S_CMP_NEQ_F16>;
def : Canon<S_CMP_NGT_F16, COP_S_CMP_NGT_F16>;
def : Canon<S_CMP_NGE_F16, COP_S_CMP_NGE_F16>;
def : Canon<S_CMP_NLT_F16, COP_S_CMP_NLT_F16>;
def : Canon<S_CMP_NLE_F16, COP_S_CMP_NLE_F16>;
def : Canon<S_CMP_NLG_F16, COP_S_CMP_NLG_F16>;
def : Canon<S_CMP_O_F16, COP_S_CMP_O_F16>;
def : Canon<S_CMP_U_F16, COP_S_CMP_U_F16>;
// ---------------------------------------------------------------------
// SOPK
// ---------------------------------------------------------------------
def : Canon<S_MOVK_I32, COP_S_MOVK_I32>;
def : Canon<S_ADDK_I32, COP_S_ADDK_I32>;
def : Canon<S_MULK_I32, COP_S_MULK_I32>;
def : Canon<S_CMPK_GE_I32, COP_S_CMPK_GE_I32>;
def : Canon<S_CMPK_GT_I32, COP_S_CMPK_GT_I32>;
def : Canon<S_CMPK_LE_I32, COP_S_CMPK_LE_I32>;
def : Canon<S_CMPK_LT_I32, COP_S_CMPK_LT_I32>;
def : Canon<S_CMPK_GE_U32, COP_S_CMPK_GE_U32>;
def : Canon<S_CMPK_GT_U32, COP_S_CMPK_GT_U32>;
def : Canon<S_CMPK_LE_U32, COP_S_CMPK_LE_U32>;
def : Canon<S_CMPK_LT_U32, COP_S_CMPK_LT_U32>;
def : Canon<S_CMPK_EQ_I32, COP_S_CMPK_EQ_I32>;
def : Canon<S_CMPK_EQ_U32, COP_S_CMPK_EQ_U32>;
def : Canon<S_CMPK_LG_I32, COP_S_CMPK_LG_I32>;
def : Canon<S_CMPK_LG_U32, COP_S_CMPK_LG_U32>;
def : Canon<S_GETREG_B32, COP_S_GETREG_B32>;
def : Canon<S_SETREG_B32, COP_S_SETREG_B32>;
def : Canon<S_SETREG_IMM32_B32, COP_S_SETREG_IMM32_B32>;
// ---------------------------------------------------------------------
// SOP1
// ---------------------------------------------------------------------
def : Canon<S_MOV_B32, COP_S_MOV_B32>;
def : Canon<S_MOV_B64, COP_S_MOV_B64>;
def : Canon<S_NOT_B32, COP_S_NOT_B32>;
def : Canon<S_NOT_B64, COP_S_NOT_B64>;
def : Canon<S_BREV_B32, COP_S_BREV_B32>;
def : Canon<S_FF0_I32_B32, COP_S_FF0_I32_B32>;
def : Canon<S_FF0_I32_B64, COP_S_FF0_I32_B64>;
def : Canon<S_FF1_I32_B32, COP_S_FF1_I32_B32>;
def : Canon<S_FF1_I32_B64, COP_S_FF1_I32_B64>;
def : Canon<S_BCNT1_I32_B32, COP_S_BCNT1_I32_B32>;
def : Canon<S_BCNT1_I32_B64, COP_S_BCNT1_I32_B64>;
def : Canon<S_FLBIT_I32_B32, COP_S_FLBIT_I32_B32>;
def : Canon<S_FLBIT_I32_B64, COP_S_FLBIT_I32_B64>;
def : Canon<S_FLBIT_I32, COP_S_FLBIT_I32>;
def : Canon<S_FLBIT_I32_I64, COP_S_FLBIT_I32_I64>;
def : Canon<S_SEXT_I32_I8, COP_S_SEXT_I32_I8>;
def : Canon<S_SEXT_I32_I16, COP_S_SEXT_I32_I16>;
def : Canon<S_CVT_F16_F32, COP_S_CVT_F16_F32>;
def : Canon<S_CVT_F32_F16, COP_S_CVT_F32_F16>;
def : Canon<S_CVT_HI_F32_F16, COP_S_CVT_HI_F32_F16>;
def : Canon<S_CVT_F32_U32, COP_S_CVT_F32_U32>;
def : Canon<S_CVT_F32_I32, COP_S_CVT_F32_I32>;
def : Canon<S_CVT_U32_F32, COP_S_CVT_U32_F32>;
def : Canon<S_CVT_I32_F32, COP_S_CVT_I32_F32>;
// Scalar F32-to-F32 integral rounding. LLVM TableGen declares these
// SOP1_F32_Inst forms with fceil/ffloor/ftrunc/froundeven, and the AMD
// ISA manual records 32-bit F32 source and destination operands.
def : Canon<S_CEIL_F32, COP_S_CEIL_F32>;
def : Canon<S_FLOOR_F32, COP_S_FLOOR_F32>;
def : Canon<S_TRUNC_F32, COP_S_TRUNC_F32>;
def : Canon<S_RNDNE_F32, COP_S_RNDNE_F32>;
def : Canon<S_AND_SAVEEXEC_B32, COP_S_AND_SAVEEXEC_B32>;
def : Canon<S_OR_SAVEEXEC_B32, COP_S_OR_SAVEEXEC_B32>;
def : Canon<S_XOR_SAVEEXEC_B32, COP_S_XOR_SAVEEXEC_B32>;
def : Canon<S_ANDN2_SAVEEXEC_B32, COP_S_ANDN2_SAVEEXEC_B32>;
def : Canon<S_ORN2_SAVEEXEC_B32, COP_S_ORN2_SAVEEXEC_B32>;
// GFX9 B64 aliases: handler uses EXEC-width-aware ops, same semantics.
def : Canon<S_AND_SAVEEXEC_B64, COP_S_AND_SAVEEXEC_B32>;
def : Canon<S_OR_SAVEEXEC_B64, COP_S_OR_SAVEEXEC_B32>;
def : Canon<S_XOR_SAVEEXEC_B64, COP_S_XOR_SAVEEXEC_B32>;
def : Canon<S_ANDN2_SAVEEXEC_B64, COP_S_ANDN2_SAVEEXEC_B32>;
def : Canon<S_ORN2_SAVEEXEC_B64, COP_S_ORN2_SAVEEXEC_B32>;
def : Canon<S_GETPC_B64, COP_S_GETPC_B64>;
// gfx1250 asm renames `S_SETPC_B64` to `s_set_pc_i64`
// (SOPInstructions.td:2208 declares the rename via
// `SOP1_Real_gfx1250<0x048, "s_set_pc_i64">`); the LLVM MC opcode
// remains `S_SETPC_B64`. We canonicalise both names onto
// CanonicalOp::S_SET_PC_I64 so the handler dispatches uniformly across
// gfx versions; the new name is the CanonicalOp because that is how the
// corpus surfaces it (gfx1250 disasm) and what the CanonicalOp enum
// comment in canonical-op.h documents.
def : Canon<S_SETPC_B64, COP_S_SET_PC_I64>;
// gfx1250 asm renames `S_SWAPPC_B64` to `s_swap_pc_i64`
// (SOPInstructions.td:336 declares the SOP1_64 with isCall=1;
// line 2311 renames the asm string via
// `SOP1_Real_gfx1250<0x049, "s_swap_pc_i64">`); the LLVM MC
// opcode remains `S_SWAPPC_B64`. We canonicalise both names
// onto CanonicalOp::S_SWAP_PC_I64 so the handler dispatches uniformly
// across gfx versions; the new name is the CanonicalOp because that
// is how the corpus surfaces it (gfx1250 disasm) and what the
// CanonicalOp enum comment in canonical-op.h documents.
def : Canon<S_SWAPPC_B64, COP_S_SWAP_PC_I64>;
def : Canon<S_ADD_PC_I64, COP_S_ADD_PC_I64>;
def : Canon<S_ABS_I32, COP_S_ABS_I32>;
def : Canon<S_SET_VGPR_MSB, COP_S_SET_VGPR_MSB>;
def : Canon<S_BITSET0_B32, COP_S_BITSET0_B32>;
def : Canon<S_BITSET1_B32, COP_S_BITSET1_B32>;
def : Canon<S_BITSET0_B64, COP_S_BITSET0_B64>;
def : Canon<S_BITSET1_B64, COP_S_BITSET1_B64>;
// SOPC bit-test family (SOPInstructions.td:1411-1414).  Each LLVM
// MC opcode is a pseudo that every real-subtarget form
// (S_BITCMP0_B32_gfx{10,11,12,13}, etc.) is canonicalised onto
// during the MC->pseudo step hotswap-tblgen performs, so a
// single entry per variant covers every AMDGPU generation back to
// gfx6.  Writes SCC only -- no scalar destination register -- which
// matches the SOPC compare shape already implemented alongside in
// handle-sopc.cpp.
def : Canon<S_BITCMP0_B32, COP_S_BITCMP0_B32>;
def : Canon<S_BITCMP1_B32, COP_S_BITCMP1_B32>;
def : Canon<S_BITCMP0_B64, COP_S_BITCMP0_B64>;
def : Canon<S_BITCMP1_B64, COP_S_BITCMP1_B64>;
// S_CMOV uses [SCC] but does not write SCC, and has no _e64 form
// (SOP1 has only one encoding). The B64 alias is included even
// though the corpus first surfaced only the B32 form, because
// the B64 form ships in the same gfx1250 SOP1 table at adjacent
// opcode (.td: 0x002 / 0x003) and the handler is symmetric.
def : Canon<S_CMOV_B32, COP_S_CMOV_B32>;
def : Canon<S_CMOV_B64, COP_S_CMOV_B64>;
// ---------------------------------------------------------------------
// SOP2
// ---------------------------------------------------------------------
def : Canon<S_ADD_U32, COP_S_ADD_U32>;
def : Canon<S_ADD_I32, COP_S_ADD_U32>;
def : Canon<S_ADDC_U32, COP_S_ADDC_U32>;
def : Canon<S_SUB_U32, COP_S_SUB_U32>;
def : Canon<S_SUB_I32, COP_S_SUB_U32>;
def : Canon<S_SUBB_U32, COP_S_SUBB_U32>;
// S_ADD_U64 is intentionally mapped below in the gfx12-rename block
// alongside S_SUB_U64 -- both surface as `CanonicalOp::S_{ADD,SUB}_NC_U64`
// to mirror the gfx12 `s_{add,sub}_nc_u64` assembler mnemonics used
// downstream in `handle-sop2.cpp` and the lit fixtures.  A naive
// `S_ADD_U64 -> S_ADD_U64` row used to live here and lost the routing
// race silently back when the table was flattened at runtime
// (keeps-first on a duplicate key), which meant every `s_add_u64` lift
// actually decoded as `CanonicalOp::S_ADD_U64` rather than the
// intended `S_ADD_NC_U64`.  The handler's defensive
// `sop == S_ADD_U64 || sop == S_ADD_NC_U64` disjunct masked the
// miscompile into a "same behavior either way" no-op -- which is
// the exact pattern a diligent reviewer catches as dead code.
// Cleaned up in the same commit that adds this comment.
def : Canon<S_AND_B32, COP_S_AND_B32>;
def : Canon<S_AND_B64, COP_S_AND_B64>;
def : Canon<S_OR_B32, COP_S_OR_B32>;
def : Canon<S_OR_B64, COP_S_OR_B64>;
def : Canon<S_XOR_B32, COP_S_XOR_B32>;
def : Canon<S_XOR_B64, COP_S_XOR_B64>;
def : Canon<S_ANDN2_B32, COP_S_ANDN2_B32>;
def : Canon<S_ANDN2_B64, COP_S_ANDN2_B64>;
def : Canon<S_ORN2_B32, COP_S_ORN2_B32>;
def : Canon<S_ORN2_B64, COP_S_ORN2_B64>;
def : Canon<S_NAND_B32, COP_S_NAND_B32>;
def : Canon<S_NAND_B64, COP_S_NAND_B64>;
def : Canon<S_NOR_B32, COP_S_NOR_B32>;
def : Canon<S_NOR_B64, COP_S_NOR_B64>;
def : Canon<S_XNOR_B32, COP_S_XNOR_B32>;
def : Canon<S_XNOR_B64, COP_S_XNOR_B64>;
def : Canon<S_ABSDIFF_I32, COP_S_ABSDIFF_I32>;
def : Canon<S_LSHL_B32, COP_S_LSHL_B32>;
def : Canon<S_LSHL_B64, COP_S_LSHL_B64>;
def : Canon<S_LSHR_B32, COP_S_LSHR_B32>;
def : Canon<S_LSHR_B64, COP_S_LSHR_B64>;
def : Canon<S_ASHR_I32, COP_S_ASHR_I32>;
def : Canon<S_ASHR_I64, COP_S_ASHR_I64>;
def : Canon<S_MUL_I32, COP_S_MUL_I32>;
def : Canon<S_MUL_HI_U32, COP_S_MUL_HI_U32>;
def : Canon<S_MUL_HI_I32, COP_S_MUL_HI_I32>;
def : Canon<S_MUL_U64, COP_S_MUL_U64>;
def : Canon<S_MUL_F32, COP_S_MUL_F32>;
def : Canon<S_ADD_F32, COP_S_ADD_F32>;
def : Canon<S_SUB_F32, COP_S_SUB_F32>;
def : Canon<S_FMAAK_F32, COP_S_FMAAK_F32>;
def : Canon<S_FMAMK_F32, COP_S_FMAMK_F32>;
def : Canon<S_FMAC_F32, COP_S_FMAC_F32>;
// LLVM's canonical pseudo is `S_{MIN,MAX}_F32` on both gfx11
// (`s_{min,max}_f32`) and gfx12+ (`s_{min,max}_num_f32`, via the
// subtarget-specific real rename). Route it through NUM CanonicalOps so the
// handler cannot be confused with the IEEE-2019 NaN-propagating
// `S_{MINIMUM,MAXIMUM}_F32` family.
def : Canon<S_MAX_F32, COP_S_MAX_NUM_F32>;
def : Canon<S_MIN_F32, COP_S_MIN_NUM_F32>;
// gfx12+ scalar IEEE-2019 NaN-propagating f16/f32 extrema, distinct from
// the NUM family above; lower to `llvm.{maximum,minimum}.{f16,f32}`.
def : Canon<S_MAXIMUM_F16, COP_S_MAXIMUM_F16>;
def : Canon<S_MINIMUM_F16, COP_S_MINIMUM_F16>;
def : Canon<S_MAXIMUM_F32, COP_S_MAXIMUM_F32>;
def : Canon<S_MINIMUM_F32, COP_S_MINIMUM_F32>;
def : Canon<S_BFE_U32, COP_S_BFE_U32>;
def : Canon<S_BFE_I32, COP_S_BFE_I32>;
def : Canon<S_BFE_I64, COP_S_BFE_I64>;
def : Canon<S_BFM_B32, COP_S_BFM_B32>;
def : Canon<S_BFM_B64, COP_S_BFM_B64>;
def : Canon<S_CSELECT_B32, COP_S_CSELECT_B32>;
def : Canon<S_CSELECT_B64, COP_S_CSELECT_B64>;
def : Canon<S_MIN_I32, COP_S_MIN_I32>;
def : Canon<S_MIN_U32, COP_S_MIN_U32>;
def : Canon<S_MAX_I32, COP_S_MAX_I32>;
def : Canon<S_MAX_U32, COP_S_MAX_U32>;
def : Canon<S_PACK_LL_B32_B16, COP_S_PACK_LL_B32_B16>;
def : Canon<S_PACK_LH_B32_B16, COP_S_PACK_LH_B32_B16>;
def : Canon<S_LSHL1_ADD_U32, COP_S_LSHL1_ADD_U32>;
def : Canon<S_LSHL2_ADD_U32, COP_S_LSHL2_ADD_U32>;
def : Canon<S_LSHL3_ADD_U32, COP_S_LSHL3_ADD_U32>;
def : Canon<S_LSHL4_ADD_U32, COP_S_LSHL4_ADD_U32>;
// gfx12 `s_add_nc_u64` (renamed from `s_add_u64` in the
// assembler -- see SOPInstructions.td 2300-ish range, same
// pattern as `s_sub_u64 ... "s_sub_nc_u64"` below).  LLVM's
// pseudo is still `S_ADD_U64`; surface it as
// `CanonicalOp::S_ADD_NC_U64` so the handler dispatch, the lit
// fixtures (`lit_tests/s_sub_nc_u64/` and any future
// `s_add_nc_u64` sibling), and the runtime diagnostics all
// speak the same gfx12+ mnemonic.  This row is the ONLY
// `S_ADD_U64` entry in this file -- an earlier version had a stray
// `S_ADD_U64 -> S_ADD_U64` row in the SOP2 family block above that
// won the keeps-first race and quietly routed every `s_add_u64` lift
// to the wrong CanonicalOp.  hotswap-tblgen now makes a duplicate row
// a build error; see the comment in that block for the audit trail.
def : Canon<S_ADD_U64, COP_S_ADD_NC_U64>;
// gfx12 `s_sub_nc_u64` (renamed from `s_sub_u64` in the
// assembler -- see SOPInstructions.td 2311
// `S_SUB_U64 ... "s_sub_nc_u64"`). LLVM's pseudo is still
// `S_SUB_U64`; surface it as `CanonicalOp::S_SUB_NC_U64` to mirror
// the S_ADD_NC_U64 convention.
def : Canon<S_SUB_U64, COP_S_SUB_NC_U64>;
// ---------------------------------------------------------------------
// VOP1 (canonical form is `_e64` after getVOPe64 collapses e32)
// ---------------------------------------------------------------------
def : Canon<V_MOV_B32_e64, COP_V_MOV_B32>;
def : Canon<V_MOV_B64_e64, COP_V_MOV_B64>;
def : Canon<V_MOV_B64_PSEUDO, COP_V_MOV_B64>;
// V_MOV_B16 is gfx11+ true16 / fake16. The pseudo-alias pass strips the
// `_t16_` and `_fake16_` infixes before lookup so only the bare e64 form
// needs an entry here.
def : Canon<V_MOV_B16_e64, COP_V_MOV_B16>;
def : Canon<V_SWAP_B32, COP_V_SWAP_B32>;
// Register-relative moves. Both e32 and e64 base pseudos are mapped
// (the suffix-stripping canonicalizer collapses subtarget/_dpp forms
// onto these). M0 supplies the relative index; see handle-valu-small-
// ops.cpp for the lowering.
def : Canon<V_MOVRELD_B32_e32, COP_V_MOVRELD_B32>;
def : Canon<V_MOVRELD_B32_e64, COP_V_MOVRELD_B32>;
def : Canon<V_MOVRELS_B32_e32, COP_V_MOVRELS_B32>;
def : Canon<V_MOVRELS_B32_e64, COP_V_MOVRELS_B32>;
def : Canon<V_MOVRELSD_B32_e32, COP_V_MOVRELSD_B32>;
def : Canon<V_MOVRELSD_B32_e64, COP_V_MOVRELSD_B32>;
def : Canon<V_NOP_e64, COP_V_NOP>;
def : Canon<V_NOT_B32_e64, COP_V_NOT_B32>;
def : Canon<V_BFREV_B32_e64, COP_V_BFREV_B32>;
def : Canon<V_FFBH_U32_e64, COP_V_FFBH_U32>;
def : Canon<V_FFBL_B32_e64, COP_V_FFBL_B32>;
def : Canon<V_FFBH_I32_e64, COP_V_FFBH_I32>;
def : Canon<V_PRNG_B32_e64, COP_V_PRNG_B32>;
def : Canon<V_CVT_F32_I32_e64, COP_V_CVT_F32_I32>;
def : Canon<V_CVT_F32_U32_e64, COP_V_CVT_F32_U32>;
def : Canon<V_CVT_I32_F32_e64, COP_V_CVT_I32_F32>;
def : Canon<V_CVT_U32_F32_e64, COP_V_CVT_U32_F32>;
// gfx11+ true16/fake16 v_cvt_u32_u16. Real subtarget opcodes
// (`_t16_e64`, `_fake16_e64`, `_t16_e32`, `_fake16_e32`, plus their
// _dpp/_dpp8/_sdwa variants) canonicalize onto these base pseudos via
// the suffix-stripping pass in hotswap-tblgen.
def : Canon<V_CVT_U32_U16_e32, COP_V_CVT_U32_U16>;
def : Canon<V_CVT_U32_U16_e64, COP_V_CVT_U32_U16>;
def : Canon<V_CVT_F16_F32_e64, COP_V_CVT_F16_F32>;
def : Canon<V_CVT_F32_F16_e64, COP_V_CVT_F32_F16>;
def : Canon<V_CVT_F32_BF16_e64, COP_V_CVT_F32_BF16>;
def : Canon<V_CVT_PK_F32_FP8_e64, COP_V_CVT_PK_F32_FP8>;
def : Canon<V_CVT_PK_F32_BF8_e64, COP_V_CVT_PK_F32_BF8>;
def : Canon<V_CVT_F32_FP8_e64, COP_V_CVT_F32_FP8>;
def : Canon<V_CVT_F32_BF8_e64, COP_V_CVT_F32_BF8>;
// gfx1250-only packed-8 FP4 -> BF16 scaled convert
// (VOP3Instructions.td:1788). The real form
// `V_CVT_SCALE_PK8_BF16_FP4_e64_gfx1250` canonicalises onto the
// `_e64` pseudo via the MC->pseudo step in
// hotswap-tblgen (getMCOpcodeGen-derived table); a single
// entry therefore covers every real subtarget slot.
def : Canon<V_CVT_SCALE_PK8_BF16_FP4_e64, COP_V_CVT_SCALE_PK8_BF16_FP4>;
def : Canon<V_CVT_F32_UBYTE0_e64, COP_V_CVT_F32_UBYTE0>;
def : Canon<V_CVT_F32_UBYTE1_e64, COP_V_CVT_F32_UBYTE1>;
def : Canon<V_CVT_F32_UBYTE2_e64, COP_V_CVT_F32_UBYTE2>;
def : Canon<V_CVT_F32_UBYTE3_e64, COP_V_CVT_F32_UBYTE3>;
// F32<->F64 converts are registered on `_e64` only.  On gfx11+/gfx12 the
// production encodings are VOP1/DPP `_e64` forms; `_e32` slots exist on
// older ISAs (see VOP1Instructions.td) but are not emitted for the gfx1250
// corpus this table targets.  An unmapped `_e32` MC opcode would surface
// as UnsupportedOpcode rather than silently using the wrong profile.
def : Canon<V_CVT_F32_F64_e64, COP_V_CVT_F32_F64>;
def : Canon<V_CVT_F64_F32_e64, COP_V_CVT_F64_F32>;
def : Canon<V_CVT_F64_U32_e64, COP_V_CVT_F64_U32>;
def : Canon<V_CVT_F64_I32_e64, COP_V_CVT_F64_I32>;
def : Canon<V_CVT_U32_F64_e64, COP_V_CVT_U32_F64>;
def : Canon<V_CVT_I32_F64_e64, COP_V_CVT_I32_F64>;
def : Canon<V_RCP_IFLAG_F32_e64, COP_V_RCP_IFLAG_F32>;
def : Canon<V_RCP_F32_e64, COP_V_RCP_F32>;
def : Canon<V_RSQ_F32_e64, COP_V_RSQ_F32>;
def : Canon<V_SQRT_F32_e64, COP_V_SQRT_F32>;
def : Canon<V_EXP_F32_e64, COP_V_EXP_F32>;
def : Canon<V_LOG_F32_e64, COP_V_LOG_F32>;
def : Canon<V_SIN_F32_e64, COP_V_SIN_F32>;
def : Canon<V_COS_F32_e64, COP_V_COS_F32>;
def : Canon<V_FREXP_EXP_I32_F32_e32, COP_V_FREXP_EXP_I32_F32>;
def : Canon<V_FREXP_EXP_I32_F32_e64, COP_V_FREXP_EXP_I32_F32>;
def : Canon<V_FREXP_EXP_I32_F64_e64, COP_V_FREXP_EXP_I32_F64>;
// Targets with native tanh support use `llvm.amdgcn.tanh.*`; targets
// without native support lower through matching OCML entry points.
def : Canon<V_TANH_F32_e64, COP_V_TANH_F32>;
// bf16 transcendentals (t16/fake16 variants collapse onto _e64).
def : Canon<V_COS_BF16_e64, COP_V_COS_BF16>;
def : Canon<V_EXP_BF16_e64, COP_V_EXP_BF16>;
def : Canon<V_LOG_BF16_e64, COP_V_LOG_BF16>;
def : Canon<V_RCP_BF16_e64, COP_V_RCP_BF16>;
def : Canon<V_RSQ_BF16_e64, COP_V_RSQ_BF16>;
def : Canon<V_SIN_BF16_e64, COP_V_SIN_BF16>;
def : Canon<V_SQRT_BF16_e64, COP_V_SQRT_BF16>;
def : Canon<V_TANH_BF16_e64, COP_V_TANH_BF16>;
def : Canon<V_S_EXP_F32_e64, COP_V_S_EXP_F32>;
def : Canon<V_S_LOG_F32_e64, COP_V_S_LOG_F32>;
def : Canon<V_S_RCP_F32_e64, COP_V_S_RCP_F32>;
def : Canon<V_S_RSQ_F32_e64, COP_V_S_RSQ_F32>;
def : Canon<V_S_SQRT_F32_e64, COP_V_S_SQRT_F32>;
def : Canon<V_FLOOR_F32_e64, COP_V_FLOOR_F32>;
def : Canon<V_CEIL_F32_e64, COP_V_CEIL_F32>;
def : Canon<V_FLOOR_F64_e64, COP_V_FLOOR_F64>;
def : Canon<V_CEIL_F64_e64, COP_V_CEIL_F64>;
def : Canon<V_TRUNC_F32_e64, COP_V_TRUNC_F32>;
def : Canon<V_TRUNC_F64_e64, COP_V_TRUNC_F64>;
def : Canon<V_RNDNE_F32_e64, COP_V_RNDNE_F32>;
def : Canon<V_RNDNE_F64_e64, COP_V_RNDNE_F64>;
def : Canon<V_FRACT_F32_e64, COP_V_FRACT_F32>;
def : Canon<V_FRACT_F64_e64, COP_V_FRACT_F64>;
def : Canon<V_FREXP_MANT_F32_e64, COP_V_FREXP_MANT_F32>;
def : Canon<V_FREXP_MANT_F64_e64, COP_V_FREXP_MANT_F64>;
def : Canon<V_READFIRSTLANE_B32, COP_V_READFIRSTLANE_B32>;
def : Canon<V_FLOOR_F16_e64, COP_V_FLOOR_F16>;
// f16 rounding + reciprocal (t16/fake16 variants collapse to _e64).
def : Canon<V_CEIL_F16_e64, COP_V_CEIL_F16>;
def : Canon<V_TRUNC_F16_e64, COP_V_TRUNC_F16>;
def : Canon<V_RNDNE_F16_e64, COP_V_RNDNE_F16>;
def : Canon<V_RCP_F16_e64, COP_V_RCP_F16>;
def : Canon<V_CVT_F16_U16_e64, COP_V_CVT_F16_U16>;
def : Canon<V_CVT_F16_I16_e64, COP_V_CVT_F16_I16>;
def : Canon<V_CVT_U16_F16_e64, COP_V_CVT_U16_F16>;
def : Canon<V_CVT_I16_F16_e64, COP_V_CVT_I16_F16>;
// ---------------------------------------------------------------------
// VOP2 / VOP3
// ---------------------------------------------------------------------
def : Canon<V_ADD_F32_e64, COP_V_ADD_F32>;
def : Canon<V_SUB_F32_e64, COP_V_SUB_F32>;
def : Canon<V_SUBREV_F32_e64, COP_V_SUBREV_F32>;
def : Canon<V_MUL_F32_e64, COP_V_MUL_F32>;
def : Canon<V_MUL_LEGACY_F32_e32, COP_V_MUL_LEGACY_F32>;
def : Canon<V_MUL_LEGACY_F32_e64, COP_V_MUL_LEGACY_F32>;
def : Canon<V_FMAC_F32_e64, COP_V_FMAC_F32>;
def : Canon<V_FMA_F32_e64, COP_V_FMA_F32>;
def : Canon<V_FMAMK_F32, COP_V_FMAMK_F32>;
def : Canon<V_FMAAK_F32, COP_V_FMAAK_F32>;
def : Canon<V_MAX_F32_e64, COP_V_MAX_NUM_F32>;
def : Canon<V_MIN_F32_e64, COP_V_MIN_NUM_F32>;
// gfx11+ IEEE-754 2019 maximum/minimum. These differ from V_MAX_NUM_F32 /
// V_MIN_NUM_F32 in their NaN semantics: maximum/minimum propagate NaN,
// while maxnum/minnum return the non-NaN operand.
def : Canon<V_MAXIMUM_F32_e64, COP_V_MAXIMUM_F32>;
def : Canon<V_MINIMUM_F32_e64, COP_V_MINIMUM_F32>;
// IEEE-754 2019 ternary NaN-propagating max/min (gfx11+/gfx12 only,
// gated by `HasMinimum3Maximum3F32` in AMDGPU.td:194).  Lowered as a
// chain of two 2-source `llvm.maximum`/`llvm.minimum` calls in
// handle-valu.cpp; llc on gfx950 selects v_max3/v_min3 when NaN
// propagation isn't material, or two 2-source ops otherwise.
def : Canon<V_MAXIMUM3_F32_e64, COP_V_MAXIMUM3_F32>;
def : Canon<V_MINIMUM3_F32_e64, COP_V_MINIMUM3_F32>;
// gfx11+/gfx12 IEEE-754 2019 NaN-propagating clamp pair. These are the
// non-.NUM forms at VOP3 opcodes 0x26c / 0x26d, distinct from
// V_MINMAX_NUM_F32's maxnum(minnum(...), ...) semantics.
def : Canon<V_MAXIMUMMINIMUM_F32_e64, COP_V_MAXIMUMMINIMUM_F32>;
def : Canon<V_MINIMUMMAXIMUM_F32_e64, COP_V_MINIMUMMAXIMUM_F32>;
def : Canon<V_DIV_FIXUP_F32_e64, COP_V_DIV_FIXUP_F32>;
def : Canon<V_DIV_FIXUP_F64_e64, COP_V_DIV_FIXUP_F64>;
// f16 div fixup: gfx8 base + gfx9 op_sel base (t16/fake16 collapse onto
// the gfx9 form via the suffix-stripping canonicalizer), like V_FMA_F16.
def : Canon<V_DIV_FIXUP_F16_e64, COP_V_DIV_FIXUP_F16>;
def : Canon<V_DIV_FIXUP_F16_gfx9_e64, COP_V_DIV_FIXUP_F16>;
def : Canon<V_DIV_FMAS_F32_e64, COP_V_DIV_FMAS_F32>;
def : Canon<V_DIV_FMAS_F64_e64, COP_V_DIV_FMAS_F64>;
def : Canon<V_DIV_SCALE_F32_e64, COP_V_DIV_SCALE_F32>;
def : Canon<V_DIV_SCALE_F64_e64, COP_V_DIV_SCALE_F64>;
// LLVM's no-carry 32-bit add/sub pseudos are just `V_ADD_U32` etc.; the
// carry-in-carry-out form is the older `V_ADDC_U32`/`V_SUBB_U32` family.
def : Canon<V_ADD_U32_e64, COP_V_ADD_NC_U32>;
def : Canon<V_BCNT_U32_B32_e64, COP_V_BCNT_U32_B32>;
def : Canon<V_BCNT_U32_B32_e32, COP_V_BCNT_U32_B32>;
def : Canon<V_SUB_U32_e64, COP_V_SUB_NC_U32>;
def : Canon<V_SUBREV_U32_e64, COP_V_SUBREV_NC_U32>;
// GFX9 VOP3-only signed add/sub (saddsat/ssubsat when clamp is set,
// plain add/sub otherwise). Distinct from the legacy GFX10+ NC variants.
def : Canon<V_ADD_I32_e64, COP_V_ADD_I32>;
def : Canon<V_SUB_I32_e64, COP_V_SUB_I32>;
def : Canon<V_ADD_CO_U32_e64, COP_V_ADD_CO_U32>;
def : Canon<V_ADDC_U32_e64, COP_V_ADD_CO_CI_U32>;
def : Canon<V_SUB_CO_U32_e64, COP_V_SUB_CO_U32>;
def : Canon<V_SUBREV_CO_U32_e64, COP_V_SUBREV_CO_U32>;
def : Canon<V_SUBB_U32_e64, COP_V_SUB_CO_CI_U32>;
def : Canon<V_SUBBREV_U32_e64, COP_V_SUBREV_CO_CI_U32>;
def : Canon<V_AND_B32_e64, COP_V_AND_B32>;
def : Canon<V_OR_B32_e64, COP_V_OR_B32>;
def : Canon<V_XOR_B32_e64, COP_V_XOR_B32>;
def : Canon<V_XNOR_B32_e64, COP_V_XNOR_B32>;
def : Canon<V_LSHLREV_B32_e64, COP_V_LSHLREV_B32>;
def : Canon<V_LSHRREV_B32_e64, COP_V_LSHRREV_B32>;
def : Canon<V_ASHRREV_I32_e64, COP_V_ASHRREV_I32>;
def : Canon<V_CNDMASK_B32_e64, COP_V_CNDMASK_B32>;
// V_CNDMASK_B16 is gfx11+ true16 / fake16 and is VOP3-only, so -- unlike
// V_MOV_B16 -- LLVM has NO bare `V_CNDMASK_B16_e64` pseudo for the
// `_t16_`/`_fake16_` alias-strip pass to collapse onto. Map both encoding
// pseudos directly to the shared 16-bit cndmask handler.
def : Canon<V_CNDMASK_B16_t16_e64, COP_V_CNDMASK_B16>;
def : Canon<V_CNDMASK_B16_fake16_e64, COP_V_CNDMASK_B16>;
// true16 16-bit bitwise ops. AND/OR/XOR are VOP3-only with no bare e64
// pseudo (map both encoding variants); V_NOT_B16 has a bare e64 form that
// the alias-strip pass collapses onto.
def : Canon<V_AND_B16_t16_e64, COP_V_AND_B16>;
def : Canon<V_AND_B16_fake16_e64, COP_V_AND_B16>;
def : Canon<V_OR_B16_t16_e64, COP_V_OR_B16>;
def : Canon<V_OR_B16_fake16_e64, COP_V_OR_B16>;
def : Canon<V_XOR_B16_t16_e64, COP_V_XOR_B16>;
def : Canon<V_XOR_B16_fake16_e64, COP_V_XOR_B16>;
def : Canon<V_NOT_B16_e64, COP_V_NOT_B16>;
def : Canon<V_MUL_LO_U32_e64, COP_V_MUL_LO_U32>;
def : Canon<V_MUL_HI_U32_e64, COP_V_MUL_HI_U32>;
def : Canon<V_MUL_HI_I32_e64, COP_V_MUL_HI_I32>;
def : Canon<V_MUL_I32_I24_e64, COP_V_MUL_I32_I24>;
def : Canon<V_MUL_U32_U24_e64, COP_V_MUL_U32_U24>;
def : Canon<V_MUL_HI_U32_U24_e64, COP_V_MUL_HI_U32_U24>;
def : Canon<V_MUL_HI_I32_I24_e64, COP_V_MUL_HI_I32_I24>;
def : Canon<V_MAD_I32_I24_e64, COP_V_MAD_I32_I24>;
def : Canon<V_MAD_I32_I24_e64_gfx11, COP_V_MAD_I32_I24>;
def : Canon<V_MAD_I32_I24_e64_gfx12, COP_V_MAD_I32_I24>;
def : Canon<V_MAD_U32_U24_e64, COP_V_MAD_U32_U24>;
def : Canon<V_MAD_U32_e64, COP_V_MAD_U32>;
def : Canon<V_ADD3_U32_e64, COP_V_ADD3_U32>;
def : Canon<V_LSHL_ADD_U32_e64, COP_V_LSHL_ADD_U32>;
def : Canon<V_ADD_LSHL_U32_e64, COP_V_ADD_LSHL_U32>;
def : Canon<V_LSHL_OR_B32_e64, COP_V_LSHL_OR_B32>;
def : Canon<V_AND_OR_B32_e64, COP_V_AND_OR_B32>;
def : Canon<V_OR3_B32_e64, COP_V_OR3_B32>;
// gfx9+ ternary xor+add (VOP3Instructions.td:684, iselect
// pattern at :831 -- `add(xor(a, b), c)`).
def : Canon<V_XAD_U32_e64, COP_V_XAD_U32>;
// gfx6+ funnel-shift right (VOP3Instructions.td:218,
// gfx11/gfx12 opcode 0x216 at :1787). The `_t16` and
// `_fake16` variants collapse onto V_ALIGNBIT_B32_e64 via
// sameSemanticShape.
def : Canon<V_ALIGNBIT_B32_e64, COP_V_ALIGNBIT_B32>;
// gfx10+ ternary xor (VOP3Instructions.td:1348). VOP3
// ternaries only have the e64 form.
def : Canon<V_XOR3_B32_e64, COP_V_XOR3_B32>;
// VOP3 true16 16-bit no-carry add/sub. Unsigned reals keep the
// V_{ADD,SUB}_NC_U16 pseudo names; signed forms use LLVM's older
// V_{ADD,SUB}_I16 pseudos even when gfx10+ real mnemonics spell them
// v_{add,sub}_nc_i16. Older non-_nc signed real spellings share the same
// no-carry semantic shape through those pseudos. sameSemanticShape
// collapses true16/fake16 variants onto the base pseudo entries below.
def : Canon<V_ADD_NC_U16_e64, COP_V_ADD_NC_U16>;
def : Canon<V_SUB_NC_U16_e64, COP_V_SUB_NC_U16>;
def : Canon<V_ADD_I16_e64, COP_V_ADD_NC_I16>;
def : Canon<V_SUB_I16_e64, COP_V_SUB_NC_I16>;
def : Canon<V_MAD_U16_e64, COP_V_MAD_U16>;
def : Canon<V_MAD_U16_gfx9_e64, COP_V_MAD_U16>;
// gfx1250 add-then-min/max VOP3. The real subtarget opcodes canonicalize
// through this pseudo via AMDGPU::getMCOpcode-derived tables.
def : Canon<V_ADD_MIN_U32_e64, COP_V_ADD_MIN_U32>;
def : Canon<V_ADD_MAX_U32_e64, COP_V_ADD_MAX_U32>;
def : Canon<V_ADD_MIN_I32_e64, COP_V_ADD_MIN_I32>;
def : Canon<V_ADD_MAX_I32_e64, COP_V_ADD_MAX_I32>;
def : Canon<V_BFE_U32_e64, COP_V_BFE_U32>;
def : Canon<V_BFE_I32_e64, COP_V_BFE_I32>;
// gfx6+ VOP3 bit-field insert. Ternary, e64-only (no VOP1/VOP2
// form). Hardware semantic: `dst = (src0 & src1) | (~src0 & src2)`
// -- src0 is the mask, src1 provides bits where mask=1, src2
// provides bits where mask=0. See VOP3Instructions.td
// `AMDGPUbfiPattern`. Same canonical opcode across gfx11/gfx12;
// libdevice's asin lowering emits this as its sign-stitching
// step, which was the real-world coverage gap that motivated
// the handler.
def : Canon<V_BFI_B32_e64, COP_V_BFI_B32>;
def : Canon<V_PERM_B32_e64, COP_V_PERM_B32>;
def : Canon<V_MBCNT_LO_U32_B32_e64, COP_V_MBCNT_LO_U32_B32>;
def : Canon<V_MBCNT_HI_U32_B32_e64, COP_V_MBCNT_HI_U32_B32>;
def : Canon<V_READLANE_B32, COP_V_READLANE_B32>;
def : Canon<V_WRITELANE_B32, COP_V_WRITELANE_B32>;
def : Canon<V_MED3_F32_e64, COP_V_MED3_NUM_F32>;
def : Canon<V_MAX3_F32_e64, COP_V_MAX3_NUM_F32>;
def : Canon<V_MIN3_F32_e64, COP_V_MIN3_NUM_F32>;
// gfx11 V_MINMAX_F32 (opcode 0x25f) and the gfx12-renamed
// V_MINMAX_NUM_F32 (opcode 0x268) share the same minnum-of-
// maxnum semantics; both LLVM pseudos canonicalize onto
// V_MINMAX_F32_e64 so a single mapping suffices.
def : Canon<V_MINMAX_F32_e64, COP_V_MINMAX_NUM_F32>;
def : Canon<V_MAXMIN_F32_e64, COP_V_MAXMIN_NUM_F32>;
def : Canon<V_MAX3_U32_e64, COP_V_MAX3_U32>;
def : Canon<V_MIN3_U32_e64, COP_V_MIN3_U32>;
def : Canon<V_MAX3_I32_e64, COP_V_MAX3_I32>;
def : Canon<V_MIN3_I32_e64, COP_V_MIN3_I32>;
def : Canon<V_MED3_I32_e64, COP_V_MED3_I32>;
def : Canon<V_MAXMIN_I32_e64, COP_V_MAXMIN_I32>;
def : Canon<V_MINMAX_I32_e64, COP_V_MINMAX_I32>;
def : Canon<V_MAXMIN_U32_e64, COP_V_MAXMIN_U32>;
def : Canon<V_MINMAX_U32_e64, COP_V_MINMAX_U32>;
def : Canon<V_MAX3_I16_e64, COP_V_MAX3_I16>;
def : Canon<V_BITOP3_B32_e64, COP_V_BITOP3_B32>;
def : Canon<V_BITOP3_B16_e64, COP_V_BITOP3_B16>;
def : Canon<V_FMA_MIX_F32, COP_V_FMA_MIX_F32>;
// gfx9.5+/gfx1250 BF16-narrow sibling (VOP3PInstructions.td:109).
// Same VOP3P operand shape (3 srcs + op_sel / op_sel_hi modifiers)
// as V_FMA_MIX_F32; only the narrow-half element type changes
// (bf16 vs f16). The real form V_FMA_MIX_F32_BF16_gfx1250 is
// canonicalized onto this pseudo via the MC->pseudo step in
// `canonicalize` (getMCOpcode-derived table), mirroring the
// V_FMA_MIX_F32 entry above.
def : Canon<V_FMA_MIX_F32_BF16, COP_V_FMA_MIX_F32_BF16>;
// F16-result mixed FMA siblings. Same source-selection modifiers as
// V_FMA_MIX_F32, but the f32 FMA result is rounded to F16 and written
// to one half of the tied output register.
def : Canon<V_FMA_MIXLO_F16, COP_V_FMA_MIXLO_F16>;
def : Canon<V_FMA_MIXHI_F16, COP_V_FMA_MIXHI_F16>;
// gfx1250 BF16-result mixed FMA siblings. Same source-selection modifiers
// as V_FMA_MIX_F32_BF16, but the f32 FMA result is rounded to BF16 and
// written to one half of the tied output register.
def : Canon<V_FMA_MIXLO_BF16, COP_V_FMA_MIXLO_BF16>;
def : Canon<V_FMA_MIXHI_BF16, COP_V_FMA_MIXHI_BF16>;
def : Canon<V_ADD_F16_e64, COP_V_ADD_F16>;
def : Canon<V_MUL_F16_e64, COP_V_MUL_F16>;
def : Canon<V_SUB_F16_e64, COP_V_SUB_F16>;
def : Canon<V_SUBREV_F16_e64, COP_V_SUBREV_F16>;
def : Canon<V_MAC_F16_e64, COP_V_MAC_F16>;
def : Canon<V_MADMK_F16, COP_V_MADMK_F16>;
def : Canon<V_MADAK_F16, COP_V_MADAK_F16>;
def : Canon<V_FMAC_F16_e64, COP_V_FMAC_F16>;
// VOP3 F16 explicit-source fused multiply-add. LLVM emits two parallel
// pseudos because gfx9 introduced op_sel on the original gfx8 V_FMA_F16:
// V_FMA_F16_e64        -- gfx8 ("vi") form (no op_sel)
// V_FMA_F16_gfx9_e64   -- gfx9+ op_sel form (also the base for t16/
// fake16 collapse below)
// Both map to the same llvm.fma.f16 lowering; the gfx9 op_sel handling
// lives in the handler. The `_t16_` / `_fake16_` 16-bit encoding
// variants on top of the gfx9 form are folded onto V_FMA_F16_gfx9_e64
// by the sameSemanticShape stripping pass in `buildPseudoAliasMap`.
def : Canon<V_FMA_F16_e64, COP_V_FMA_F16>;
def : Canon<V_FMA_F16_gfx9_e64, COP_V_FMA_F16>;
def : Canon<V_MAX_F16_e64, COP_V_MAX_NUM_F16>;
def : Canon<V_MIN_F16_e64, COP_V_MIN_NUM_F16>;
def : Canon<V_MINMAX_F16_e64, COP_V_MINMAX_NUM_F16>;
def : Canon<V_MAXMIN_F16_e64, COP_V_MAXMIN_NUM_F16>;
def : Canon<V_MAXIMUM_F16_e64, COP_V_MAXIMUM_F16>;
def : Canon<V_MINIMUM_F16_e64, COP_V_MINIMUM_F16>;
def : Canon<V_MAXIMUM3_F16_e64, COP_V_MAXIMUM3_F16>;
def : Canon<V_MINIMUM3_F16_e64, COP_V_MINIMUM3_F16>;
def : Canon<V_MAXIMUMMINIMUM_F16_e64, COP_V_MAXIMUMMINIMUM_F16>;
def : Canon<V_MINIMUMMAXIMUM_F16_e64, COP_V_MINIMUMMAXIMUM_F16>;
def : Canon<V_LDEXP_F16_e64, COP_V_LDEXP_F16>;
def : Canon<V_TANH_F16_e64, COP_V_TANH_F16>;
def : Canon<V_LDEXP_F32_e64, COP_V_LDEXP_F32>;
def : Canon<V_DOT2C_I32_I16_e64, COP_V_DOT2C_I32_I16>;
def : Canon<V_DOT4C_I32_I8_e64, COP_V_DOT4C_I32_I8>;
def : Canon<V_DOT8C_I32_I4_e64, COP_V_DOT8C_I32_I4>;
def : Canon<V_DOT4_I32_IU8, COP_V_DOT4_I32_IU8>;
def : Canon<V_DOT4_I32_IU8_gfx11, COP_V_DOT4_I32_IU8>;
def : Canon<V_DOT4_I32_IU8_gfx12, COP_V_DOT4_I32_IU8>;
def : Canon<V_PK_FMAC_F16_e64, COP_V_PK_FMAC_F16>;
def : Canon<V_MAX_U16_e64, COP_V_MAX_U16>;
def : Canon<V_MIN_U16_e64, COP_V_MIN_U16>;
def : Canon<V_MAX_I16_e64, COP_V_MAX_I16>;
def : Canon<V_MIN_I16_e64, COP_V_MIN_I16>;
def : Canon<V_ADD_U16_e64, COP_V_ADD_U16>;
def : Canon<V_SUB_U16_e64, COP_V_SUB_U16>;
def : Canon<V_SUBREV_U16_e64, COP_V_SUBREV_U16>;
def : Canon<V_MUL_LO_U16_e64, COP_V_MUL_LO_U16>;
def : Canon<V_ASHRREV_I16_e64, COP_V_ASHRREV_I16>;
def : Canon<V_LSHRREV_B16_e64, COP_V_LSHRREV_B16>;
def : Canon<V_LSHLREV_B16_e64, COP_V_LSHLREV_B16>;
def : Canon<V_PACK_B32_F16_e64, COP_V_PACK_B32_F16>;
def : Canon<V_CVT_PK_BF16_F32_e64, COP_V_CVT_PK_BF16_F32>;
def : Canon<V_CVT_PK_BF8_F32_e64, COP_V_CVT_PK_BF8_F32>;
def : Canon<V_CVT_PK_FP8_F32_e64, COP_V_CVT_PK_FP8_F32>;
def : Canon<V_CVT_PKRTZ_F16_F32_e64, COP_V_CVT_PKRTZ_F16_F32>;
def : Canon<V_CVT_PK_F16_F32_e64, COP_V_CVT_PK_F16_F32>;
def : Canon<V_CVT_SCALEF32_PK_FP4_F32_e64, COP_V_CVT_SCALEF32_PK_FP4_F32>;
// gfx1250-only packed-8 scaled FP8 conversion (VOP3, opcode 0x2c3,
// VOP3Instructions.td:2384).  Same-target maps to native intrinsic
// `int_amdgcn_cvt_scalef32_pk8_fp8_f32`; cross-target paths with FP8
// conversion support use the software emulation in handle-valu.cpp.
def : Canon<V_CVT_SCALEF32_PK8_FP8_F32_e64, COP_V_CVT_SCALEF32_PK8_FP8_F32>;
def : Canon<V_BFM_B32_e64, COP_V_BFM_B32>;
// ---------------------------------------------------------------------
// FP64
// ---------------------------------------------------------------------
def : Canon<V_ADD_F64_e64, COP_V_ADD_F64>;
def : Canon<V_MUL_F64_e64, COP_V_MUL_F64>;
def : Canon<V_FMA_F64_e64, COP_V_FMA_F64>;
def : Canon<V_FMAC_F64_e64, COP_V_FMAC_F64>;
// VOP2-only mandatory-literal FMA forms. The pseudos have no e32/e64
// split (analogous to V_FMAMK_F32 / V_FMAAK_F32 above); subtarget reals
// (V_FMAMK_F64_gfx1250 / V_FMAAK_F64_gfx1250) collapse back via
// hotswap-tblgen.
def : Canon<V_FMAMK_F64, COP_V_FMAMK_F64>;
def : Canon<V_FMAAK_F64, COP_V_FMAAK_F64>;
def : Canon<V_MAX_NUM_F64_e64, COP_V_MAX_NUM_F64>;
def : Canon<V_MIN_NUM_F64_e64, COP_V_MIN_NUM_F64>;
def : Canon<V_MAXIMUM_F64_e64, COP_V_MAXIMUM_F64>;
def : Canon<V_MINIMUM_F64_e64, COP_V_MINIMUM_F64>;
// V_RCP_F64 has no DPP (`VOP1_Real_..._NO_DPP_...`); the e32
// form is collapsed to e64 by getVOPe64 before lookup, so a
// single e64 entry covers both encodings.
def : Canon<V_RCP_F64_e64, COP_V_RCP_F64>;
def : Canon<V_RSQ_F64_e64, COP_V_RSQ_F64>;
def : Canon<V_SQRT_F64_e64, COP_V_SQRT_F64>;
def : Canon<V_LDEXP_F64_e64, COP_V_LDEXP_F64>;
def : Canon<V_TRIG_PREOP_F64_e64, COP_V_TRIG_PREOP_F64>;
// ---------------------------------------------------------------------
// More VOP3 32-bit int min/max and lane perms
// ---------------------------------------------------------------------
def : Canon<V_MAX_U32_e64, COP_V_MAX_U32>;
def : Canon<V_MIN_U32_e64, COP_V_MIN_U32>;
def : Canon<V_MAX_I32_e64, COP_V_MAX_I32>;
def : Canon<V_MIN_I32_e64, COP_V_MIN_I32>;
def : Canon<V_PERMLANE16_B32_e64, COP_V_PERMLANE16_B32>;
def : Canon<V_PERMLANEX16_B32_e64, COP_V_PERMLANEX16_B32>;
def : Canon<V_PERMLANE64_B32, COP_V_PERMLANE64_B32>;
def : Canon<V_PERMLANE16_SWAP_B32_e32, COP_V_PERMLANE16_SWAP_B32>;
def : Canon<V_PERMLANE16_SWAP_B32_e64, COP_V_PERMLANE16_SWAP_B32>;
def : Canon<V_PERMLANE32_SWAP_B32_e32, COP_V_PERMLANE32_SWAP_B32>;
def : Canon<V_PERMLANE32_SWAP_B32_e64, COP_V_PERMLANE32_SWAP_B32>;
// ---------------------------------------------------------------------
// VOPC (V_CMP_* and V_CMPX_*) are NOT enumerated here. They're picked up
// by scanning canonical pseudo names in hotswap-tblgen and routed to
// the two collapsed CanonicalOps (V_CMP / V_CMPX) with metadata (predicate,
// element width, int/float) in the `vcmp_` side table.
//
// ---------------------------------------------------------------------
// VOP3P (canonical is bare pseudo name, no _e64)
// ---------------------------------------------------------------------
def : Canon<V_PK_ADD_F32, COP_V_PK_ADD_F32>;
def : Canon<V_PK_MUL_F32, COP_V_PK_MUL_F32>;
def : Canon<V_PK_FMA_F32, COP_V_PK_FMA_F32>;
def : Canon<V_PK_FMA_F16, COP_V_PK_FMA_F16>;
def : Canon<V_PK_ADD_F16, COP_V_PK_ADD_F16>;
def : Canon<V_PK_MUL_F16, COP_V_PK_MUL_F16>;
// gfx1250-only packed BF16 arithmetic. The `_gfx1250` real opcodes
// collapse to these bare pseudos via the MC->pseudo map, matching the
// existing packed BF16 TableGen family.
def : Canon<V_PK_ADD_BF16, COP_V_PK_ADD_BF16>;
def : Canon<V_PK_MUL_BF16, COP_V_PK_MUL_BF16>;
def : Canon<V_PK_MIN_NUM_BF16, COP_V_PK_MIN_NUM_BF16>;
def : Canon<V_PK_MAX_NUM_BF16, COP_V_PK_MAX_NUM_BF16>;
def : Canon<V_PK_FMA_BF16, COP_V_PK_FMA_BF16>;
// LLVM has no `V_PK_MAX_F32`/`V_PK_MIN_F32` pseudo (only F16 variants);
// leave the matching CanonicalOps unmapped until one appears.
def : Canon<V_PK_MOV_B32, COP_V_PK_MOV_B32>;
// Packed `<2 x i16>` int family.
def : Canon<V_PK_MAD_U16, COP_V_PK_MAD_U16>;
def : Canon<V_PK_ADD_U16, COP_V_PK_ADD_U16>;
def : Canon<V_PK_SUB_I16, COP_V_PK_SUB_I16>;
def : Canon<V_PK_LSHLREV_B16, COP_V_PK_LSHLREV_B16>;
def : Canon<V_PK_LSHRREV_B16, COP_V_PK_LSHRREV_B16>;
def : Canon<V_PK_ASHRREV_I16, COP_V_PK_ASHRREV_I16>;
def : Canon<V_PK_MUL_LO_U16, COP_V_PK_MUL_LO_U16>;
def : Canon<V_PK_MAX_I16, COP_V_PK_MAX_I16>;
def : Canon<V_PK_MAX3_I16, COP_V_PK_MAX3_I16>;
// ---------------------------------------------------------------------
// 64-bit vector
// ---------------------------------------------------------------------
def : Canon<V_LSHLREV_B64_e64, COP_V_LSHLREV_B64>;
// V_LSHRREV_B64 / V_ASHRREV_I64 -- gfx8+ 64-bit logical / arithmetic
// right shift with reversed operands (`dst = src1 >> shamt`). Only
// the plain `_e64` pseudo exists; per-ISA realtriples (gfx10 / gfx11
// / gfx1250) all collapse back to it through the disassembler's
// pseudo-alias step (TableGen treats the gfx1250 form as a `_e64`
// realtriple, not a `_t16_e64` half-precision variant -- see
// VOP3Instructions.td:2341 vs the b16 family at 2336-2340).
def : Canon<V_LSHRREV_B64_e64, COP_V_LSHRREV_B64>;
def : Canon<V_ASHRREV_I64_e64, COP_V_ASHRREV_I64>;
def : Canon<V_LSHL_ADD_U64_e64, COP_V_LSHL_ADD_U64>;
// LLVM's 64-bit no-carry add pseudo is simply `V_ADD_U64_e64`.
def : Canon<V_ADD_U64_e64, COP_V_ADD_NC_U64>;
// gfx1250 VOP2 64-bit no-carry subtract. LLVM's real opcode is
// `V_SUB_U64_e64` while the printed ISA mnemonic is `v_sub_nc_u64`;
// route by opcode, not by mnemonic text. VOP2Instructions.td marks it
// non-commutable and SIInstructions.td gives the i64 `src0 - src1`
// pattern, so the handler must preserve operand order exactly.
def : Canon<V_SUB_U64_e64, COP_V_SUB_NC_U64>;
// gfx1250 VOP3 64-bit integer min/max.  VOP3Instructions.td attaches
// smax/umax/smin/umin patterns respectively; the handler mirrors the
// manual's compare-and-select semantics on the full i64 lane value.
def : Canon<V_MAX_I64_e64, COP_V_MAX_I64>;
def : Canon<V_MAX_U64_e64, COP_V_MAX_U64>;
def : Canon<V_MIN_I64_e64, COP_V_MIN_I64>;
def : Canon<V_MIN_U64_e64, COP_V_MIN_U64>;
// gfx1250 VOP2 V_MUL_U64; canonicalizer collapses `_gfx1250` suffix
// onto the bare `V_MUL_U64_e64` pseudo.
def : Canon<V_MUL_U64_e64, COP_V_MUL_U64>;
def : Canon<V_MAD_U64_U32_e64, COP_V_MAD_U64_U32>;
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
def : Canon<V_MAD_NC_U64_U32_e64, COP_V_MAD_NC_U64_U32>;
def : Canon<V_MAD_NC_I64_I32_e64, COP_V_MAD_NC_I64_I32>;
// ---------------------------------------------------------------------
// FLAT
// ---------------------------------------------------------------------
def : Canon<FLAT_LOAD_UBYTE, COP_FLAT_LOAD_UBYTE>;
def : Canon<FLAT_LOAD_SBYTE, COP_FLAT_LOAD_SBYTE>;
def : Canon<FLAT_LOAD_USHORT, COP_FLAT_LOAD_USHORT>;
def : Canon<FLAT_LOAD_SSHORT, COP_FLAT_LOAD_SSHORT>;
def : Canon<FLAT_LOAD_DWORD, COP_FLAT_LOAD_DWORD>;
def : Canon<FLAT_LOAD_DWORDX2, COP_FLAT_LOAD_DWORDX2>;
def : Canon<FLAT_LOAD_DWORDX3, COP_FLAT_LOAD_DWORDX3>;
def : Canon<FLAT_LOAD_DWORDX4, COP_FLAT_LOAD_DWORDX4>;
def : Canon<FLAT_STORE_BYTE, COP_FLAT_STORE_BYTE>;
def : Canon<FLAT_STORE_BYTE_D16_HI, COP_FLAT_STORE_BYTE_D16_HI>;
def : Canon<FLAT_STORE_SHORT, COP_FLAT_STORE_SHORT>;
def : Canon<FLAT_STORE_SHORT_D16_HI, COP_FLAT_STORE_SHORT_D16_HI>;
def : Canon<FLAT_STORE_DWORD, COP_FLAT_STORE_DWORD>;
def : Canon<FLAT_STORE_DWORDX2, COP_FLAT_STORE_DWORDX2>;
def : Canon<FLAT_STORE_DWORDX3, COP_FLAT_STORE_DWORDX3>;
def : Canon<FLAT_STORE_DWORDX4, COP_FLAT_STORE_DWORDX4>;
def : Canon<GLOBAL_LOAD_UBYTE, COP_GLOBAL_LOAD_UBYTE>;
def : Canon<GLOBAL_LOAD_SBYTE, COP_GLOBAL_LOAD_SBYTE>;
def : Canon<GLOBAL_LOAD_USHORT, COP_GLOBAL_LOAD_USHORT>;
def : Canon<GLOBAL_LOAD_SSHORT, COP_GLOBAL_LOAD_SSHORT>;
def : Canon<GLOBAL_LOAD_SHORT_D16_HI, COP_GLOBAL_LOAD_SHORT_D16_HI>;
def : Canon<GLOBAL_LOAD_DWORD, COP_GLOBAL_LOAD_DWORD>;
def : Canon<GLOBAL_LOAD_DWORDX2, COP_GLOBAL_LOAD_DWORDX2>;
def : Canon<GLOBAL_LOAD_DWORDX3, COP_GLOBAL_LOAD_DWORDX3>;
def : Canon<GLOBAL_LOAD_DWORDX4, COP_GLOBAL_LOAD_DWORDX4>;
def : Canon<GLOBAL_STORE_BYTE, COP_GLOBAL_STORE_BYTE>;
def : Canon<GLOBAL_STORE_SHORT, COP_GLOBAL_STORE_SHORT>;
def : Canon<GLOBAL_STORE_BYTE_D16_HI, COP_GLOBAL_STORE_BYTE_D16_HI>;
def : Canon<GLOBAL_STORE_SHORT_D16_HI, COP_GLOBAL_STORE_SHORT_D16_HI>;
def : Canon<GLOBAL_STORE_DWORD, COP_GLOBAL_STORE_DWORD>;
def : Canon<GLOBAL_STORE_DWORDX2, COP_GLOBAL_STORE_DWORDX2>;
def : Canon<GLOBAL_STORE_DWORDX3, COP_GLOBAL_STORE_DWORDX3>;
def : Canon<GLOBAL_STORE_DWORDX4, COP_GLOBAL_STORE_DWORDX4>;
def : Canon<GLOBAL_WB, COP_GLOBAL_WB>;
def : Canon<GLOBAL_INV, COP_GLOBAL_INV>;
// Scratch/private-segment VMEM. Keep these as explicit CanonicalOps rather
// than
// routing through GLOBAL_*: the handler must preserve scratch swizzling and
// KD private-segment ABI semantics, not global address-space semantics.
def : Canon<SCRATCH_LOAD_DWORD_ST, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SADDR, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SVS, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORDX2_ST, COP_SCRATCH_LOAD_DWORDX2>;
def : Canon<SCRATCH_LOAD_DWORDX2_SADDR, COP_SCRATCH_LOAD_DWORDX2>;
def : Canon<SCRATCH_LOAD_DWORDX2, COP_SCRATCH_LOAD_DWORDX2>;
def : Canon<SCRATCH_LOAD_DWORDX2_SVS, COP_SCRATCH_LOAD_DWORDX2>;
def : Canon<SCRATCH_LOAD_DWORDX3_ST, COP_SCRATCH_LOAD_DWORDX3>;
def : Canon<SCRATCH_LOAD_DWORDX3_SADDR, COP_SCRATCH_LOAD_DWORDX3>;
def : Canon<SCRATCH_LOAD_DWORDX3, COP_SCRATCH_LOAD_DWORDX3>;
def : Canon<SCRATCH_LOAD_DWORDX3_SVS, COP_SCRATCH_LOAD_DWORDX3>;
def : Canon<SCRATCH_LOAD_DWORDX4_ST, COP_SCRATCH_LOAD_DWORDX4>;
def : Canon<SCRATCH_LOAD_DWORDX4_SADDR, COP_SCRATCH_LOAD_DWORDX4>;
def : Canon<SCRATCH_LOAD_DWORDX4, COP_SCRATCH_LOAD_DWORDX4>;
def : Canon<SCRATCH_LOAD_DWORDX4_SVS, COP_SCRATCH_LOAD_DWORDX4>;
def : Canon<SCRATCH_LOAD_DWORD_ST_gfx11, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_ST_gfx12, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_ST_gfx13, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SADDR_gfx11, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SADDR_gfx12, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SADDR_gfx13, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_gfx11, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_gfx12, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_gfx13, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SVS_gfx11, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SVS_gfx12, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_LOAD_DWORD_SVS_gfx13, COP_SCRATCH_LOAD_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_ST, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SADDR, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SVS, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORDX2_ST, COP_SCRATCH_STORE_DWORDX2>;
def : Canon<SCRATCH_STORE_DWORDX2_SADDR, COP_SCRATCH_STORE_DWORDX2>;
def : Canon<SCRATCH_STORE_DWORDX2, COP_SCRATCH_STORE_DWORDX2>;
def : Canon<SCRATCH_STORE_DWORDX2_SVS, COP_SCRATCH_STORE_DWORDX2>;
def : Canon<SCRATCH_STORE_DWORDX3_ST, COP_SCRATCH_STORE_DWORDX3>;
def : Canon<SCRATCH_STORE_DWORDX3_SADDR, COP_SCRATCH_STORE_DWORDX3>;
def : Canon<SCRATCH_STORE_DWORDX3, COP_SCRATCH_STORE_DWORDX3>;
def : Canon<SCRATCH_STORE_DWORDX3_SVS, COP_SCRATCH_STORE_DWORDX3>;
def : Canon<SCRATCH_STORE_DWORDX4_ST, COP_SCRATCH_STORE_DWORDX4>;
def : Canon<SCRATCH_STORE_DWORDX4_SADDR, COP_SCRATCH_STORE_DWORDX4>;
def : Canon<SCRATCH_STORE_DWORDX4, COP_SCRATCH_STORE_DWORDX4>;
def : Canon<SCRATCH_STORE_DWORDX4_SVS, COP_SCRATCH_STORE_DWORDX4>;
def : Canon<SCRATCH_STORE_DWORD_ST_gfx11, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_ST_gfx12, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_ST_gfx13, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SADDR_gfx11, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SADDR_gfx12, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SADDR_gfx13, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_gfx11, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_gfx12, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_gfx13, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SVS_gfx11, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SVS_gfx12, COP_SCRATCH_STORE_DWORD>;
def : Canon<SCRATCH_STORE_DWORD_SVS_gfx13, COP_SCRATCH_STORE_DWORD>;
// Sub-dword scratch load/store (byte/short). Same addressing-mode
// variants (ST/SADDR/SVS + gfx11/12/13 reals) as the DWORD family above;
// canonicalize() folds the subtarget reals onto these pseudos.
def : Canon<SCRATCH_LOAD_UBYTE_ST, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SADDR, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SVS, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_ST_gfx11, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SADDR_gfx11, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_gfx11, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SVS_gfx11, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_ST_gfx12, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SADDR_gfx12, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_gfx12, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SVS_gfx12, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_ST_gfx13, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SADDR_gfx13, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_gfx13, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_UBYTE_SVS_gfx13, COP_SCRATCH_LOAD_UBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_ST, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SADDR, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SVS, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_ST_gfx11, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SADDR_gfx11, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_gfx11, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SVS_gfx11, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_ST_gfx12, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SADDR_gfx12, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_gfx12, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SVS_gfx12, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_ST_gfx13, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SADDR_gfx13, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_gfx13, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_SBYTE_SVS_gfx13, COP_SCRATCH_LOAD_SBYTE>;
def : Canon<SCRATCH_LOAD_USHORT_ST, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SADDR, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SVS, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_ST_gfx11, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SADDR_gfx11, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_gfx11, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SVS_gfx11, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_ST_gfx12, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SADDR_gfx12, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_gfx12, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SVS_gfx12, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_ST_gfx13, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SADDR_gfx13, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_gfx13, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_USHORT_SVS_gfx13, COP_SCRATCH_LOAD_USHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_ST, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SADDR, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SVS, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_ST_gfx11, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SADDR_gfx11, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_gfx11, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SVS_gfx11, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_ST_gfx12, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SADDR_gfx12, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_gfx12, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SVS_gfx12, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_ST_gfx13, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SADDR_gfx13, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_gfx13, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_LOAD_SSHORT_SVS_gfx13, COP_SCRATCH_LOAD_SSHORT>;
def : Canon<SCRATCH_STORE_BYTE_ST, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SADDR, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SVS, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_ST_gfx11, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SADDR_gfx11, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_gfx11, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SVS_gfx11, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_ST_gfx12, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SADDR_gfx12, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_gfx12, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SVS_gfx12, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_ST_gfx13, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SADDR_gfx13, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_gfx13, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_BYTE_SVS_gfx13, COP_SCRATCH_STORE_BYTE>;
def : Canon<SCRATCH_STORE_SHORT_ST, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SADDR, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SVS, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_ST_gfx11, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SADDR_gfx11, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_gfx11, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SVS_gfx11, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_ST_gfx12, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SADDR_gfx12, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_gfx12, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SVS_gfx12, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_ST_gfx13, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SADDR_gfx13, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_gfx13, COP_SCRATCH_STORE_SHORT>;
def : Canon<SCRATCH_STORE_SHORT_SVS_gfx13, COP_SCRATCH_STORE_SHORT>;
// ---------------------------------------------------------------------
// FLAT atomics (canonicalized to non-SADDR via getGlobalVaddrOp; no-return
// form)
// ---------------------------------------------------------------------
def : Canon<FLAT_ATOMIC_ADD, COP_FLAT_ATOMIC_ADD>;
def : Canon<FLAT_ATOMIC_SUB, COP_FLAT_ATOMIC_SUB>;
def : Canon<FLAT_ATOMIC_AND, COP_FLAT_ATOMIC_AND>;
def : Canon<FLAT_ATOMIC_OR, COP_FLAT_ATOMIC_OR>;
def : Canon<FLAT_ATOMIC_XOR, COP_FLAT_ATOMIC_XOR>;
def : Canon<FLAT_ATOMIC_SMIN, COP_FLAT_ATOMIC_SMIN>;
def : Canon<FLAT_ATOMIC_SMAX, COP_FLAT_ATOMIC_SMAX>;
def : Canon<FLAT_ATOMIC_UMIN, COP_FLAT_ATOMIC_UMIN>;
def : Canon<FLAT_ATOMIC_UMAX, COP_FLAT_ATOMIC_UMAX>;
def : Canon<FLAT_ATOMIC_SWAP, COP_FLAT_ATOMIC_SWAP>;
def : Canon<FLAT_ATOMIC_CMPSWAP, COP_FLAT_ATOMIC_CMPSWAP>;
def : Canon<FLAT_ATOMIC_ADD_X2, COP_FLAT_ATOMIC_ADD_X2>;
def : Canon<FLAT_ATOMIC_SUB_X2, COP_FLAT_ATOMIC_SUB_X2>;
def : Canon<FLAT_ATOMIC_AND_X2, COP_FLAT_ATOMIC_AND_X2>;
def : Canon<FLAT_ATOMIC_OR_X2, COP_FLAT_ATOMIC_OR_X2>;
def : Canon<FLAT_ATOMIC_XOR_X2, COP_FLAT_ATOMIC_XOR_X2>;
def : Canon<FLAT_ATOMIC_SMIN_X2, COP_FLAT_ATOMIC_SMIN_X2>;
def : Canon<FLAT_ATOMIC_SMAX_X2, COP_FLAT_ATOMIC_SMAX_X2>;
def : Canon<FLAT_ATOMIC_UMIN_X2, COP_FLAT_ATOMIC_UMIN_X2>;
def : Canon<FLAT_ATOMIC_UMAX_X2, COP_FLAT_ATOMIC_UMAX_X2>;
def : Canon<FLAT_ATOMIC_SWAP_X2, COP_FLAT_ATOMIC_SWAP_X2>;
def : Canon<FLAT_ATOMIC_CMPSWAP_X2, COP_FLAT_ATOMIC_CMPSWAP_X2>;
def : Canon<FLAT_ATOMIC_ADD_F32, COP_FLAT_ATOMIC_ADD_F32>;
// FP64 FLAT atomics. `_SADDR` is a distinct pseudo (not alias-stripped),
// so list both; `_RTN` suffixes collapse via the strip rules below.
def : Canon<FLAT_ATOMIC_ADD_F64, COP_FLAT_ATOMIC_ADD_F64>;
def : Canon<FLAT_ATOMIC_ADD_F64_gfx1250, COP_FLAT_ATOMIC_ADD_F64>;
def : Canon<FLAT_ATOMIC_ADD_F64_SADDR, COP_FLAT_ATOMIC_ADD_F64>;
def : Canon<FLAT_ATOMIC_MIN_F64, COP_FLAT_ATOMIC_MIN_NUM_F64>;
def : Canon<FLAT_ATOMIC_MIN_F64_gfx1250, COP_FLAT_ATOMIC_MIN_NUM_F64>;
def : Canon<FLAT_ATOMIC_MIN_F64_SADDR, COP_FLAT_ATOMIC_MIN_NUM_F64>;
def : Canon<FLAT_ATOMIC_MAX_F64, COP_FLAT_ATOMIC_MAX_NUM_F64>;
def : Canon<FLAT_ATOMIC_MAX_F64_gfx1250, COP_FLAT_ATOMIC_MAX_NUM_F64>;
def : Canon<FLAT_ATOMIC_MAX_F64_SADDR, COP_FLAT_ATOMIC_MAX_NUM_F64>;
// ---------------------------------------------------------------------
// GLOBAL atomics
// ---------------------------------------------------------------------
def : Canon<GLOBAL_ATOMIC_ADD, COP_GLOBAL_ATOMIC_ADD>;
def : Canon<GLOBAL_ATOMIC_ADD_X2, COP_GLOBAL_ATOMIC_ADD_X2>;
def : Canon<GLOBAL_ATOMIC_SUB, COP_GLOBAL_ATOMIC_SUB>;
def : Canon<GLOBAL_ATOMIC_AND, COP_GLOBAL_ATOMIC_AND>;
def : Canon<GLOBAL_ATOMIC_OR, COP_GLOBAL_ATOMIC_OR>;
def : Canon<GLOBAL_ATOMIC_XOR, COP_GLOBAL_ATOMIC_XOR>;
def : Canon<GLOBAL_ATOMIC_SMIN, COP_GLOBAL_ATOMIC_SMIN>;
def : Canon<GLOBAL_ATOMIC_SMAX, COP_GLOBAL_ATOMIC_SMAX>;
def : Canon<GLOBAL_ATOMIC_UMIN, COP_GLOBAL_ATOMIC_UMIN>;
def : Canon<GLOBAL_ATOMIC_UMAX, COP_GLOBAL_ATOMIC_UMAX>;
def : Canon<GLOBAL_ATOMIC_SWAP, COP_GLOBAL_ATOMIC_SWAP>;
// b64 (X2) swap: reuse the GLOBAL_ATOMIC_SWAP canonical op; the
// handler detects 64-bit width from the MC opcode.
def : Canon<GLOBAL_ATOMIC_SWAP_X2, COP_GLOBAL_ATOMIC_SWAP>;
def : Canon<GLOBAL_ATOMIC_SWAP_X2_SADDR, COP_GLOBAL_ATOMIC_SWAP>;
def : Canon<GLOBAL_ATOMIC_CMPSWAP, COP_GLOBAL_ATOMIC_CMPSWAP>;
def : Canon<GLOBAL_ATOMIC_ADD_F32, COP_GLOBAL_ATOMIC_ADD_F32>;
def : Canon<GLOBAL_ATOMIC_PK_ADD_BF16, COP_GLOBAL_ATOMIC_PK_ADD_BF16>;
def : Canon<GLOBAL_ATOMIC_PK_ADD_F16, COP_GLOBAL_ATOMIC_PK_ADD_F16>;
// FP64 GLOBAL atomics. Same SADDR / mnemonic-rename considerations
// as the FLAT block above.
def : Canon<GLOBAL_ATOMIC_ADD_F64, COP_GLOBAL_ATOMIC_ADD_F64>;
def : Canon<GLOBAL_ATOMIC_ADD_F64_gfx1250, COP_GLOBAL_ATOMIC_ADD_F64>;
def : Canon<GLOBAL_ATOMIC_ADD_F64_SADDR, COP_GLOBAL_ATOMIC_ADD_F64>;
def : Canon<GLOBAL_ATOMIC_MIN_F64, COP_GLOBAL_ATOMIC_MIN_NUM_F64>;
def : Canon<GLOBAL_ATOMIC_MIN_F64_gfx1250, COP_GLOBAL_ATOMIC_MIN_NUM_F64>;
def : Canon<GLOBAL_ATOMIC_MIN_F64_SADDR, COP_GLOBAL_ATOMIC_MIN_NUM_F64>;
def : Canon<GLOBAL_ATOMIC_MAX_F64, COP_GLOBAL_ATOMIC_MAX_NUM_F64>;
def : Canon<GLOBAL_ATOMIC_MAX_F64_gfx1250, COP_GLOBAL_ATOMIC_MAX_NUM_F64>;
def : Canon<GLOBAL_ATOMIC_MAX_F64_SADDR, COP_GLOBAL_ATOMIC_MAX_NUM_F64>;
// ---------------------------------------------------------------------
// SMEM atomics (enumerate addressing forms: IMM / SGPR / SGPR_IMM)
// ---------------------------------------------------------------------
def : Canon<S_ATOMIC_SWAP_IMM, COP_S_ATOMIC_SWAP>;
def : Canon<S_ATOMIC_SWAP_SGPR, COP_S_ATOMIC_SWAP>;
def : Canon<S_ATOMIC_SWAP_SGPR_IMM, COP_S_ATOMIC_SWAP>;
// S_ATOMIC_DEC (scalar wrap-decrement) -- the three addressing forms
// (IMM / SGPR / SGPR_IMM) share a CanonicalOp.  The `_RTN` variants the
// disassembler emits for the GLC=1 returned-old-value form collapse
// onto these non-RTN pseudos via the `_RTN` alias rule below; the
// handler keys off `di.numDefs` (== MCInstrDesc::getNumDefs()) for
// the write-back decision, so a single lowering covers both halves
// of the contract.
def : Canon<S_ATOMIC_DEC_IMM, COP_S_ATOMIC_DEC>;
def : Canon<S_ATOMIC_DEC_SGPR, COP_S_ATOMIC_DEC>;
def : Canon<S_ATOMIC_DEC_SGPR_IMM, COP_S_ATOMIC_DEC>;
// ---------------------------------------------------------------------
// DS
// ---------------------------------------------------------------------
def : Canon<DS_LOAD_TR16_B128, COP_DS_LOAD_TR16_B128>;
def : Canon<DS_READ_B64_TR_B16, COP_DS_READ_B64_TR_B16>;
def : Canon<DS_READ_B64_TR_B8, COP_DS_READ_B64_TR_B8>;
// gfx1250 64-bit transposed LDS load (i8 element, 8 elements per
// lane post-transpose). Distinct LLVM MC pseudo from the gfx950
// `DS_READ_B64_TR_B8` sibling -- same semantics, different isel
// gate (isGFX1250Plus vs HasGFX950Insts). The handler folds both
// paths into a shared hand-rolled emulation; canonicalising the
// CanonicalOp on the gfx1250 spelling matches the disassembly the
// raise_cli operator sees.
def : Canon<DS_LOAD_TR8_B64, COP_DS_LOAD_TR8_B64>;
def : Canon<DS_READ_B32, COP_DS_READ_B32>;
def : Canon<DS_READ_B64, COP_DS_READ_B64>;
// 96-bit LDS load. LLVM MC keeps the legacy `DS_READ_B96`
// pseudo name for what gfx11+ disassembles as `ds_load_b96`
// (DSInstructions.td:1578); we canonicalise on the gfx11+
// spelling, mirroring the s_set_pc_i64 / DS_LOAD_TR8_B64
// precedent. The handler dispatches via the existing generic
// DS read range -- see dsClassify in handle-ds.cpp for the
// {dwords=3, loadBits=96} entry.
def : Canon<DS_READ_B96, COP_DS_READ_B96>;
def : Canon<DS_READ_B128, COP_DS_READ_B128>;
def : Canon<DS_READ2_B32, COP_DS_READ2_B32>;
def : Canon<DS_READ2_B64, COP_DS_READ2_B64>;
// gfx11+ stride-64 two-address LDS loads (DSInstructions.td:1529,
// 1542 -- `ds_load_2addr_stride64_b{32,64}`). Same two-offset MC
// shape as the non-ST64 siblings; the canonicalization chain
// collapses the _gfx9/_gfx11/_gfx12 reals onto the pseudo
// forms here, mirroring the DS_READ2_B{32,64} entries above.
def : Canon<DS_READ2ST64_B32, COP_DS_READ2ST64_B32>;
def : Canon<DS_READ2ST64_B64, COP_DS_READ2ST64_B64>;
def : Canon<DS_READ_U16, COP_DS_READ_U16>;
def : Canon<DS_READ_I16, COP_DS_READ_I16>;
def : Canon<DS_READ_U8, COP_DS_READ_U8>;
def : Canon<DS_READ_I8, COP_DS_READ_I8>;
def : Canon<DS_WRITE_B32, COP_DS_WRITE_B32>;
def : Canon<DS_WRITE_B64, COP_DS_WRITE_B64>;
// Symmetric 96-bit LDS store. gfx11+ asm spelling is
// `ds_store_b96` (DSInstructions.td:1576); the LLVM MC opcode
// remains `DS_WRITE_B96`.
def : Canon<DS_WRITE_B96, COP_DS_WRITE_B96>;
def : Canon<DS_WRITE_B128, COP_DS_WRITE_B128>;
def : Canon<DS_WRITE2_B32, COP_DS_WRITE2_B32>;
def : Canon<DS_WRITE2_B64, COP_DS_WRITE2_B64>;
// gfx11+ stride-64 two-address LDS stores -- mirror of the
// DS_READ2ST64 read-side entries above.
def : Canon<DS_WRITE2ST64_B32, COP_DS_WRITE2ST64_B32>;
def : Canon<DS_WRITE2ST64_B64, COP_DS_WRITE2ST64_B64>;
def : Canon<DS_WRITE_B16, COP_DS_WRITE_B16>;
def : Canon<DS_WRITE_B8, COP_DS_WRITE_B8>;
// gfx8+ HasD16LoadStore D16_HI store family (DSInstructions.td
// sec. 604-606). Stores bits [31:16] (B16_HI) or bits [23:16] (B8_HI)
// of the source VGPR to LDS -- same VGPR/i32 source operand
// shape as their non-_HI siblings, so the canonical-table macro
// routes both encoding forks (gfx10 m0-based vs gfx11+ no-m0)
// through the same CanonicalOp.
def : Canon<DS_WRITE_B16_D16_HI, COP_DS_WRITE_B16_D16_HI>;
def : Canon<DS_WRITE_B8_D16_HI, COP_DS_WRITE_B8_D16_HI>;
def : Canon<DS_BPERMUTE_B32, COP_DS_BPERMUTE_B32>;
// ds_permute_b32 -- forward cross-lane PUSH permute, mirror of the
// PULL ds_bpermute_b32 above. Lifted through llvm.amdgcn.ds.permute
// in handle-ds.cpp (see the DS_PERMUTE_B32 handler).
def : Canon<DS_PERMUTE_B32, COP_DS_PERMUTE_B32>;
// ds_swizzle_b32 -- wave-width-specific cross-lane shuffle. The
// handler refuses with `unsupportedInstructionForm` until the P6 rewrite
// lands (see the ds_swizzle_b32 row of hotswap/docs/wave-size-
// translation.md sec. 5.3); the wave-size classifier (Phase 1.4.5)
// flags it as a Class 2 obstruction (wave-size-translation.md
// sec. 6) in the cross-wave case.
def : Canon<DS_SWIZZLE_B32, COP_DS_SWIZZLE_B32>;
// DS atomics carry `_RTN` as an infix (`DS_ADD_RTN_F64`), so the
// trailing-suffix strip rule doesn't fire; alias it explicitly.
def : Canon<DS_ADD_F64, COP_DS_ADD_F64>;
def : Canon<DS_ADD_RTN_F64, COP_DS_ADD_F64>;
def : Canon<DS_ADD_U32, COP_DS_ADD_U32>;
def : Canon<DS_ADD_RTN_U32, COP_DS_ADD_U32>;
// ---------------------------------------------------------------------
// MUBUF direct-to-LDS loads (distinct semantics from VGPR-dest loads)
// ---------------------------------------------------------------------
// LLVM only ships DWORD and DWORDX4 LDS pseudos -- the DWORDX2_LDS
// CanonicalOp
// stays unmapped until an LLVM pseudo exists for it.
defm : CanonMUBUF4<"BUFFER_LOAD_DWORD_LDS", COP_BUFFER_LOAD_DWORD_LDS>;
defm : CanonMUBUF4<"BUFFER_LOAD_DWORDX4_LDS", COP_BUFFER_LOAD_DWORDX4_LDS>;
// ---------------------------------------------------------------------
// MUBUF loads
// ---------------------------------------------------------------------
// gfx10- uses `BUFFER_LOAD_DWORD_*`; gfx11+ (vbuffer) uses the same
// pseudo name with a `_VBUFFER_` infix rather than a `B32` rename.
defm : CanonMUBUF4<"BUFFER_LOAD_DWORD", COP_BUFFER_LOAD_DWORD>;
defm : CanonVBUF4<"BUFFER_LOAD_DWORD", COP_BUFFER_LOAD_DWORD>;
defm : CanonMUBUF4<"BUFFER_LOAD_DWORDX2", COP_BUFFER_LOAD_DWORDX2>;
defm : CanonMUBUF4<"BUFFER_LOAD_DWORDX3", COP_BUFFER_LOAD_DWORDX3>;
defm : CanonMUBUF4<"BUFFER_LOAD_DWORDX4", COP_BUFFER_LOAD_DWORDX4>;
defm : CanonMUBUF4<"BUFFER_LOAD_UBYTE", COP_BUFFER_LOAD_UBYTE>;
defm : CanonMUBUF4<"BUFFER_LOAD_SBYTE", COP_BUFFER_LOAD_SBYTE>;
defm : CanonMUBUF4<"BUFFER_LOAD_USHORT", COP_BUFFER_LOAD_USHORT>;
defm : CanonMUBUF4<"BUFFER_LOAD_SSHORT", COP_BUFFER_LOAD_SSHORT>;
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
defm : CanonMUBUF4<"BUFFER_LOAD_SHORT_D16", COP_BUFFER_LOAD_SHORT_D16>;
defm : CanonVBUF4<"BUFFER_LOAD_SHORT_D16", COP_BUFFER_LOAD_SHORT_D16>;
defm : CanonMUBUF4<"BUFFER_LOAD_SHORT_D16_HI", COP_BUFFER_LOAD_SHORT_D16_HI>;
defm : CanonVBUF4<"BUFFER_LOAD_SHORT_D16_HI", COP_BUFFER_LOAD_SHORT_D16_HI>;
defm : CanonMUBUF4<"BUFFER_LOAD_UBYTE_D16", COP_BUFFER_LOAD_UBYTE_D16>;
defm : CanonVBUF4<"BUFFER_LOAD_UBYTE_D16", COP_BUFFER_LOAD_UBYTE_D16>;
defm : CanonMUBUF4<"BUFFER_LOAD_UBYTE_D16_HI", COP_BUFFER_LOAD_UBYTE_D16_HI>;
defm : CanonVBUF4<"BUFFER_LOAD_UBYTE_D16_HI", COP_BUFFER_LOAD_UBYTE_D16_HI>;
defm : CanonMUBUF4<"BUFFER_LOAD_SBYTE_D16", COP_BUFFER_LOAD_SBYTE_D16>;
defm : CanonVBUF4<"BUFFER_LOAD_SBYTE_D16", COP_BUFFER_LOAD_SBYTE_D16>;
defm : CanonMUBUF4<"BUFFER_LOAD_SBYTE_D16_HI", COP_BUFFER_LOAD_SBYTE_D16_HI>;
defm : CanonVBUF4<"BUFFER_LOAD_SBYTE_D16_HI", COP_BUFFER_LOAD_SBYTE_D16_HI>;
// ---------------------------------------------------------------------
// MUBUF stores
// ---------------------------------------------------------------------
defm : CanonMUBUF4<"BUFFER_STORE_DWORD", COP_BUFFER_STORE_DWORD>;
defm : CanonVBUF4<"BUFFER_STORE_DWORD", COP_BUFFER_STORE_DWORD>;
defm : CanonMUBUF4<"BUFFER_STORE_DWORDX2", COP_BUFFER_STORE_DWORDX2>;
defm : CanonMUBUF4<"BUFFER_STORE_DWORDX3", COP_BUFFER_STORE_DWORDX3>;
defm : CanonMUBUF4<"BUFFER_STORE_DWORDX4", COP_BUFFER_STORE_DWORDX4>;
defm : CanonMUBUF4<"BUFFER_STORE_BYTE", COP_BUFFER_STORE_BYTE>;
defm : CanonMUBUF4<"BUFFER_STORE_SHORT", COP_BUFFER_STORE_SHORT>;
// ---------------------------------------------------------------------
// MUBUF atomics (non-return; handlers don't distinguish RTN yet)
// ---------------------------------------------------------------------
defm : CanonMUBUF4<"BUFFER_ATOMIC_ADD", COP_BUFFER_ATOMIC_ADD>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_SUB", COP_BUFFER_ATOMIC_SUB>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_AND", COP_BUFFER_ATOMIC_AND>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_OR", COP_BUFFER_ATOMIC_OR>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_XOR", COP_BUFFER_ATOMIC_XOR>;
// Class 3 non-commutative atomics (hotswap/docs/wave-size-
// translation.md sec. 6). The wave-size classifier flags them in
// the cross-wave case before
// dispatch ever reaches handle-mubuf.cpp's switch. Same-wave and
// same-target lifts are still modeled with raw-buffer atomics there.
defm : CanonMUBUF4<"BUFFER_ATOMIC_SWAP", COP_BUFFER_ATOMIC_SWAP>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_CMPSWAP", COP_BUFFER_ATOMIC_CMPSWAP>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_ADD_F32", COP_BUFFER_ATOMIC_ADD_F32>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_PK_ADD_BF16", COP_BUFFER_ATOMIC_PK_ADD_BF16>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_PK_ADD_F16", COP_BUFFER_ATOMIC_PK_ADD_F16>;
// FP64 BUFFER atomics. The gfx942 MUBUF MIN/MAX_F64 is a raw `<`/`>`
// comparator; the gfx12 VBUFFER form (below) is IEEE 754-2019
// minimumNumber/maximumNumber. Keep them as distinct canonical ops.
defm : CanonMUBUF4<"BUFFER_ATOMIC_ADD_F64", COP_BUFFER_ATOMIC_ADD_F64>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_MIN_F64", COP_BUFFER_ATOMIC_MIN_F64>;
defm : CanonMUBUF4<"BUFFER_ATOMIC_MAX_F64", COP_BUFFER_ATOMIC_MAX_F64>;
// gfx11+/gfx12 VBUFFER fork. LLVM MC keeps the legacy
// `BUFFER_ATOMIC_*` pseudo names but suffixes them `_VBUFFER_<addr>`
// for the gfx12 buffer-descriptor encoding; map them to the same
// CanonicalOps. PK_ADD_BF16/F16 are omitted: no VBUFFER Real form
// exists, so a VBUF4 entry would fail to compile.
defm : CanonVBUF4<"BUFFER_ATOMIC_ADD", COP_BUFFER_ATOMIC_ADD>;
defm : CanonVBUF4<"BUFFER_ATOMIC_SUB", COP_BUFFER_ATOMIC_SUB>;
defm : CanonVBUF4<"BUFFER_ATOMIC_AND", COP_BUFFER_ATOMIC_AND>;
defm : CanonVBUF4<"BUFFER_ATOMIC_OR", COP_BUFFER_ATOMIC_OR>;
defm : CanonVBUF4<"BUFFER_ATOMIC_XOR", COP_BUFFER_ATOMIC_XOR>;
defm : CanonVBUF4<"BUFFER_ATOMIC_SWAP", COP_BUFFER_ATOMIC_SWAP>;
defm : CanonVBUF4<"BUFFER_ATOMIC_CMPSWAP", COP_BUFFER_ATOMIC_CMPSWAP>;
defm : CanonVBUF4<"BUFFER_ATOMIC_ADD_F32", COP_BUFFER_ATOMIC_ADD_F32>;
defm : CanonVBUF4<"BUFFER_ATOMIC_ADD_F64", COP_BUFFER_ATOMIC_ADD_F64>;
defm : CanonVBUF4<"BUFFER_ATOMIC_MIN_F64", COP_BUFFER_ATOMIC_MIN_NUM_F64>;
defm : CanonVBUF4<"BUFFER_ATOMIC_MAX_F64", COP_BUFFER_ATOMIC_MAX_NUM_F64>;
// ---------------------------------------------------------------------
// AGPR moves
// ---------------------------------------------------------------------
def : Canon<V_ACCVGPR_READ_B32_e64, COP_V_ACCVGPR_READ_B32>;
def : Canon<V_ACCVGPR_WRITE_B32_e64, COP_V_ACCVGPR_WRITE_B32>;
// ---------------------------------------------------------------------
// MFMA shapes with a single TableGen pseudo per intrinsic.
// Register-class variants (`_vgprcd_`, `_mac_`) collapse to the base
// `_e64` via pseudoAlias, so we only list the base pseudo here.
// ---------------------------------------------------------------------
def : Canon<V_MFMA_F32_16X16X16F16_e64, COP_V_MFMA_F32_16x16x16_F16>;
def : Canon<V_MFMA_F32_32X32X8F16_e64, COP_V_MFMA_F32_32x32x8_F16>;
def : Canon<V_MFMA_F32_16X16X4F32_e64, COP_V_MFMA_F32_16x16x4_F32>;
def : Canon<V_MFMA_F32_32X32X1F32_e64, COP_V_MFMA_F32_32x32x1_F32>;
def : Canon<V_MFMA_F32_32X32X2F32_e64, COP_V_MFMA_F32_32x32x2_F32>;
def : Canon<V_MFMA_F32_4X4X1F32_e64, COP_V_MFMA_F32_4x4x1_F32>;
def : Canon<V_MFMA_F32_16X16X1F32_e64, COP_V_MFMA_F32_16x16x1_F32>;
def : Canon<V_MFMA_F32_32X32X4F16_e64, COP_V_MFMA_F32_32x32x4_F16>;
def : Canon<V_MFMA_F32_16X16X4F16_e64, COP_V_MFMA_F32_16x16x4_F16>;
def : Canon<V_MFMA_F32_4X4X4F16_e64, COP_V_MFMA_F32_4x4x4_F16>;
def : Canon<V_MFMA_I32_16X16X32I8_e64, COP_V_MFMA_I32_16x16x32_I8>;
def : Canon<V_MFMA_I32_32X32X16I8_e64, COP_V_MFMA_I32_32x32x16_I8>;
def : Canon<V_MFMA_I32_32X32X4I8_e64, COP_V_MFMA_I32_32x32x4_I8>;
def : Canon<V_MFMA_I32_16X16X4I8_e64, COP_V_MFMA_I32_16x16x4_I8>;
def : Canon<V_MFMA_I32_4X4X4I8_e64, COP_V_MFMA_I32_4x4x4_I8>;
def : Canon<V_MFMA_F32_16X16X8XF32_e64, COP_V_MFMA_F32_16x16x8_XF32>;
def : Canon<V_MFMA_F32_32X32X4XF32_e64, COP_V_MFMA_F32_32x32x4_XF32>;
def : Canon<V_MFMA_F32_32X32X2BF16_e64, COP_V_MFMA_F32_32x32x2_BF16>;
def : Canon<V_MFMA_F32_16X16X2BF16_e64, COP_V_MFMA_F32_16x16x2_BF16>;
def : Canon<V_MFMA_F32_4X4X2BF16_e64, COP_V_MFMA_F32_4x4x2_BF16>;
// Only the 16x16x16 and 32x32x8 1K shapes were in the legacy mnemonic
// table; the 4x4x4/16x16x4/32x32x4 1K variants are distinct intrinsics
// that the raiser does not (yet) model and therefore stay unmapped.
def : Canon<V_MFMA_F32_16X16X16BF16_1K_e64, COP_V_MFMA_F32_16x16x16_BF16_1K>;
def : Canon<V_MFMA_F32_32X32X8BF16_1K_e64, COP_V_MFMA_F32_32x32x8_BF16_1K>;
def : Canon<V_MFMA_F32_16X16X32_BF16_e64, COP_V_MFMA_F32_16x16x32_BF16>;
def : Canon<V_MFMA_F32_32X32X16_BF16_e64, COP_V_MFMA_F32_32x32x16_BF16>;
def : Canon<V_MFMA_F32_16X16X32_F16_e64, COP_V_MFMA_F32_16x16x32_F16>;
def : Canon<V_MFMA_F32_16X16X32_FP8_FP8_e64, COP_V_MFMA_F32_16x16x32_FP8_FP8>;
def : Canon<V_MFMA_F32_16X16X32_FP8_BF8_e64, COP_V_MFMA_F32_16x16x32_FP8_BF8>;
def : Canon<V_MFMA_F32_16X16X32_BF8_FP8_e64, COP_V_MFMA_F32_16x16x32_BF8_FP8>;
def : Canon<V_MFMA_F32_16X16X32_BF8_BF8_e64, COP_V_MFMA_F32_16x16x32_BF8_BF8>;
def : Canon<V_MFMA_F32_32X32X16_FP8_FP8_e64, COP_V_MFMA_F32_32x32x16_FP8_FP8>;
def : Canon<V_MFMA_F32_32X32X16_FP8_BF8_e64, COP_V_MFMA_F32_32x32x16_FP8_BF8>;
def : Canon<V_MFMA_F32_32X32X16_BF8_FP8_e64, COP_V_MFMA_F32_32x32x16_BF8_FP8>;
def : Canon<V_MFMA_F32_32X32X16_BF8_BF8_e64, COP_V_MFMA_F32_32x32x16_BF8_BF8>;
// ---------------------------------------------------------------------
// MFMA F8F6F4: LLVM enumerates all 9 src0/src1 mantissa-type pairs as
// separate pseudos; the raiser handles them identically.
// ---------------------------------------------------------------------
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f4_f4_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f4_f6_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f4_f8_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f6_f4_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f6_f6_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f6_f8_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f8_f4_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f8_f6_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_16X16X128_F8F6F4_f8_f8_e64, COP_V_MFMA_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_e64, COP_V_MFMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f4_f4_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f4_f6_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f4_f8_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f6_f4_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f6_f6_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f6_f8_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f8_f4_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f8_f6_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_F32_32X32X64_F8F6F4_f8_f8_e64, COP_V_MFMA_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
def : Canon<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_e64, COP_V_MFMA_SCALE_F32_32x32x64_F8F6F4>;
// ---------------------------------------------------------------------
// WMMA (gfx1250): twoaddr/threeaddr pseudo variants both mean the same
// semantic op to our raiser.
// ---------------------------------------------------------------------
def : Canon<V_WMMA_F32_16X16X32_F16_w32_twoaddr, COP_V_WMMA_F32_16x16x32_F16>;
def : Canon<V_WMMA_F32_16X16X32_F16_w32_threeaddr, COP_V_WMMA_F32_16x16x32_F16>;
// 16x16x4 WMMA, 32-bit f32 inputs and accumulator (gfx1250 RDNA4
// VOP3P opcode 0x05D, K=4 across 32 lanes -> per-lane <2 x f32>
// for A and B). Lifts to the gfx1250-only intrinsic
// amdgcn_wmma_f32_16x16x4_f32 in the same-target path; cross-
// target (gfx942) refuses loudly because no K=4 f32 -> MFMA
// decomposition exists in `wmma-lowering.cpp` yet.
def : Canon<V_WMMA_F32_16X16X4_F32_w32_twoaddr, COP_V_WMMA_F32_16x16x4_F32>;
def : Canon<V_WMMA_F32_16X16X4_F32_w32_threeaddr, COP_V_WMMA_F32_16x16x4_F32>;
// 16x16x32 WMMA, BF16 inputs, F32 accumulator (gfx1250 RDNA4 VOP3P
// opcode 0x062). Same fragment shape as the F16 variant -- the only
// delta is the input element type, which routes to the bf16 MFMA
// intrinsic on gfx942 (mfma_f32_16x16x16bf16_1k) and to
// amdgcn_wmma_f32_16x16x32_bf16 when the target supports WMMA12.
def : Canon<V_WMMA_F32_16X16X32_BF16_w32_twoaddr, COP_V_WMMA_F32_16x16x32_BF16>;
def : Canon<V_WMMA_F32_16X16X32_BF16_w32_threeaddr, COP_V_WMMA_F32_16x16x32_BF16>;
// 16x16x64 WMMA, 8-bit element types (gfx1250 RDNA4 VOP3P opcodes
// 0x06a..0x06d). All four AB combinations share the per-lane
// fragment shape (A,B: <8 x i32>, C/D: <8 x f32>) and route through
// the gfx942 MFMA lowering with a per-variant intrinsic dispatch
// (see `runGroupPass` in wmma-lowering.cpp). Both `_twoaddr` and
// `_threeaddr` MC pseudo variants represent the same semantic op,
// mirroring the F16/BF16 mapping above.
def : Canon<V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr, COP_V_WMMA_F32_16x16x64_FP8_FP8>;
def : Canon<V_WMMA_F32_16X16X64_FP8_FP8_w32_threeaddr, COP_V_WMMA_F32_16x16x64_FP8_FP8>;
def : Canon<V_WMMA_F32_16X16X64_FP8_BF8_w32_twoaddr, COP_V_WMMA_F32_16x16x64_FP8_BF8>;
def : Canon<V_WMMA_F32_16X16X64_FP8_BF8_w32_threeaddr, COP_V_WMMA_F32_16x16x64_FP8_BF8>;
def : Canon<V_WMMA_F32_16X16X64_BF8_FP8_w32_twoaddr, COP_V_WMMA_F32_16x16x64_BF8_FP8>;
def : Canon<V_WMMA_F32_16X16X64_BF8_FP8_w32_threeaddr, COP_V_WMMA_F32_16x16x64_BF8_FP8>;
def : Canon<V_WMMA_F32_16X16X64_BF8_BF8_w32_twoaddr, COP_V_WMMA_F32_16x16x64_BF8_BF8>;
def : Canon<V_WMMA_F32_16X16X64_BF8_BF8_w32_threeaddr, COP_V_WMMA_F32_16x16x64_BF8_BF8>;
// 16x16x64 WMMA, IU8 (signed/unsigned 8-bit integer inputs, i32
// accumulator; gfx1250 RDNA4 VOP3P opcode 0x072). The MC opcode
// sign-extension knobs travel through the `neg_lo` operand bits
// rather than the opcode itself, so there's a single CanonicalOp for
// both signed and unsigned interpretations. On gfx942 this
// routes to mfma_i32_16x16x32_i8 (i64 packed A/B, <4 x i32>
// accumulator). Both `_twoaddr` and `_threeaddr` MC pseudo
// variants represent the same semantic op, mirroring the
// F16/BF16 / FP8/BF8 mappings above.
def : Canon<V_WMMA_I32_16X16X64_IU8_w32_twoaddr, COP_V_WMMA_I32_16x16x64_IU8>;
def : Canon<V_WMMA_I32_16X16X64_IU8_w32_threeaddr, COP_V_WMMA_I32_16x16x64_IU8>;
// ---------------------------------------------------------------------
// Scaled WMMA F8F6F4 (gfx1250 RDNA4 -- VOP3PX2 paired form, real opcode
// 0x033 + 0x35 / 0x3a, FLATInstructions / VOP3PInstructions.td:1987).
// The 9 mantissa-pair pseudos (`f4_f4`, `f4_f6`, ..., `f8_f8`) all share
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
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr, COP_V_WMMA_SCALE_F32_16x16x128_F8F6F4>;
// Scaled WMMA F8F6F4 with K=16 scale granularity (gfx1250
// `v_wmma_scale16_*`; scale_src operands are i64).
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr, COP_V_WMMA_SCALE16_F32_16x16x128_F8F6F4>;
def : Canon<V_WMMA_SCALE16_F32_32X16X128_F4_w32_gfx1250, COP_V_WMMA_SCALE16_F32_32x16x128_F4>;
def : Canon<V_WMMA_SCALE16_F32_32X16X128_F4_w32_twoaddr, COP_V_WMMA_SCALE16_F32_32x16x128_F4>;
def : Canon<V_WMMA_SCALE16_F32_32X16X128_F4_w32_threeaddr, COP_V_WMMA_SCALE16_F32_32x16x128_F4>;
// ---------------------------------------------------------------------
// VIMAGE TENSOR (gfx1250 RDNA4 -- VIMAGE 0xc4 / 0xc5).
// The disassembler's MC opcodes are the `_gfx1250` reals
// (`TENSOR_LOAD_TO_LDS_d{2,4}_gfx1250`,
// `TENSOR_STORE_FROM_LDS_d{2,4}_gfx1250`,
// `MIMGInstructions.td:2115`); the canonicalization chain in
// hotswap-tblgen collapses each real onto its pseudo
// (`TENSOR_LOAD_TO_LDS_d{2,4}` / `TENSOR_STORE_FROM_LDS_d{2,4}`)
// via `buildMcToPseudoMap`. Both the `_d2` (up-to-2D) and `_d4`
// (up-to-4D) operand-shape variants share a CanonicalOp because their
// semantic intent and their cross-target refusal contract are
// identical; `handleVIMAGE` uses `di.Mnemonic` directly when it
// needs to discriminate (e.g., a future native-target intrinsic
// lowering that zero-fills D# group 2/3 for the `_d2` form).
// ---------------------------------------------------------------------
def : Canon<TENSOR_LOAD_TO_LDS_d2, COP_TENSOR_LOAD_TO_LDS>;
def : Canon<TENSOR_LOAD_TO_LDS_d4, COP_TENSOR_LOAD_TO_LDS>;
def : Canon<TENSOR_STORE_FROM_LDS_d2, COP_TENSOR_STORE_FROM_LDS>;
def : Canon<TENSOR_STORE_FROM_LDS_d4, COP_TENSOR_STORE_FROM_LDS>;
// ---------------------------------------------------------------------
// FLAT async global -> LDS (gfx1250 -- VFLAT 0x60-0x62 reals).
// The disassembler's MC opcodes are the `_gfx1250` (or
// `_SADDR_gfx1250`) reals declared by
// `VFLAT_Real_AllAddr_gfx1250` (FLATInstructions.td:2003-2018)
// off the `FLAT_Global_Load_LDS_Pseudo<..., IsAsync=1>` family
// (FLATInstructions.td:391-417). The canonicalization chain in
// hotswap-tblgen collapses each real onto its pseudo
// (`GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` / `_SADDR`
// sibling) via `buildMcToPseudoMap`; both the plain and SADDR
// variants per width share a CanonicalOp because the lift differs
// only in the source-bank wiring (vaddr-only VGPR_64 vs
// saddr+voff SGPR_64+VGPR_32). `handleFLAT` uses `op.nSrcs()`
// (4 for plain, 5 for SADDR) to pick the operand-decoder path,
// exactly mirroring how `handleVIMAGE` discriminates
// `tensor_load_to_lds_d2` vs `_d4`.
// ---------------------------------------------------------------------
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B8, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B8>;
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B8_SADDR, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B8>;
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B32, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B32>;
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B32_SADDR, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B32>;
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B64, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B64>;
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B64_SADDR, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B64>;
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B128, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B128>;
def : Canon<GLOBAL_LOAD_ASYNC_TO_LDS_B128_SADDR, COP_GLOBAL_LOAD_ASYNC_TO_LDS_B128>;
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
def : Canon<GLOBAL_PREFETCH_B8, COP_GLOBAL_PREFETCH_B8>;
def : Canon<GLOBAL_PREFETCH_B8_SADDR, COP_GLOBAL_PREFETCH_B8>;
def : Canon<FLAT_PREFETCH_B8, COP_FLAT_PREFETCH_B8>;
def : Canon<FLAT_PREFETCH_B8_SADDR, COP_FLAT_PREFETCH_B8>;
// FLAT WMMA load-with-transpose (gfx1250 wave32).
def : Canon<GLOBAL_LOAD_TR_B128_w32, COP_GLOBAL_LOAD_TR16_B128>;
def : Canon<GLOBAL_LOAD_TR_B128_w32_SADDR, COP_GLOBAL_LOAD_TR16_B128>;
def : Canon<GLOBAL_LOAD_TR_B64_w32, COP_GLOBAL_LOAD_TR8_B64>;
def : Canon<GLOBAL_LOAD_TR_B64_w32_SADDR, COP_GLOBAL_LOAD_TR8_B64>;
def : Canon<GLOBAL_LOAD_TR4_B64, COP_GLOBAL_LOAD_TR4_B64>;
def : Canon<GLOBAL_LOAD_TR4_B64_SADDR, COP_GLOBAL_LOAD_TR4_B64>;
def : Canon<GLOBAL_LOAD_TR6_B96, COP_GLOBAL_LOAD_TR6_B96>;
def : Canon<GLOBAL_LOAD_TR6_B96_SADDR, COP_GLOBAL_LOAD_TR6_B96>;
