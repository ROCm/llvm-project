// COM: Exact decomposition of the corpus's M=32 FP4 block-16 scaled WMMA.
// COM: One 13-VGPR bank-zero block holds an eight-register masked-A half, four
// COM: gathered scale values, and one temporary. The schedule is M0 low/high,
// COM: then M1 low/high. Original A and Scale16 tuples are never modified.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: physical forward-dead proof at offset
// API-SAME: found 13 VGPRs
// API: wmma_scale16: exact M+K split
// API-SAME: masked-A=v52:59, scales=v60,v61,v62,v63, tmp=v64, +0 vgpr, 4 WMMAs
// API-NOT: error:
// API: liveness: kernel test_wmma_scale16_32x16:
// API-SAME: scratch_reused=13, scratch_above_kd=0
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_scale16_32x16>:
// DISASM-NOT: v_wmma_scale16
// DISASM: s_branch
// DISASM: s_endpgm
//
// COM: Even/odd byte gathers land in dedicated low-bank scale registers.
// COM: The original scale pairs remain read-only; no reversible v_perm
// COM: mutation/restore sequence is emitted.
// DISASM-NOT: v_perm_b32
// DISASM: v_and_b32{{(_e32)?}} v60, 0xff, v40
// DISASM: v_and_b32{{(_e32)?}} v61, 0xff, v42
// DISASM: v_bfe_u32 v62, v40, 8, 8
// DISASM: v_bfe_u32 v63, v42, 8, 8
//
// COM: D==C, M0 low/high then M1 low/high. Reuse hints are stripped, and the
// COM: original C modifier appears only on the two low-K passes.
// DISASM-NOT: matrix_a_reuse
// DISASM-NOT: matrix_b_reuse
// DISASM: v_mov_b32{{(_e32)?}} v52, v16
// DISASM: v_mov_b32{{(_e32)?}} v54, 0
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[52:59], v[32:39], v[0:7], v60, v61{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4{{.*}}neg_lo:[0,0,1]
// DISASM: v_nop
// DISASM: v_mov_b32{{(_e32)?}} v52, 0
// DISASM: v_mov_b32{{(_e32)?}} v54, v18
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[52:59], v[32:39], v[0:7], v62, v63{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-NOT: neg_lo
// DISASM: v_nop
// DISASM: v_mov_b32{{(_e32)?}} v52, v24
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15], v[52:59], v[32:39], v[8:15], v60, v61{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4{{.*}}matrix_a_scale:MATRIX_SCALE_ROW1{{.*}}neg_lo:[0,0,1]
// DISASM: v_nop
// DISASM: v_mov_b32{{(_e32)?}} v52, 0
// DISASM: v_mov_b32{{(_e32)?}} v54, v26
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15], v[52:59], v[32:39], v[8:15], v62, v63{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4{{.*}}matrix_a_scale:MATRIX_SCALE_ROW1
// DISASM-NOT: neg_lo
// DISASM: v_nop
// DISASM-NOT: v_perm_b32
//
// COM: The second lowering permits matrix B to overlap C. Its first low pass
// COM: therefore has the same v[32:39] range in src1 and src2.
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[64:71], v[32:39], v[32:39], v72, v73
// DISASM-NOT: v_wmma_scale16

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// Reduced opaque sequence from the f4gemm corpus. The A0 decoder splits this
// legacy B0 VOP3 into two unknown dwords. The sequence before the WMMA tests
// local MODE recovery; the sequence after it tests forward physical liveness.
.macro opaque_b0_vop3
  .long 0xd0310000
  .long 0x00100000
  v_cmp_ge_u16_e32 vcc_lo, s32, v18.l
.endm

.globl test_wmma_scale16_32x16
.p2align 8
.type test_wmma_scale16_32x16,@function
test_wmma_scale16_32x16:
  s_set_vgpr_msb 0
  opaque_b0_vop3
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[0:15], v[40:41], v[42:43] matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_a_reuse matrix_b_reuse neg_lo:[0,0,1]
  opaque_b0_vop3
  v_mov_b32 v52, 0
  v_mov_b32 v53, 0
  v_mov_b32 v54, 0
  v_mov_b32 v55, 0
  v_mov_b32 v56, 0
  v_mov_b32 v57, 0
  v_mov_b32 v58, 0
  v_mov_b32 v59, 0
  v_mov_b32 v60, 0
  v_mov_b32 v61, 0
  v_mov_b32 v62, 0
  v_mov_b32 v63, 0
  v_mov_b32 v64, 0
  s_branch test_wmma_scale16_32x16_bc_overlap
.Ltest_wmma_scale16_32x16_end:
.size test_wmma_scale16_32x16, .Ltest_wmma_scale16_32x16_end-test_wmma_scale16_32x16

.globl test_wmma_scale16_32x16_bc_overlap
.p2align 8
.type test_wmma_scale16_32x16_bc_overlap,@function
test_wmma_scale16_32x16_bc_overlap:
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[32:47], v[48:49], v[50:51] matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
  s_endpgm
.Ltest_wmma_scale16_32x16_bc_overlap_end:
.size test_wmma_scale16_32x16_bc_overlap, .Ltest_wmma_scale16_32x16_bc_overlap_end-test_wmma_scale16_32x16_bc_overlap

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_32x16
  .amdhsa_next_free_vgpr 1024
  .amdhsa_next_free_sgpr 34
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdhsa_kernel test_wmma_scale16_32x16_bc_overlap
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale16_32x16
      .symbol: test_wmma_scale16_32x16.kd
      .sgpr_count: 34
      .vgpr_count: 1024
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: test_wmma_scale16_32x16_bc_overlap
      .symbol: test_wmma_scale16_32x16_bc_overlap.kd
      .sgpr_count: 2
      .vgpr_count: 52
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
