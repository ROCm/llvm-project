// COM: The M=32 Scale16 lowering does not require all 13 scratch VGPRs to be
// COM: contiguous. This fixture exposes one eight-register dead run for masked
// COM: A plus five separate dead low-bank registers for generated scales and
// COM: the gather temporary. The kernel is already at a metadata digit
// COM: boundary, so unnecessary growth would also make the rewrite fail.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: wmma_scale16: exact M+K split
// API-SAME: masked-A=v120:127
// API-SAME: scales=v119,v118,v117,v115, tmp=v114, +0 vgpr, 4 WMMAs
// API-NOT: updating AMDGPU metadata changes note size
// API-NOT: error:
// API: liveness: kernel test_wmma_scale16_32x16_split_scratch:
// API-SAME: vgprs_before=128, vgprs_after=128
// API-SAME: scratch_reused=13, scratch_above_kd=0
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_scale16_32x16_split_scratch>:
// DISASM-NOT: v_wmma_scale16
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: v_and_b32{{(_e32)?}} v119, 0xff, v64
// DISASM: v_and_b32{{(_e32)?}} v118, 0xff, v46
// DISASM: v_bfe_u32 v117, v64, 8, 8
// DISASM: v_bfe_u32 v115, v46, 8, 8
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[120:127], v[34:41], 0, v119, v118
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[120:127], v[34:41], v[16:23], v117, v115
// DISASM-NOT: v_wmma_scale16

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.macro opaque_b0_vop3
  .long 0xd0310000
  .long 0x00100000
  v_cmp_ge_u16_e32 vcc_lo, s32, v18.l
.endm

.globl test_wmma_scale16_32x16_split_scratch
.p2align 8
.type test_wmma_scale16_32x16_split_scratch,@function
test_wmma_scale16_32x16_split_scratch:
  s_set_vgpr_msb 0
  opaque_b0_vop3
  v_wmma_scale16_f32_32x16x128_f4 v[16:31], v[0:15], v[34:41], 0, v[64:65], v[46:47] matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
  opaque_b0_vop3
  // Keep one register live in every otherwise available 13-register interval.
  v_mov_b32 v0, v48
  v_mov_b32 v0, v60
  v_mov_b32 v0, v72
  v_mov_b32 v0, v84
  v_mov_b32 v0, v96
  v_mov_b32 v0, v108
  v_mov_b32 v0, v116
  v_mov_b32 v117, 0
  v_mov_b32 v118, 0
  v_mov_b32 v119, 0
  v_mov_b32 v120, 0
  v_mov_b32 v121, 0
  v_mov_b32 v122, 0
  v_mov_b32 v123, 0
  v_mov_b32 v124, 0
  v_mov_b32 v125, 0
  v_mov_b32 v126, 0
  v_mov_b32 v127, 0
  v_mov_b32 v61, 0
  v_mov_b32 v53, 0
  s_endpgm
.Ltest_wmma_scale16_32x16_split_scratch_end:
.size test_wmma_scale16_32x16_split_scratch, .Ltest_wmma_scale16_32x16_split_scratch_end-test_wmma_scale16_32x16_split_scratch

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_32x16_split_scratch
  .amdhsa_next_free_vgpr 128
  .amdhsa_next_free_sgpr 34
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale16_32x16_split_scratch
      .symbol: test_wmma_scale16_32x16_split_scratch.kd
      .sgpr_count: 34
      .vgpr_count: 128
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
