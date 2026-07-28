// Test the regular block-32 scaled M=32 lowering used by ROCM-28594.
//
// The source operation is B0-only. A0 supports the M=16 regular-scale form,
// so the replacement splits only M and preserves the original block-32 scale
// operands for both halves.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: wmma_scale: exact M split
// API-SAME: 2 WMMAs
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_scale_32x16>:
// DISASM-NOT: v_wmma_scale_f32_32x16x128_f4
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM-NOT: matrix_a_reuse
// DISASM-NOT: matrix_b_reuse
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[16:23], v[32:39], v[0:7], v40, v42{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4{{.*}}neg_lo:[0,0,1]
// DISASM: v_nop
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15], v[24:31], v[32:39], v[8:15], v40, v42{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4{{.*}}matrix_a_scale:MATRIX_SCALE_ROW1{{.*}}neg_lo:[0,0,1]
// DISASM: v_nop

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl test_wmma_scale_32x16
.p2align 8
.type test_wmma_scale_32x16,@function
test_wmma_scale_32x16:
  v_wmma_scale_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[0:15], v40, v42 matrix_a_scale_fmt:MATRIX_SCALE_FMT_E8 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E8 matrix_a_reuse matrix_b_reuse neg_lo:[0,0,1]
  s_endpgm
.Ltest_wmma_scale_32x16_end:
.size test_wmma_scale_32x16, .Ltest_wmma_scale_32x16_end-test_wmma_scale_32x16

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale_32x16
  .amdhsa_next_free_vgpr 43
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale_32x16
      .symbol: test_wmma_scale_32x16.kd
      .sgpr_count: 2
      .vgpr_count: 43
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
