// A kernel may allocate more than 256 physical VGPRs through gfx1250's
// VGPR-MSB mode. The scale16 lowering's generated assembly must still use
// encodable v0-v255 names and select its above-KD scratch bank explicitly.
// The bump is occupancy-neutral (the original and rewritten allocations both
// admit one wave/EU), despite the maximum-workgroup metadata asking for two.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: wmma_scale16: exact K-split
// API-NOT: register index is out of range
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_scale16_large_vgpr_count>:
// DISASM-NOT: v_wmma_scale16
// COM: Masked A and generated scale operands share a preserved bank-zero
// COM: block. Matrix B remains in above-KD scratch. Each WMMA consumes the
// COM: exact low-bank values produced by its gather.
// DISASM: v_and_b32_e32 [[LO_A:v[0-9]+]], 0xff, v0{{[[:space:]]+//}}
// DISASM: v_and_b32_e32 [[LO_B:v[0-9]+]], 0xff, v18{{[[:space:]]+//}}
// DISASM: v_bfe_u32 [[HI_A:v[0-9]+]], v0, 8, 8
// DISASM: v_bfe_u32 [[HI_B:v[0-9]+]], v18, 8, 8
// DISASM: v_lshl_or_b32 [[HI_B]], {{.*}}, 24, [[HI_B]]
// DISASM-NEXT: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb {{.*}}
// DISASM-NEXT: v_wmma_scale_f32_16x16x128_f8f6f4 v[38:45], v[2:9], v[190:197] /*v[702:709]*/, 0, [[LO_A]], [[LO_B]]
// DISASM: v_mov_b32_e32 v9, v181
// DISASM-NEXT: v_wmma_scale_f32_16x16x128_f8f6f4 v[38:45], v[2:9], v[190:197] /*v[702:709]*/, v[38:45], [[HI_A]], [[HI_B]]
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb 0x80a
// DISASM-NEXT: v_mov_b32_e32 v2, v176 /*v688*/
// DISASM: v_and_b32_e32 [[LO_A_2:v[0-9]+]], 0xff, v0{{[[:space:]]+//}}
// DISASM: v_and_b32_e32 [[LO_B_2:v[0-9]+]], 0xff, v18{{[[:space:]]+//}}
// DISASM: v_bfe_u32 [[HI_A_2:v[0-9]+]], v0, 8, 8
// DISASM: v_bfe_u32 [[HI_B_2:v[0-9]+]], v18, 8, 8
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[38:45], v[2:9], v[190:197] /*v[702:709]*/, 0, [[LO_A_2]], [[LO_B_2]]
// DISASM: v_mov_b32_e32 v9, v181
// DISASM-NEXT: v_wmma_scale_f32_16x16x128_f8f6f4 v[38:45], v[2:9], v[190:197] /*v[702:709]*/, v[38:45], [[HI_A_2]], [[HI_B_2]]
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb 0x80a
// DISASM-NEXT: v_mov_b32_e32 v2, v176 /*v688*/
// DISASM: v_mov_b32_e32 v14, v188 /*v700*/
// DISASM-NEXT: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb 0xa00

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_scale16_large_vgpr_count
.p2align 8
.type test_wmma_scale16_large_vgpr_count,@function
test_wmma_scale16_large_vgpr_count:
  s_set_vgpr_msb 0x100
  v_mov_b32 v10, v10
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[38:45], v[174:181], v[254:261], 0, v[0:1], v[18:19] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[38:45], v[174:181], v[240:247], 0, v[0:1], v[18:19] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
  s_endpgm
.Ltest_wmma_scale16_large_vgpr_count_end:
.size test_wmma_scale16_large_vgpr_count, .Ltest_wmma_scale16_large_vgpr_count_end-test_wmma_scale16_large_vgpr_count

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_large_vgpr_count
  .amdhsa_next_free_vgpr 688
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale16_large_vgpr_count
      .symbol: test_wmma_scale16_large_vgpr_count.kd
      .sgpr_count: 2
      .vgpr_count: 688
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
