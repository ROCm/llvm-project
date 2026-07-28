// COM: M=32 Scale16 prefix operands always address bank-zero VGPRs even when
// COM: matrix SRC0/SRC1 select nonzero physical banks through VGPR-MSB. Every
// COM: generated mode transition must also drain XCNT before remapping VGPRs.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: wmma_scale16: exact M+K split
// API: wmma_scale16: exact M+K split
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-NOT: v_wmma_scale16
// DISASM-NOT: v_perm_b32
// DISASM: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb
// DISASM-NEXT: v_and_b32{{(_e32)?}} v250, 0xff, v40
// DISASM: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb
// DISASM: v_bfe_u32 v254, v40, 16, 8
// DISASM: v_bfe_u32 v252, v40, 8, 8
// DISASM: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb
// DISASM-NEXT: v_mov_b32{{(_e32)?}} v242, v16{{.*}}v272
// DISASM: s_wait_xcnt 0x0
// DISASM-NEXT: s_set_vgpr_msb
// DISASM-NEXT: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[242:249], {{.*}}, v[0:7], v250, v251
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[242:249], {{.*}}, v[0:7], v252, v253
//
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 {{.*}}, 1.0, v56, v57{{.*}}neg_hi:[0,0,1]
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 {{.*}}, v[0:7], v58, v59
// DISASM-NOT: neg_hi
// DISASM: v_nop

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl test_m32_scale_bank_zero
.p2align 8
.type test_m32_scale_bank_zero,@function
test_m32_scale_bank_zero:
  s_set_vgpr_msb 0x5
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[0:15], v[40:41], v[42:43] matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
  s_endpgm
.Ltest_m32_scale_bank_zero_end:
.size test_m32_scale_bank_zero, .Ltest_m32_scale_bank_zero_end-test_m32_scale_bank_zero

.globl test_m32_immediate_neg_hi
.p2align 8
.type test_m32_immediate_neg_hi,@function
test_m32_immediate_neg_hi:
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], 1.0, v[40:41], v[42:43] matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3 neg_hi:[0,0,1]
  s_endpgm
.Ltest_m32_immediate_neg_hi_end:
.size test_m32_immediate_neg_hi, .Ltest_m32_immediate_neg_hi_end-test_m32_immediate_neg_hi

.rodata
.p2align 8
.amdhsa_kernel test_m32_scale_bank_zero
  .amdhsa_next_free_vgpr 304
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdhsa_kernel test_m32_immediate_neg_hi
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_m32_scale_bank_zero
      .symbol: test_m32_scale_bank_zero.kd
      .sgpr_count: 2
      .vgpr_count: 304
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: test_m32_immediate_neg_hi
      .symbol: test_m32_immediate_neg_hi.kd
      .sgpr_count: 2
      .vgpr_count: 44
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
