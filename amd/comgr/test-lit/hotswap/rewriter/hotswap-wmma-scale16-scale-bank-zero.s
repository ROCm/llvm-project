// COM: Scale16 prefix operands always address bank-zero VGPRs, even when the
// COM: matrix SRC0/SRC1 roles select nonzero banks through VGPR-MSB. Generated
// COM: scale values must therefore be produced in the same low-bank registers
// COM: consumed by each replacement WMMA.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=RESULT %s
// RESULT: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_scale16_scale_bank_zero>:
// DISASM-NOT: v_wmma_scale16
// COM: The gather reads the original scale tuples in bank zero and writes
// COM: directly addressable low-bank results. Each WMMA consumes those exact
// COM: result registers rather than their nonzero-bank aliases.
// DISASM: v_and_b32_e32 [[SCALE_A_LO:v[0-9]+]], 0xff, v48{{[[:space:]]+//}}
// DISASM: v_and_b32_e32 [[SCALE_B_LO:v[0-9]+]], 0xff, v50{{[[:space:]]+//}}
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 {{.*}}, [[SCALE_A_LO]], [[SCALE_B_LO]]

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_scale16_scale_bank_zero
.p2align 8
.type test_wmma_scale16_scale_bank_zero,@function
test_wmma_scale16_scale_bank_zero:
  s_set_vgpr_msb 0x5
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[16:23], v[32:39], v[0:7], v[48:49], v[50:51] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
  s_endpgm
.Ltest_wmma_scale16_scale_bank_zero_end:
.size test_wmma_scale16_scale_bank_zero, .Ltest_wmma_scale16_scale_bank_zero_end-test_wmma_scale16_scale_bank_zero

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_scale_bank_zero
  .amdhsa_next_free_vgpr 304
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale16_scale_bank_zero
      .symbol: test_wmma_scale16_scale_bank_zero.kd
      .sgpr_count: 2
      .vgpr_count: 304
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
