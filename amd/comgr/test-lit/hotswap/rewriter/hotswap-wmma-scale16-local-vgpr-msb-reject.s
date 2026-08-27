// Negative counterpart to hotswap-wmma-scale16-local-vgpr-msb.s.
//
// The register-target call again declines object-wide VGPR-MSB recovery, but
// nothing in the straight-line run up to the Scale16 establishes a mode: the
// backward scan reaches the kernel entry without finding a setter. Neither
// recovery path can prove the incoming mode, so the required lowering must
// fail closed instead of assuming the ABI entry mode.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK: unresolved call target
// CHECK-NOT: exact K-split
// CHECK-LABEL: error: wmma_scale16{{.*}}cannot determine active VGPR-MSB mode
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_scale16_no_local_mode
.p2align 8
.type test_scale16_no_local_mode,@function
test_scale16_no_local_mode:
  s_nop 0
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[16:23], v[32:39], v[0:7], v[48:49], v[50:51] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
  s_swap_pc_i64 s[30:31], s[0:1]
  s_endpgm
.Ltest_scale16_no_local_mode_end:
.size test_scale16_no_local_mode, .Ltest_scale16_no_local_mode_end-test_scale16_no_local_mode

.rodata
.p2align 8
.amdhsa_kernel test_scale16_no_local_mode
  .amdhsa_next_free_vgpr 304
  .amdhsa_next_free_sgpr 32
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_scale16_no_local_mode
      .symbol: test_scale16_no_local_mode.kd
      .sgpr_count: 32
      .vgpr_count: 304
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
