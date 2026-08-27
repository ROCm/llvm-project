// Verify that the Scale16 lowering stays occupancy-safe at the boundary where
// its above-KD scratch decides whether a required patch can be applied at all.
//
// A 96-VGPR wave32 kernel with max_flat_workgroup_size 1024 needs 8 waves/EU to
// admit one maximum-size workgroup, which caps the rewrite at 128 allocated
// VGPRs. The FP8 lane-mask path needs one A-width-plus-5 low-bank block, so it
// only fits if the lowering reserves nothing it does not use: matrix B is
// already addressed by the scratch bank and must not be copied, and a low-bank
// block that borrowed no live register must not reserve save slots.
//
// Reserving either one pushes the request to 134 logical / 144 allocated VGPRs,
// which drops the kernel to 7 waves/EU and fails the required patch.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// COM: The required patch must apply rather than fail the occupancy check.
// API-NOT: would grow VGPRs
// API-NOT: error:
// COM: B is consumed where it is, and no save slots are reserved.
// API: wmma_scale16: exact K-split{{.*}}B in place=v64:79{{.*}}+21 vgpr
// API: RESULT: SUCCESS

// COM: 96 + 21 = 117 logical, which rounds to the 128 allocated VGPRs that
// COM: still admit 8 waves/EU. A copied B or unused save block would exceed it.
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=META %s
// META: .max_flat_workgroup_size: 1024
// META: .vgpr_count:     117

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_scale16_occupancy_boundary>:
// DISASM-NOT: v_wmma_scale16
// COM: Both replacement passes read the original matrix B range directly, so
// COM: neither depends on an above-KD copy of it.
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[96:111], v[64:79], v[80:87], v112, v113
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[96:111], v[64:79], v[80:87], v114, v115

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_scale16_occupancy_boundary
.p2align 8
.type test_scale16_occupancy_boundary,@function
test_scale16_occupancy_boundary:
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[80:87], v[48:63], v[64:79], v[80:87], v[88:89], v[90:91] matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
  s_endpgm
.Ltest_scale16_occupancy_boundary_end:
.size test_scale16_occupancy_boundary, .Ltest_scale16_occupancy_boundary_end-test_scale16_occupancy_boundary

.rodata
.p2align 8
.amdhsa_kernel test_scale16_occupancy_boundary
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr 16
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_scale16_occupancy_boundary
      .symbol: test_scale16_occupancy_boundary.kd
      .sgpr_count: 16
      .vgpr_count: 96
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 1024
.end_amdgpu_metadata
