// COM: A far patch must fail closed when every candidate pair is live at the
// COM: resume point and no globally unused SGPR pair remains.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK: hotswap: error: safe far return: no aligned block of 2 safe SGPRs fits below s106
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far_no_sgpr
.p2align 8
.type test_far_no_sgpr,@function
test_far_no_sgpr:
  s_cbranch_scc1 .Lresume
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lresume:
  // No numbered pair is defined before exit, and the original function does
  // not use VCC. The rewrite therefore cannot introduce VCC as scratch.
  s_endpgm
.Ltest_far_no_sgpr_end:
.size test_far_no_sgpr, .Ltest_far_no_sgpr_end-test_far_no_sgpr

// Non-NOP far filler leaves no local replacement sled or forward gateway.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_far_no_sgpr
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_far_no_sgpr
      .symbol: test_far_no_sgpr.kd
      .sgpr_count: 106
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
