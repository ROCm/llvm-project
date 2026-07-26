// COM: Complete descriptor/metadata coverage with explicit mixed B0/A0
// COM: revisions is ambiguous. A generic rewrite must fail rather than split
// COM: an A0 DS2 instruction and rescale its byte offsets.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK: hotswap: error: gfx1250 revision metadata is mixed across
// CHECK-SAME: complete kernel coverage.
// CHECK: hotswap: error: generic gfx1250 source has ambiguous
// CHECK-SAME: .gfx1250_revision state; refusing the rewrite
// CHECK: RESULT: ERROR

// COM: Explicit stepping remains caller-authoritative and bypasses the generic
// COM: metadata classifier.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.explicit.elf | %FileCheck --check-prefix=EXPLICIT %s
// EXPLICIT: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_b0
.p2align 8
.type test_b0,@function
test_b0:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:0 offset1:1
  s_endpgm
.Ltest_b0_end:
.size test_b0, .Ltest_b0_end-test_b0

.globl test_a0
.p2align 8
.type test_a0,@function
test_a0:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:0 offset1:8
  s_endpgm
.Ltest_a0_end:
.size test_a0, .Ltest_a0_end-test_a0

.rodata
.p2align 8
.amdhsa_kernel test_b0
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
.amdhsa_kernel test_a0
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_b0
      .symbol: test_b0.kd
      .gfx1250_revision: B0
      .sgpr_count: 1
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_a0
      .symbol: test_a0.kd
      .gfx1250_revision: A0
      .sgpr_count: 1
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
