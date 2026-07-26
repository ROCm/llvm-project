// COM: Both kernel descriptors have metadata entries, but only one entry has a
// COM: revision marker. Once any marker is present, partial marker coverage is
// COM: ambiguous and a generic rewrite must fail before touching DS2 offsets.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK: hotswap: error: gfx1250 revision marker coverage is partial:
// CHECK-SAME: 1 marker(s) for 2 metadata kernel(s).
// CHECK: hotswap: error: generic gfx1250 source has ambiguous
// CHECK-SAME: .gfx1250_revision state; refusing the rewrite
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_marked
.p2align 8
.type test_marked,@function
test_marked:
  s_endpgm
.Ltest_marked_end:
.size test_marked, .Ltest_marked_end-test_marked

.globl test_unmarked
.p2align 8
.type test_unmarked,@function
test_unmarked:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:0 offset1:1
  s_endpgm
.Ltest_unmarked_end:
.size test_unmarked, .Ltest_unmarked_end-test_unmarked

.rodata
.p2align 8
.amdhsa_kernel test_marked
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
.amdhsa_kernel test_unmarked
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_marked
      .symbol: test_marked.kd
      .gfx1250_revision: B0
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_unmarked
      .symbol: test_unmarked.kd
      .sgpr_count: 1
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
