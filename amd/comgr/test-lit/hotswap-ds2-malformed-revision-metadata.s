// COM: A present but unknown revision value is malformed, not a legacy
// COM: all-missing marker state. Generic rewriting must fail before touching
// COM: the representable DS2 instruction.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK: hotswap: error: metadata cache: .gfx1250_revision for
// CHECK-SAME: 'test_malformed' is 'X0', expected B0 or A0.
// CHECK: hotswap: error: gfx1250 revision state is ambiguous because
// CHECK-SAME: revision metadata is malformed.
// CHECK: hotswap: error: generic gfx1250 source has ambiguous
// CHECK-SAME: .gfx1250_revision state; refusing the rewrite
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_malformed
.p2align 8
.type test_malformed,@function
test_malformed:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:0 offset1:1
  s_endpgm
.Ltest_malformed_end:
.size test_malformed, .Ltest_malformed_end-test_malformed

.rodata
.p2align 8
.amdhsa_kernel test_malformed
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_malformed
      .symbol: test_malformed.kd
      .gfx1250_revision: X0
      .sgpr_count: 1
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
