// COM: A metadata revision entry without any matching kernel descriptor is not
// COM: an object-wide DS2 marker. Because a marker is present, the generic
// COM: source is ambiguous and rewriting must fail rather than rescale it.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK: hotswap: error: gfx1250 revision markers are present but the
// CHECK-SAME: code object has no kernel descriptors.
// CHECK: hotswap: error: generic gfx1250 source has ambiguous
// CHECK-SAME: .gfx1250_revision state; refusing the rewrite
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_descriptorless
.p2align 8
.type test_ds2_descriptorless,@function
test_ds2_descriptorless:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:0 offset1:1
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds2_descriptorless_end:
.size test_ds2_descriptorless, .Ltest_ds2_descriptorless_end-test_ds2_descriptorless

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_ds2_descriptorless
      .symbol: test_ds2_descriptorless.kd
      .gfx1250_revision: B0
      .sgpr_count: 1
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
