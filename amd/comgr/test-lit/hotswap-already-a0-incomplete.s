// COM: An A0 tag on only some kernels is not a completion certificate. Missing
// COM: revision metadata must take the normal analysis path, even if no rule
// COM: ultimately matches this small object.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck %s
// CHECK-NOT: every kernel already reports gfx1250 revision A0
// CHECK: applied 0 instruction patches
// CHECK: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl tagged
.p2align 8
.type tagged,@function
tagged:
  s_endpgm
.size tagged, .-tagged

.globl untagged
.p2align 8
.type untagged,@function
untagged:
  s_endpgm
.size untagged, .-untagged

.rodata
.p2align 8
.amdhsa_kernel tagged
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel untagged
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: tagged
      .symbol: tagged.kd
      .gfx1250_revision: A0
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: untagged
      .symbol: untagged.kd
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
