// COM: Uniform metadata entries are insufficient when metadata omits a real
// COM: kernel descriptor. The target-state fast path requires complete name
// COM: coverage of the descriptor set.

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
.globl listed
.p2align 8
.type listed,@function
listed:
  s_endpgm
.size listed, .-listed

.globl omitted
.p2align 8
.type omitted,@function
omitted:
  s_endpgm
.size omitted, .-omitted

.rodata
.p2align 8
.amdhsa_kernel listed
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel omitted
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: listed
      .symbol: listed.kd
      .gfx1250_revision: A0
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
