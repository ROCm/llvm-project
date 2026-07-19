// COM: A strictly proven explicit-NOP run after first_kernel has one physical
// COM: allocation cursor. The first DS2 rewrite consumes it as owner-local
// COM: storage; the second kernel may consume the remaining bytes only through
// COM: the certified DS2 relocation-body policy. The bodies must not overlap.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <first_kernel>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: ds_load_b32 v0
// DISASM-NEXT: ds_load_b32 v1
// DISASM-NEXT: s_branch
// DISASM-NEXT: ds_store_b32 v2, v0
// DISASM-NEXT: ds_store_b32 v2, v1
// DISASM-NEXT: s_branch
// DISASM-LABEL: <second_kernel>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl first_kernel
.p2align 8
.type first_kernel,@function
first_kernel:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0
  s_endpgm
.size first_kernel, .-first_kernel

.globl second_kernel
.p2align 8
.type second_kernel,@function
second_kernel:
  ds_store_2addr_stride64_b32 v2, v0, v1 offset0:1 offset1:3
  s_wait_dscnt 0
  s_endpgm
.size second_kernel, .-second_kernel

.rodata
.p2align 8
.amdhsa_kernel first_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel second_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: first_kernel
      .symbol: first_kernel.kd
      .sgpr_count: 1
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: second_kernel
      .symbol: second_kernel.kd
      .sgpr_count: 1
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
