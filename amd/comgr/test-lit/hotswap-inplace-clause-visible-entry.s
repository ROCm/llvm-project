// COM: A loader-visible function nested in a descriptor-owned range is an
// COM: independent entry and cannot inherit the outer kernel's prior VMEM.

// RUN: %clang -x assembler-with-cpp -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --output %t.out.elf | \
// RUN:   %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck %s
// RUN: hotswap-rewrite %t.out.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --output %t.out2.elf
// RUN: cmp %t.out.elf %t.out2.elf

// API: RESULT: SUCCESS
// CHECK-LABEL: <visible_nested_entry>:
// CHECK-NEXT: s_nop 0
// CHECK-NEXT: global_load_b32 v0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl visible_entry_kernel
.p2align 8
.type visible_entry_kernel,@function
visible_entry_kernel:
  global_wb scope:SCOPE_CU
  v_nop
  s_branch .Lvisible_entry_done

.globl visible_nested_entry
.type visible_nested_entry,@function
visible_nested_entry:
  s_clause 0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0
  s_endpgm
.Lvisible_nested_entry_end:
.size visible_nested_entry, \
    .Lvisible_nested_entry_end-visible_nested_entry

.Lvisible_entry_done:
  s_endpgm
.Lvisible_entry_kernel_end:
.size visible_entry_kernel, .Lvisible_entry_kernel_end-visible_entry_kernel

.rodata
.p2align 8
.amdhsa_kernel visible_entry_kernel
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: visible_entry_kernel
      .symbol: visible_entry_kernel.kd
      .sgpr_count: 0
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
