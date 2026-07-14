// COM: Initial-VMEM analysis must model entries and exits that cross a kernel
// COM: function boundary. A direct external entry can bypass the kernel's
// COM: prior VMEM, while an unrecognized external branch is not proof that a
// COM: textually later clause is unreachable.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_external_ingress>:
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1, v[4:5], off

// DISASM-LABEL: <test_unresolved_external_branch>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[2:3], off

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.local external_ingress_source
.p2align 8
.type external_ingress_source,@function
external_ingress_source:
  s_branch .Ltest_external_ingress_clause
.Lexternal_ingress_source_end:
.size external_ingress_source, .Lexternal_ingress_source_end-external_ingress_source

.globl test_external_ingress
.p2align 8
.type test_external_ingress,@function
test_external_ingress:
  global_wb scope:SCOPE_CU
  v_nop
  global_load_b32 v0, v[2:3], off
.Ltest_external_ingress_clause:
  s_clause 0x0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_external_ingress_end:
.size test_external_ingress, .Ltest_external_ingress_end-test_external_ingress

.globl test_unresolved_external_branch
.p2align 8
.type test_unresolved_external_branch,@function
test_unresolved_external_branch:
  s_branch external_exit
  s_clause 0x0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_unresolved_external_branch_end:
.size test_unresolved_external_branch, .Ltest_unresolved_external_branch_end-test_unresolved_external_branch

.local external_exit
.p2align 8
.type external_exit,@function
external_exit:
  s_endpgm
.Lexternal_exit_end:
.size external_exit, .Lexternal_exit_end-external_exit

.rodata
.p2align 8
.amdhsa_kernel test_external_ingress
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_unresolved_external_branch
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_external_ingress
      .symbol: test_external_ingress.kd
      .sgpr_count: 0
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_unresolved_external_branch
      .symbol: test_unresolved_external_branch.kd
      .sgpr_count: 0
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
