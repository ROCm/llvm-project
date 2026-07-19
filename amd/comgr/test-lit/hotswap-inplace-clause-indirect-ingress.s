// COM: A genuinely arbitrary set-PC in any function can enter another kernel
// COM: after its prior VMEM. Initial-VMEM clause analysis must therefore fail
// COM: closed across every descriptor-backed range.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

// DISASM-LABEL: <indirect_ingress_victim>:
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: global_load_b32 v0
// DISASM-NEXT: s_wait_loadcnt 0x0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1

// DISASM-LABEL: <arbitrary_ingress_source>:
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_set_pc_i64 s[8:9]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl indirect_ingress_victim
.p2align 8
.type indirect_ingress_victim,@function
indirect_ingress_victim:
  global_wb scope:SCOPE_CU
  v_nop
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0
  s_clause 0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0
  s_endpgm
.Lindirect_ingress_victim_end:
.size indirect_ingress_victim, .Lindirect_ingress_victim_end-indirect_ingress_victim

.globl arbitrary_ingress_source
.p2align 8
.type arbitrary_ingress_source,@function
arbitrary_ingress_source:
  global_wb scope:SCOPE_CU
  v_nop
  s_set_pc_i64 s[8:9]
  global_load_b32 v2, v[6:7], off
  s_wait_loadcnt 0
  s_endpgm
.Larbitrary_ingress_source_end:
.size arbitrary_ingress_source, .Larbitrary_ingress_source_end-arbitrary_ingress_source

.rodata
.p2align 8
.amdhsa_kernel indirect_ingress_victim
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel arbitrary_ingress_source
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 10
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: indirect_ingress_victim
      .symbol: indirect_ingress_victim.kd
      .sgpr_count: 0
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: arbitrary_ingress_source
      .symbol: arbitrary_ingress_source.kd
      .sgpr_count: 10
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
