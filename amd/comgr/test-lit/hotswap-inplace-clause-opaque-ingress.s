// COM: Opaque PC-valued control flow in any descriptor-backed function can
// COM: enter any instruction, so it globally invalidates the prior-VMEM proof.

// RUN: %clang -x assembler-with-cpp -DCASE=0 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.rfe.elf
// RUN: hotswap-rewrite %t.rfe.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.rfe.out.elf | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.rfe.out.elf | %FileCheck %s

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.unknown.elf
// RUN: hotswap-rewrite %t.unknown.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.unknown.out.elf | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.unknown.out.elf | %FileCheck %s

// API: RESULT: SUCCESS

// CHECK-LABEL: <opaque_ingress_victim>:
// CHECK-NEXT: global_wb
// CHECK-NEXT: v_nop
// CHECK-NEXT: global_load_b32 v0
// CHECK-NEXT: s_wait_loadcnt 0x0
// CHECK-NEXT: s_nop 0
// CHECK-NEXT: global_load_b32 v1

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl opaque_ingress_source
.p2align 8
.type opaque_ingress_source,@function
opaque_ingress_source:
#if CASE == 0
  s_rfe_i64 s[0:1]
#elif CASE == 1
  .long 0xffffffff
#endif
  s_endpgm
.Lopaque_ingress_source_end:
.size opaque_ingress_source, .Lopaque_ingress_source_end-opaque_ingress_source

.globl opaque_ingress_victim
.p2align 8
.type opaque_ingress_victim,@function
opaque_ingress_victim:
  global_wb scope:SCOPE_CU
  v_nop
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0
  s_clause 0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0
  s_endpgm
.Lopaque_ingress_victim_end:
.size opaque_ingress_victim, .Lopaque_ingress_victim_end-opaque_ingress_victim

.rodata
.p2align 8
.amdhsa_kernel opaque_ingress_source
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel opaque_ingress_victim
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: opaque_ingress_source
      .symbol: opaque_ingress_source.kd
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: opaque_ingress_victim
      .symbol: opaque_ingress_victim.kd
      .sgpr_count: 0
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
