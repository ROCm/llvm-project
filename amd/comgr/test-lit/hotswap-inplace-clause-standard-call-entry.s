// COM: An unresolved ABI-link call may enter any callable STT_FUNC. When an
// COM: interior callable lies inside an outer descriptor-owned range, it must
// COM: be a fresh initial-VMEM root rather than inheriting the outer VMEM.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: recognized ABI standard-link indirect call
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <standard_call_interior>:
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl standard_call_outer
.p2align 8
.type standard_call_outer,@function
standard_call_outer:
  global_wb scope:SCOPE_CU
  v_nop
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0
  s_endpgm

.local standard_call_interior
.type standard_call_interior,@function
standard_call_interior:
  s_clause 0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0
  s_set_pc_i64 s[30:31]
.Lstandard_call_outer_end:
.size standard_call_outer, .Lstandard_call_outer_end-standard_call_outer
.size standard_call_interior, .Lstandard_call_outer_end-standard_call_interior

.globl standard_call_source
.p2align 8
.type standard_call_source,@function
standard_call_source:
  global_wb scope:SCOPE_CU
  v_nop
  s_swap_pc_i64 s[30:31], s[2:3]
  s_endpgm
.Lstandard_call_source_end:
.size standard_call_source, .Lstandard_call_source_end-standard_call_source

.rodata
.p2align 8
.amdhsa_kernel standard_call_outer
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel standard_call_source
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: standard_call_outer
      .symbol: standard_call_outer.kd
      .sgpr_count: 32
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: standard_call_source
      .symbol: standard_call_source.kd
      .sgpr_count: 32
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
