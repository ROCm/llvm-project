// COM: Backward source growth does not move the resume point. A site-dead
// COM: scratch pair therefore remains safe when every prepended instruction
// COM: does not read the pair's clobbered value. A pure definition may be
// COM: copied because it reconstructs the original state. This provides the
// COM: complete direct
// COM: set-PC source window without requiring nearby gateway padding.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: safe far return: reusing original site-dead s[62:63]
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <site_dead_backward_expansion>:
// DISASM-NEXT: s_get_pc_i64 s[62:63]
// DISASM-NEXT: s_add_nc_u64 s[62:63], s[62:63],
// DISASM-NEXT: s_set_pc_i64 s[62:63]
// DISASM: v_add3_u32 v18, v10, v18, v11
// DISASM-NEXT: s_mov_b64 s[62:63], 0
// DISASM-NEXT: ds_load_b64 v[8:9], v6 offset:944
// DISASM-NEXT: ds_load_b64 v[6:7], v6 offset:880

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl site_dead_backward_expansion
.p2align 8
.type site_dead_backward_expansion,@function
site_dead_backward_expansion:
  v_add3_u32 v18, v10, v18, v11
  s_mov_b64 s[62:63], 0
  ds_load_2addr_b64 v[6:9], v6 offset0:110 offset1:118
  s_mov_b64 s[62:63], 0
  s_cmp_lg_u64 s[62:63], 0
  s_endpgm
.size site_dead_backward_expansion, .-site_dead_backward_expansion

// Put the appended trampoline pool beyond short-branch reach without donating
// any in-function NOP padding to the source.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel site_dead_backward_expansion
  .amdhsa_next_free_vgpr 19
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: site_dead_backward_expansion
      .symbol: site_dead_backward_expansion.kd
      .sgpr_count: 106
      .vgpr_count: 19
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
