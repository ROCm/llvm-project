// COM: A far device-function patch may use VCC only when the original
// COM: function already requires VCC and both halves are dead at the resume.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: safe far return: reusing site-dead vcc
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <site_dead_vcc>:
// DISASM-NEXT: s_get_pc_i64 vcc
// DISASM-NEXT: s_add_nc_u64 vcc, vcc,
// DISASM-NEXT: s_set_pc_i64 vcc
// DISASM-NEXT: s_set_pc_i64 s[30:31]
// DISASM: s_mov_b32 vcc_lo, 0
// DISASM: v_cmp_lt_u32_e32 vcc_lo
// DISASM-NEXT: s_get_pc_i64 vcc
// DISASM: s_set_pc_i64 vcc

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl site_dead_vcc
.p2align 8
.type site_dead_vcc,@function
site_dead_vcc:
  s_mov_b32 vcc_lo, 0
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  v_cmp_lt_u32_e32 vcc_lo, v0, v1
  s_set_pc_i64 s[30:31]
.size site_dead_vcc, .-site_dead_vcc

// Safe gateway space outside the function, followed by non-NOP far filler.
.rept 8
  s_nop 0
.endr
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel site_dead_vcc
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: site_dead_vcc
      .symbol: site_dead_vcc.kd
      .sgpr_count: 108
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
