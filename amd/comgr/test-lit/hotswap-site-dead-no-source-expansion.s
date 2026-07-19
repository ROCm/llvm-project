// COM: A site-dead SGPR pair is safe only at the original resume point. The
// COM: far trampoline must not move the following pure definition into its
// COM: source window, because the set-PC return would then overwrite it.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: safe far return: reusing original site-dead s[62:63]
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <site_dead_no_source_expansion>:
// DISASM-NEXT: s_call_i64 s[62:63],
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_mov_b64 s[62:63], 0

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl site_dead_no_source_expansion
.p2align 8
.type site_dead_no_source_expansion,@function
site_dead_no_source_expansion:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_mov_b64 s[62:63], 0
.rept 4
  s_nop 0
.endr
  s_cmp_lg_u64 s[62:63], 0
  s_endpgm
.size site_dead_no_source_expansion, .-site_dead_no_source_expansion

// Safe nearby gateway space followed by enough non-NOP code to put the
// appended trampoline pool outside short-branch reach.
.rept 8
  s_nop 0
.endr
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel site_dead_no_source_expansion
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: site_dead_no_source_expansion
      .symbol: site_dead_no_source_expansion.kd
      .sgpr_count: 106
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
