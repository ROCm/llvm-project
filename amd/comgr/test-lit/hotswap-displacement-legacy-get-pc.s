// COM: Direct entry displacement must repair the older get-PC/add-with-carry/
// COM: set-PC jump sequence when a later kernel-entry prefix moves its target.
// COM: The repair preserves both forced-literal add widths and does not emit
// COM: the GFX1250 A0-incompatible s_add_pc_i64 instruction.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.out.elf \
// RUN:   | %FileCheck --check-prefix=DISASM \
// RUN:       --implicit-check-not=s_add_pc_i64 %s

// API: RESULT: SUCCESS

// DISASM-LABEL: <legacy_jump>:
// DISASM-NEXT: s_get_pc_i64 s[8:9]
// DISASM-NEXT: s_add_co_u32 s8, s8, 0x110
// DISASM-NEXT: s_add_co_ci_u32 s9, s9, lit(0x0)
// DISASM-NEXT: s_set_pc_i64 s[8:9]
// DISASM-LABEL: <entry_kernel>:
// DISASM-NEXT: global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl legacy_jump
.type legacy_jump,@function
legacy_jump:
  s_get_pc_i64 s[8:9]
  s_add_u32 s8, s8, lit(0x100)
  s_addc_u32 s9, s9, lit(0x0)
  s_set_pc_i64 s[8:9]
.Llegacy_jump_end:
.size legacy_jump, .Llegacy_jump_end-legacy_jump

.p2align 8
.globl entry_kernel
.type entry_kernel,@function
entry_kernel:
  s_endpgm
.Lentry_kernel_end:
.size entry_kernel, .Lentry_kernel_end-entry_kernel

.Llegacy_jump_target:
  s_endpgm

.rodata
.p2align 8
.amdhsa_kernel entry_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 10
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: entry_kernel
      .symbol: entry_kernel.kd
      .sgpr_count: 10
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
