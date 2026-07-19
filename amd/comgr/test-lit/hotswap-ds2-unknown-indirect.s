// COM: An unresolved indirect transfer may enter any instruction in the
// COM: object. It therefore prevents the long-range proof from assuming that
// COM: an aligned address definition actively dominates a later DS2 use.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <guarded_ds2>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <unknown_indirect>:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl guarded_ds2
.p2align 8
.type guarded_ds2,@function
guarded_ds2:
  v_mul_lo_u32 v4, v0, 56
  s_and_saveexec_b32 s4, s5
  s_or_b32 exec_lo, exec_lo, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_endpgm
  .rept 8
    s_nop 0
  .endr
.size guarded_ds2, .-guarded_ds2

.globl unknown_indirect
.p2align 8
.type unknown_indirect,@function
unknown_indirect:
  s_set_pc_i64 s[8:9]
.size unknown_indirect, .-unknown_indirect
