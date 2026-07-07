// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// RUN: cmp %t.out.elf %t.out2.elf

// API: RESULT: SUCCESS
// DISASM: v_wmma_i32_16x16x64_iu8
// DISASM-NEXT: s_branch
// DISASM: s_endpgm
// DISASM-COUNT-8: v_nop
// DISASM-NEXT: v_add_f32
// API2: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_hazard
.p2align 8
.type test_wmma_hazard,@function
test_wmma_hazard:
  // WMMA integer instruction: A0 needs 8 nops, B0 needs 4
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  // VALU that overlaps WMMA dest (writes v16) -- should trigger hazard
  v_add_f32 v16, v0, v1
  s_endpgm
.Ltest_wmma_hazard_end:
.size test_wmma_hazard, .Ltest_wmma_hazard_end-test_wmma_hazard

.rodata
.p2align 8
.amdhsa_kernel test_wmma_hazard
  .amdhsa_next_free_vgpr 24
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
