// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// DISASM-LABEL: <test_split_and_hazard>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
// DISASM:       v_wmma_i32_16x16x64_iu8
// DISASM-NEXT:  s_branch
// DISASM:       s_endpgm
.globl test_split_and_hazard
.p2align 8
.type test_split_and_hazard,@function
test_split_and_hazard:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  v_add_f32 v16, v0, v1
  s_endpgm
.Ltest_split_and_hazard_end:
.size test_split_and_hazard, .Ltest_split_and_hazard_end-test_split_and_hazard

// DISASM:       v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], v[32:39]
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]
// DISASM-NEXT:  s_branch

// DISASM-COUNT-8: v_nop
// DISASM-NEXT:  v_add_f32

.rodata
.p2align 8
.amdhsa_kernel test_split_and_hazard
  .amdhsa_next_free_vgpr 48
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
