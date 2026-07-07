// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
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
// DISASM-NOT: cluster_load
// DISASM-NOT: s_clause
// DISASM: global_load_b32 v0
// DISASM: s_endpgm
// API2: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_noop_kernel
.p2align 8
.type test_noop_kernel,@function
test_noop_kernel:
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_noop_kernel_end:
.size test_noop_kernel, .Ltest_noop_kernel_end-test_noop_kernel

.rodata
.p2align 8
.amdhsa_kernel test_noop_kernel
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
