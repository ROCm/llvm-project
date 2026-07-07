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
// DISASM-NOT: cluster_load_b32
// DISASM-NOT: cluster_load_b128
// DISASM-NOT: s_clause
// DISASM-DAG: global_load_b32 v0
// DISASM-DAG: global_load_b128 v[4:7]
// DISASM-DAG: s_nop
// DISASM-DAG: global_load_b32 v10
// DISASM-DAG: global_load_b32 v11
// API2: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_inplace_kernel
.p2align 8
.type test_inplace_kernel,@function
test_inplace_kernel:
  cluster_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  cluster_load_b128 v[4:7], v[8:9], off
  s_wait_loadcnt 0x0
  s_clause 0x1
  global_load_b32 v10, v[2:3], off
  global_load_b32 v11, v[2:3], off offset:4
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_inplace_kernel_end:
.size test_inplace_kernel, .Ltest_inplace_kernel_end-test_inplace_kernel

.rodata
.p2align 8
.amdhsa_kernel test_inplace_kernel
  .amdhsa_next_free_vgpr 12
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
