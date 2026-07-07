// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_bf8
// DISASM:       s_branch
// DISASM:       s_endpgm

// DISASM:       v_wmma_f16_16x16x64_fp8_bf8 v[32:35], v[0:7], v[16:23], v[32:35] neg_hi:[0,0,1]
// DISASM-NEXT:  v_wmma_f16_16x16x64_fp8_bf8 v[32:35], v[8:15], v[24:31], v[32:35]{{[[:space:]]*\/\/}}
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f16_16x16x128_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] neg_hi:[0,0,1]
  s_endpgm
.size kernel, .-kernel

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
