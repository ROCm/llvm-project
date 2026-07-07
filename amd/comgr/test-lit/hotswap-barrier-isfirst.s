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
// DISASM-NOT: s_barrier_signal_isfirst
// DISASM: s_barrier_signal -1
// DISASM-NEXT: s_barrier_wait 0xffff
// DISASM-NEXT: s_barrier_signal -3
// DISASM-NEXT: s_barrier_wait 0xfffd
// DISASM-NEXT: s_endpgm
// DISASM-NOT: s_barrier_signal_isfirst
// API2: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_barrier_isfirst
.p2align 8
.type test_barrier_isfirst,@function
test_barrier_isfirst:
  // Two isfirst sites with different barrier IDs; both must be swapped
  // to s_barrier_signal in place with their operand values preserved.
  s_barrier_signal_isfirst -1
  s_barrier_wait -1
  s_barrier_signal_isfirst -3
  s_barrier_wait -3
  s_endpgm
.Ltest_barrier_isfirst_end:
.size test_barrier_isfirst, .Ltest_barrier_isfirst_end-test_barrier_isfirst

.rodata
.p2align 8
.amdhsa_kernel test_barrier_isfirst
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
