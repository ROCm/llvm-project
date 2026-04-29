// COM: Test HotSwap trampoline patch: ds_load_2addr_stride64_b32 expansion
// COM: into two ds_load_b32 instructions with s_wait_dscnt bump.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The dual-address instruction should be gone
// DISASM-NOT: ds_load_2addr_stride64_b32

// COM: The wait count should be bumped from 0x0 to 0x1
// DISASM: s_wait_dscnt 0x1

// COM: Two single-address ds_load_b32 instructions should appear
// DISASM-DAG: ds_load_b32 v0
// DISASM-DAG: ds_load_b32 v1

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_kernel
.p2align 8
.type test_ds_kernel,@function
test_ds_kernel:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_kernel_end:
.size test_ds_kernel, .Ltest_ds_kernel_end-test_ds_kernel

.rodata
.p2align 8
.amdhsa_kernel test_ds_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
