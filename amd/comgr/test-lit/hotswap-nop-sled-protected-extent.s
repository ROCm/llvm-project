// COM: A sized OBJECT whose point starts before post-function NOP padding but
// COM: whose extent covers that padding prevents it from donating replacement
// COM: body storage. The later kernel must use the appended pool instead.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: %llvm-readelf -s %t.elf | %FileCheck --check-prefix=SYMBOL %s
// SYMBOL: OBJECT LOCAL DEFAULT {{[0-9]+}} protected_extent
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <protected_extent>:
// DISASM-NEXT: {{[0-9a-f]+}}: 00 00 b0 bf 00 00 80 bf
// DISASM-LABEL: <second_kernel>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: s_endpgm
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_branch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl first_kernel
.p2align 8
.type first_kernel,@function
first_kernel:
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.local protected_extent
.type protected_extent,@object
protected_extent:
  s_endpgm
.Lfirst_kernel_end:
.size first_kernel, .Lfirst_kernel_end-first_kernel

.globl second_kernel
.p2align 8
.size protected_extent, .-protected_extent
.type second_kernel,@function
second_kernel:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.Lsecond_kernel_end:
.size second_kernel, .Lsecond_kernel_end-second_kernel

.rodata
.p2align 8
.amdhsa_kernel first_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel second_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
