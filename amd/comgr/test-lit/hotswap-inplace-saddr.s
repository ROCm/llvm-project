// COM: Test HotSwap in-place patch selectivity for cluster_load addressing
// COM: forms. The B0->A0 replacement templates target the saddr=off (64-bit
// COM: vaddr) encoding. The SGPR-relative (_SADDR) variant shares the display
// COM: mnemonic but is a distinct MC opcode with a different operand layout,
// COM: so reusing the off-form opcode would mis-encode its operands and corrupt
// COM: the address at runtime. The patcher must skip the _SADDR form and only
// COM: rewrite the off form.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The two loads appear in program order. The SGPR-relative (_SADDR) site
// COM: comes first and must be preserved verbatim as a cluster_load with an
// COM: s[...] base; if the patcher wrongly swapped it, this line would instead
// COM: disassemble as a global_load and the match would fail.
// DISASM: cluster_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// COM: Between the preserved _SADDR site and the rewritten off-form site, no
// COM: off-form cluster_load may survive (the off form must have been swapped).
// DISASM-NOT: cluster_load_b32 {{.*}}, off
// COM: The saddr=off site that followed is rewritten to global_load_b32,
// COM: proving the skip is specific to the _SADDR form, not a blanket opt-out.
// DISASM: global_load_b32 v{{[0-9]+}}, v[{{[0-9:]+}}], off

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_saddr_kernel
.p2align 8
.type test_saddr_kernel,@function
test_saddr_kernel:
  // SGPR-relative (SADDR) form -- must be left unchanged.
  cluster_load_b32 v4, v1, s[2:3]
  s_wait_loadcnt 0x0
  // saddr=off form -- must be swapped to global_load_b32.
  cluster_load_b32 v5, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_saddr_kernel_end:
.size test_saddr_kernel, .Ltest_saddr_kernel_end-test_saddr_kernel

.rodata
.p2align 8
.amdhsa_kernel test_saddr_kernel
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel
