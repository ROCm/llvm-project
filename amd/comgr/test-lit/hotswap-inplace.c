// COM: Test HotSwap in-place patches: cluster_load -> global_load and
// COM: s_clause -> s_nop.

// COM: -- Test 1: cluster_load_b32 + cluster_load_b128 + s_clause -----------

// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n.text\n.globl test_inplace_kernel\n.p2align 8\n.type test_inplace_kernel,@function\ntest_inplace_kernel:\n  cluster_load_b32 v0, v[2:3], off\n  s_wait_loadcnt 0x0\n  cluster_load_b128 v[4:7], v[8:9], off\n  s_wait_loadcnt 0x0\n  s_clause 0x1\n  global_load_b32 v10, v[2:3], off\n  global_load_b32 v11, v[2:3], off offset:4\n  s_wait_loadcnt 0x0\n  s_endpgm\n.Ltest_inplace_kernel_end:\n.size test_inplace_kernel, .Ltest_inplace_kernel_end-test_inplace_kernel\n.rodata\n.p2align 8\n.amdhsa_kernel test_inplace_kernel\n  .amdhsa_next_free_vgpr 12\n  .amdhsa_next_free_sgpr 2\n.end_amdhsa_kernel\n' > %t.s

// RUN: hotswap-inplace %t.s \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s

// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: cluster_load mnemonics should be gone, replaced by global_load
// DISASM-NOT: cluster_load_b32
// DISASM-NOT: cluster_load_b128

// COM: s_clause should be gone, replaced by s_nop
// DISASM-NOT: s_clause

// COM: Replacement global_load instructions should be present
// DISASM-DAG: global_load_b32 v0
// DISASM-DAG: global_load_b128 v[4:7]

// COM: The s_nop replacement for s_clause
// DISASM-DAG: s_nop

// COM: Original global_load instructions should still be there
// DISASM-DAG: global_load_b32 v10
// DISASM-DAG: global_load_b32 v11

// COM: -- Test 2: kernel with NO cluster_load or s_clause (passthrough) ------

// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n.text\n.globl test_noop_kernel\n.p2align 8\n.type test_noop_kernel,@function\ntest_noop_kernel:\n  global_load_b32 v0, v[2:3], off\n  s_wait_loadcnt 0x0\n  s_endpgm\n.Ltest_noop_kernel_end:\n.size test_noop_kernel, .Ltest_noop_kernel_end-test_noop_kernel\n.rodata\n.p2align 8\n.amdhsa_kernel test_noop_kernel\n  .amdhsa_next_free_vgpr 4\n  .amdhsa_next_free_sgpr 2\n.end_amdhsa_kernel\n' > %t2.s

// RUN: hotswap-inplace %t2.s \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 %t2.out.elf \
// RUN:   | %FileCheck --check-prefix=NOOP %s

// NOOP: REWRITE: SUCCESS
// NOOP: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t2.out.elf | %FileCheck --check-prefix=DISASM2 %s

// COM: No cluster_load or s_clause -- nothing should be patched
// DISASM2-NOT: cluster_load
// DISASM2-NOT: s_clause
// DISASM2: global_load_b32 v0
// DISASM2: s_endpgm

// COM: -- Test 3: cluster_load_b64 variant ------------------------------------

// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n.text\n.globl test_b64_kernel\n.p2align 8\n.type test_b64_kernel,@function\ntest_b64_kernel:\n  cluster_load_b64 v[0:1], v[2:3], off\n  s_wait_loadcnt 0x0\n  s_endpgm\n.Ltest_b64_kernel_end:\n.size test_b64_kernel, .Ltest_b64_kernel_end-test_b64_kernel\n.rodata\n.p2align 8\n.amdhsa_kernel test_b64_kernel\n  .amdhsa_next_free_vgpr 4\n  .amdhsa_next_free_sgpr 2\n.end_amdhsa_kernel\n' > %t3.s

// RUN: hotswap-inplace %t3.s \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 %t3.out.elf \
// RUN:   | %FileCheck --check-prefix=B64 %s

// B64: REWRITE: SUCCESS

// RUN: %llvm-objdump -d %t3.out.elf | %FileCheck --check-prefix=DISASM3 %s

// COM: cluster_load_b64 should be swapped to global_load_b64
// DISASM3-NOT: cluster_load_b64
// DISASM3-DAG: global_load_b64 v[0:1]
