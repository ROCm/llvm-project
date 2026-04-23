// COM: Test HotSwap trampoline patches: ds_*_2addr_stride64 expansion and
// COM: tensor_load_to_lds multicast fix (dead-SGPR and live-SGPR variants).

// COM: -- Test 1: ds_load_2addr_stride64_b32 -> two ds_load_b32 --------------

// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n.text\n.globl test_ds_kernel\n.p2align 8\n.type test_ds_kernel,@function\ntest_ds_kernel:\n ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3\n s_wait_dscnt 0x0\n s_endpgm\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n.Ltest_ds_kernel_end:\n.size test_ds_kernel, .Ltest_ds_kernel_end-test_ds_kernel\n.rodata\n.p2align 8\n.amdhsa_kernel test_ds_kernel\n .amdhsa_next_free_vgpr 3\n .amdhsa_next_free_sgpr 1\n.end_amdhsa_kernel\n' > %t1.s

// RUN: hotswap-trampoline %t1.s \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 %t1.out.elf \
// RUN:   | %FileCheck --check-prefix=DS-API %s

// DS-API: REWRITE: SUCCESS
// DS-API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t1.out.elf | %FileCheck --check-prefix=DS %s

// COM: The dual-address instruction should be gone
// DS-NOT: ds_load_2addr_stride64_b32

// COM: Two single-address ds_load_b32 instructions should appear
// DS-DAG: ds_load_b32 v0
// DS-DAG: ds_load_b32 v1

// COM: -- Test 2: tensor_load_to_lds with dead SGPR (s_endpgm follows) -------

// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n.text\n.globl test_tensor_dead\n.p2align 8\n.type test_tensor_dead,@function\ntest_tensor_dead:\n tensor_load_to_lds s[0:3], s[4:11]\n s_endpgm\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n.Ltest_tensor_dead_end:\n.size test_tensor_dead, .Ltest_tensor_dead_end-test_tensor_dead\n.rodata\n.p2align 8\n.amdhsa_kernel test_tensor_dead\n .amdhsa_next_free_vgpr 1\n .amdhsa_next_free_sgpr 12\n.end_amdhsa_kernel\n' > %t2.s

// RUN: hotswap-trampoline %t2.s \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 %t2.out.elf \
// RUN:   | %FileCheck --check-prefix=DEAD-API %s

// DEAD-API: REWRITE: SUCCESS
// DEAD-API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t2.out.elf | %FileCheck --check-prefix=DEAD %s

// COM: s_pack_hh should appear (clears multicast routing bits)
// DEAD-DAG: s_pack_hh_b32_b16

// COM: The original tensor_load_to_lds should still be present (in trampoline)
// DEAD-DAG: tensor_load_to_lds

// COM: No save/restore needed for dead SGPR
// DEAD-NOT: v_writelane_b32
// DEAD-NOT: v_readlane_b32

// COM: -- Test 3: tensor_load_to_lds with live SGPR (s4 used afterwards) ------

// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n.text\n.globl test_tensor_live\n.p2align 8\n.type test_tensor_live,@function\ntest_tensor_live:\n tensor_load_to_lds s[0:3], s[4:11]\n s_mov_b32 s0, s4\n s_endpgm\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n s_nop 0\n.Ltest_tensor_live_end:\n.size test_tensor_live, .Ltest_tensor_live_end-test_tensor_live\n.rodata\n.p2align 8\n.amdhsa_kernel test_tensor_live\n .amdhsa_next_free_vgpr 1\n .amdhsa_next_free_sgpr 12\n.end_amdhsa_kernel\n' > %t3.s

// RUN: hotswap-trampoline %t3.s \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 %t3.out.elf \
// RUN:   | %FileCheck --check-prefix=LIVE-API %s

// LIVE-API: REWRITE: SUCCESS
// LIVE-API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t3.out.elf | %FileCheck --check-prefix=LIVE %s

// COM: Save, pack, original instruction, and restore should all appear
// LIVE-DAG: v_writelane_b32
// LIVE-DAG: s_pack_hh_b32_b16
// LIVE-DAG: tensor_load_to_lds
// LIVE-DAG: v_readlane_b32
