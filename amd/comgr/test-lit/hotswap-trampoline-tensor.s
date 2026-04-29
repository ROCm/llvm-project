// COM: Test HotSwap trampoline patch: tensor_load_to_lds multicast fix.
// COM: Two variants: dead SGPR (no save/restore) and live SGPR (with
// COM: v_writelane/v_readlane save/restore).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: s_pack_hh should appear (clears multicast routing bits)
// DISASM-DAG: s_pack_hh_b32_b16

// COM: tensor_load_to_lds should still be present (in trampoline)
// DISASM-DAG: tensor_load_to_lds

// COM: Dead-SGPR kernel should NOT have save/restore for s4
// COM: (checked via the live kernel having it -- we just verify presence)

// COM: Live-SGPR kernel should have save/restore
// DISASM-DAG: v_writelane_b32
// DISASM-DAG: v_readlane_b32

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

// ---- Kernel 1: tensor_load_to_lds with dead SGPR (s_endpgm follows) --------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_dead
.p2align 8
.type test_tensor_dead,@function
test_tensor_dead:
  tensor_load_to_lds s[0:3], s[4:11]
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
.Ltest_tensor_dead_end:
.size test_tensor_dead, .Ltest_tensor_dead_end-test_tensor_dead

// ---- Kernel 2: tensor_load_to_lds with live SGPR (s4 used after) -----------

.globl test_tensor_live
.p2align 8
.type test_tensor_live,@function
test_tensor_live:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
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
.Ltest_tensor_live_end:
.size test_tensor_live, .Ltest_tensor_live_end-test_tensor_live

.rodata
.p2align 8
.amdhsa_kernel test_tensor_dead
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_live
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
