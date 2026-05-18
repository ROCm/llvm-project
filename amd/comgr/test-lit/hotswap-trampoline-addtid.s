// COM: Test HotSwap trampoline patch: ds_*_addtid_b32 expansion.
// COM:
// COM: On A0 the DS unit truncates M0 to 16 bits, so ADDTID address
// COM: encodings (M0 + lane_id*4 + offset) silently wrap above 64KB
// COM: (DEGFXMI400-12025). The trampoline materialises the lane-id math
// COM: in the ALU using the full 32-bit M0 and issues a regular
// COM: ds_load_b32 / ds_store_b32, bypassing the buggy address path.
// COM:
// COM: Coverage:
// COM:   test_addtid_load        : ds_load_addtid_b32 + offset (NOP sled)
// COM:   test_addtid_load_zero   : ds_load_addtid_b32 + offset:0
// COM:   test_addtid_store       : ds_store_addtid_b32 needs a scratch VGPR

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1: ds_load_addtid_b32 with non-zero offset.
// COM:   Original site is replaced with s_branch (forward to NOP sled);
// COM:   the sled body computes lane_id*4 + m0 in vDST, then reads LDS
// COM:   with the original offset folded into the DS encoding; finally
// COM:   s_branch returns to the next instruction past the original.
// DISASM-LABEL: <test_addtid_load>:
// DISASM-NOT: ds_load_addtid_b32
// DISASM: s_branch
// DISASM: v_mbcnt_lo_u32_b32
// DISASM-NEXT: v_mbcnt_hi_u32_b32
// DISASM: v_lshlrev_b32
// DISASM-NEXT: v_add_nc_u32
// DISASM: ds_load_b32
// DISASM: s_branch

// COM: Kernel 2: ds_load_addtid_b32 with offset:0 (offset suffix omitted).
// DISASM-LABEL: <test_addtid_load_zero>:
// DISASM-NOT: ds_load_addtid_b32
// DISASM: s_branch
// DISASM: v_mbcnt_lo_u32_b32
// DISASM: ds_load_b32
// DISASM: s_branch

// COM: Kernel 3: ds_store_addtid_b32. Scratch VGPR holds the computed
// COM: address; original data VGPR is preserved as the store source.
// DISASM-LABEL: <test_addtid_store>:
// DISASM-NOT: ds_store_addtid_b32
// DISASM: s_branch
// DISASM: v_mbcnt_lo_u32_b32
// DISASM: v_add_nc_u32
// DISASM: ds_store_b32
// DISASM: s_branch

// COM: Idempotency: rewriting the output a second time must produce
// COM: identical bytes (the patched body has no ADDTID mnemonic so the
// COM: dispatcher leaves it untouched on subsequent runs).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// ---- Kernel 1: ds_load_addtid_b32 with offset --------------------------------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_addtid_load
.p2align 8
.type test_addtid_load,@function
test_addtid_load:
  ds_load_addtid_b32 v5 offset:128
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
.Ltest_addtid_load_end:
.size test_addtid_load, .Ltest_addtid_load_end-test_addtid_load

// ---- Kernel 2: ds_load_addtid_b32 with offset:0 ------------------------------

.globl test_addtid_load_zero
.p2align 8
.type test_addtid_load_zero,@function
test_addtid_load_zero:
  ds_load_addtid_b32 v6 offset:0
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
.Ltest_addtid_load_zero_end:
.size test_addtid_load_zero, .Ltest_addtid_load_zero_end-test_addtid_load_zero

// ---- Kernel 3: ds_store_addtid_b32 ------------------------------------------

.globl test_addtid_store
.p2align 8
.type test_addtid_store,@function
test_addtid_store:
  ds_store_addtid_b32 v8 offset:64
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
.Ltest_addtid_store_end:
.size test_addtid_store, .Ltest_addtid_store_end-test_addtid_store

.rodata
.p2align 8
.amdhsa_kernel test_addtid_load
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_addtid_load_zero
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_addtid_store
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
