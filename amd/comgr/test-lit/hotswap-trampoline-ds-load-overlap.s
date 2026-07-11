// COM: Test the register-overlap reorder in expandDs2AddrLoad: when a
// COM: ds_load_2addr_* address register aliases the destination window, the
// COM: split must emit the half that carries the address LAST. A single DS
// COM: load reads its address before writing its destination, so the
// COM: address-carrying half is safe as the final use; the other (disjoint)
// COM: half, emitted first, never clobbers the address.
// COM:
// COM: Motivating real case (gfx1250 rocThrust, hotswap OFF disasm):
// COM:   ds_load_2addr_b64 v[2:5], v2 offset1:1
// COM: address v2 aliases the low destination pair v[2:3]; the naive low-first
// COM: split clobbers v2 before the high load reads it (intermittent torn read).
// COM:
// COM: Covers the whole destination window for b64 (address = v2,v3 in the low
// COM: pair / v4,v5 in the high pair) and b32 (address = v2 low / v3 high).
// COM: Only loads are affected (only loads write the destination); the
// COM: store/exchange split paths are unchanged and covered by the other
// COM: hotswap-trampoline-ds*.s tests, which also cover the no-overlap order.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// ---- Kernel 1: b64, address in LOW half (v2) -- the real rocThrust case -----
// COM: offset0 defaults to 0 (omitted in disasm); offset1:1 -> byte 8. v2
// COM: aliases the low pair v[2:3] -> reorder: high v[4:5] first, low v[2:3]
// COM: last so v2 stays live until the low load reads-then-overwrites it.
// DISASM-LABEL: <test_ds_load_b64_addr_low_v2>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b64 v[4:5], v2 offset:8
// DISASM-NEXT: ds_load_b64 v[2:3], v2
// DISASM: s_branch
.globl test_ds_load_b64_addr_low_v2
.p2align 8
.type test_ds_load_b64_addr_low_v2,@function
test_ds_load_b64_addr_low_v2:
  ds_load_2addr_b64 v[2:5], v2 offset1:1
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
.Ltest_ds_load_b64_addr_low_v2_end:
.size test_ds_load_b64_addr_low_v2, .Ltest_ds_load_b64_addr_low_v2_end-test_ds_load_b64_addr_low_v2

// ---- Kernel 2: b64, address in LOW half (v3, upper dword of the low pair) ---
// COM: offset0:1 offset1:2 -> bytes 8 and 16. v3 is still in low pair v[2:3],
// COM: so this is a low-half overlap -> reorder (high v[4:5] first).
// DISASM-LABEL: <test_ds_load_b64_addr_low_v3>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b64 v[4:5], v3 offset:16
// DISASM-NEXT: ds_load_b64 v[2:3], v3 offset:8
// DISASM: s_branch
.globl test_ds_load_b64_addr_low_v3
.p2align 8
.type test_ds_load_b64_addr_low_v3,@function
test_ds_load_b64_addr_low_v3:
  ds_load_2addr_b64 v[2:5], v3 offset0:1 offset1:2
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
.Ltest_ds_load_b64_addr_low_v3_end:
.size test_ds_load_b64_addr_low_v3, .Ltest_ds_load_b64_addr_low_v3_end-test_ds_load_b64_addr_low_v3

// ---- Kernel 3: b64, address in HIGH half (v4) -- no reorder (default) -------
// COM: offset0:1 offset1:2 -> bytes 8 and 16. The low load v[2:3] is disjoint
// COM: from v4, so the default low-first order is already safe; the high load
// COM: (which overwrites v4) is the address's last use.
// DISASM-LABEL: <test_ds_load_b64_addr_high_v4>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b64 v[2:3], v4 offset:8
// DISASM-NEXT: ds_load_b64 v[4:5], v4 offset:16
// DISASM: s_branch
.globl test_ds_load_b64_addr_high_v4
.p2align 8
.type test_ds_load_b64_addr_high_v4,@function
test_ds_load_b64_addr_high_v4:
  ds_load_2addr_b64 v[2:5], v4 offset0:1 offset1:2
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
.Ltest_ds_load_b64_addr_high_v4_end:
.size test_ds_load_b64_addr_high_v4, .Ltest_ds_load_b64_addr_high_v4_end-test_ds_load_b64_addr_high_v4

// ---- Kernel 4: b64, address in HIGH half (v5) -- no reorder -----------------
// COM: v5 is the upper dword of the high pair v[4:5]; still a high-half
// COM: overlap, so default low-first order is retained.
// DISASM-LABEL: <test_ds_load_b64_addr_high_v5>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b64 v[2:3], v5 offset:8
// DISASM-NEXT: ds_load_b64 v[4:5], v5 offset:16
// DISASM: s_branch
.globl test_ds_load_b64_addr_high_v5
.p2align 8
.type test_ds_load_b64_addr_high_v5,@function
test_ds_load_b64_addr_high_v5:
  ds_load_2addr_b64 v[2:5], v5 offset0:1 offset1:2
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
.Ltest_ds_load_b64_addr_high_v5_end:
.size test_ds_load_b64_addr_high_v5, .Ltest_ds_load_b64_addr_high_v5_end-test_ds_load_b64_addr_high_v5

// ---- Kernel 5: b32, address in LOW half (v2) -- reorder ---------------------
// COM: b32 split: low half = v2, high half = v3 (Half == 1). offset0:1
// COM: offset1:2 -> bytes 4 and 8. v2 in low -> reorder: high v3 first.
// DISASM-LABEL: <test_ds_load_b32_addr_low_v2>:
// DISASM-NOT: ds_load_2addr_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b32 v3, v2 offset:8
// DISASM-NEXT: ds_load_b32 v2, v2 offset:4
// DISASM: s_branch
.globl test_ds_load_b32_addr_low_v2
.p2align 8
.type test_ds_load_b32_addr_low_v2,@function
test_ds_load_b32_addr_low_v2:
  ds_load_2addr_b32 v[2:3], v2 offset0:1 offset1:2
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
.Ltest_ds_load_b32_addr_low_v2_end:
.size test_ds_load_b32_addr_low_v2, .Ltest_ds_load_b32_addr_low_v2_end-test_ds_load_b32_addr_low_v2

// ---- Kernel 6: b32, address in HIGH half (v3) -- no reorder -----------------
// COM: v3 is the high half -> default low-first order (low v2 first).
// DISASM-LABEL: <test_ds_load_b32_addr_high_v3>:
// DISASM-NOT: ds_load_2addr_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b32 v2, v3 offset:4
// DISASM-NEXT: ds_load_b32 v3, v3 offset:8
// DISASM: s_branch
.globl test_ds_load_b32_addr_high_v3
.p2align 8
.type test_ds_load_b32_addr_high_v3,@function
test_ds_load_b32_addr_high_v3:
  ds_load_2addr_b32 v[2:3], v3 offset0:1 offset1:2
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
.Ltest_ds_load_b32_addr_high_v3_end:
.size test_ds_load_b32_addr_high_v3, .Ltest_ds_load_b32_addr_high_v3_end-test_ds_load_b32_addr_high_v3

// COM: Idempotency: rewriting the output again is a no-op (no DS2 mnemonic
// COM: remains, second pass produces identical bytes).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.rodata
.p2align 8
.amdhsa_kernel test_ds_load_b64_addr_low_v2
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b64_addr_low_v3
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b64_addr_high_v4
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b64_addr_high_v5
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b32_addr_low_v2
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b32_addr_high_v3
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
