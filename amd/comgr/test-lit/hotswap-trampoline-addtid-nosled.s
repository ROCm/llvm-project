// COM: Test the trampoline fallback path for ds_*_addtid_b32 when no NOP
// COM: sled is available. With zero NOP padding inside the kernel,
// COM: emitReplacementCode falls back to emitToTrampoline: the original
// COM: ADDTID is rewritten to s_branch and the 5-instruction expansion
// COM: (lane-id math + ds_load_b32) is appended after .text via
// COM: growWithTrampolines. Companion to hotswap-trampoline-addtid.s
// COM: which exercises the in-place NOP-sled path on the same opcode.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The original ADDTID is gone; an s_branch forward replaces it. The
// COM: surrounding s_wait_dscnt and s_endpgm are untouched.
// DISASM-LABEL: <test_addtid_nosled>:
// DISASM-NOT: ds_load_addtid_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: s_endpgm

// COM: Trampoline body appended after .text: lane-id math then ds_load_b32
// COM: with the original offset folded in, then s_branch back to the
// COM: instruction following the original ADDTID site.
// DISASM: v_mbcnt_lo_u32_b32
// DISASM-NEXT: v_mbcnt_hi_u32_b32
// DISASM: v_lshlrev_b32
// DISASM-NEXT: v_add_nc_u32
// DISASM: ds_load_b32
// DISASM: s_branch

// COM: Idempotency: rewriting the patched output a second time must
// COM: produce identical bytes. The trampoline body uses plain ds_load_b32
// COM: (no ADDTID mnemonic), so the dispatcher leaves it untouched on
// COM: subsequent runs.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_addtid_nosled
.p2align 8
.type test_addtid_nosled,@function
test_addtid_nosled:
  ds_load_addtid_b32 v5 offset:128
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_addtid_nosled_end:
.size test_addtid_nosled, .Ltest_addtid_nosled_end-test_addtid_nosled

.rodata
.p2align 8
.amdhsa_kernel test_addtid_nosled
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
