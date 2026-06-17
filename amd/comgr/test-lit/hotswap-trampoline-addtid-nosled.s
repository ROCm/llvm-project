// COM: Test the trampoline fallback path for ds_*_addtid_b32 when no NOP
// COM: sled is available. With zero NOP padding inside the kernel,
// COM: emitReplacementCode falls back to emitToTrampoline: the original
// COM: ADDTID is rewritten to s_branch and the 6-instruction expansion
// COM: (lane-id math + 20-bit M0 mask + ds_load_b32) is appended after
// COM: .text via growWithTrampolines. Companion to hotswap-trampoline-
// COM: addtid.s which exercises the in-place NOP-sled path on the same
// COM: opcode.
// COM:
// COM: DISASM convention: the kernel-local sequence (s_branch, structural
// COM: s_nop pad, s_wait_dscnt, s_endpgm) is checked with a strict
// COM: DISASM-NEXT chain. The s_nop is structural: ds_load_addtid_b32 is
// COM: 8 bytes, s_branch is 4 bytes, so emitToTrampoline always pads the
// COM: tail half of the original instruction slot with one s_nop -- pinning
// COM: it here catches any change to that padding scheme. The trampoline
// COM: body lives in a separate region appended by growWithTrampolines, so
// COM: the second block uses a non-consecutive 'DISASM:' on v_mbcnt_lo to
// COM: skip over the kernel terminator and any padding the assembler emits
// COM: between regions, then DISASM-NEXT chains every body instruction so
// COM: regressions in the math sequence, the 20-bit mask, or the operand
// COM: order of ds_load_b32 are caught.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// ---- Kernel: ds_load_addtid_b32 with no sled --------------------------------
//
// COM: Inside the kernel function the original ADDTID is gone; an s_branch
// COM: forward replaces it. The surrounding s_wait_dscnt and s_endpgm are
// COM: untouched.

// DISASM-LABEL: <test_addtid_nosled>:
// DISASM-NOT:   ds_load_addtid_b32
// DISASM:       s_branch
// DISASM-NEXT:  s_nop 0
// DISASM-NEXT:  s_wait_dscnt 0x0
// DISASM-NEXT:  s_endpgm

// COM: Trampoline body appended after .text: lane-id math, 20-bit M0 mask
// COM: (matches B0's DS-unit M0 read width and is a no-op for any
// COM: conforming M0 value), ds_load_b32 with the original offset:128
// COM: folded into the DS encoding, then s_branch back to the instruction
// COM: following the original ADDTID site. All operand-pinned so that an
// COM: offset/operand/shift/scalar-source regression is caught here.

// DISASM:       v_mbcnt_lo_u32_b32 v5, -1, 0
// DISASM-NEXT:  v_mbcnt_hi_u32_b32 v5, -1, v5
// DISASM-NEXT:  v_lshlrev_b32_e32 v5, 2, v5
// DISASM-NEXT:  v_add_nc_u32_e32 v5, m0, v5
// DISASM-NEXT:  v_and_b32_e32 v5, 0xfffff, v5
// DISASM-NEXT:  ds_load_b32 v5, v5 offset:128
// DISASM-NEXT:  s_branch

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

// COM: Idempotency: rewriting the patched output a second time must
// COM: produce identical bytes. The trampoline body uses plain ds_load_b32
// COM: (no ADDTID mnemonic), so the dispatcher leaves it untouched on
// COM: subsequent runs.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES
