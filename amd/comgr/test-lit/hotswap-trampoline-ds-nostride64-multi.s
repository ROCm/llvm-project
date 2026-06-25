// COM: Test multi-DS stacking for the non-stride64 DS 2-address family:
// COM: two ds_load_2addr_b32 sites before a single s_wait_dscnt 0x0
// COM: (drain). Each split is expanded into two single-address loads
// COM: followed by its own s_wait_dscnt 0x0 drain, and the pre-existing
// COM: downstream drain is left in place -- so both split halves always
// COM: complete before any consumer.
// COM:
// COM: Companion test:
// COM:   hotswap-trampoline-ds-nostride64.s -- non-stride64 base case
// COM:     (load b32, load b64, store b32, exchange b32, store b64,
// COM:     exchange b64, and a load+store+xchg combination kernel) drain
// COM:     forms.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// ---- Kernel: 2x ds_load_2addr_b32 + drain s_wait_dscnt 0x0 ----------------
// COM: Both DS2 instructions are replaced by s_branch to their respective
// COM: expansion sleds. The single shared s_wait_dscnt stays at 0x0
// COM: (drain preservation under stacking in the non-stride64 path).
// COM: The two sleds together expand to 4 single-address ds_load_b32
// COM: instructions; offsets (raw 1/2 and 3/4) scale by ElemBytes=4 to
// COM: byte offsets 4/8 and 12/16 across the two sites.
// DISASM-LABEL: <test_multi_ds_nostride64>:
// DISASM-NOT: ds_load_2addr_b32
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// COM: Sled 1 (first DS2 site, vdst pair v[0:1]).
// DISASM: ds_load_b32 v0, v4 offset:4
// DISASM-NEXT: ds_load_b32 v1, v4 offset:8
// COM: Sled 2 (second DS2 site, vdst pair v[2:3]).
// DISASM: ds_load_b32 v2, v4 offset:12
// DISASM-NEXT: ds_load_b32 v3, v4 offset:16

.globl test_multi_ds_nostride64
.p2align 8
.type test_multi_ds_nostride64,@function
test_multi_ds_nostride64:
  ds_load_2addr_b32 v[0:1], v4 offset0:1 offset1:2
  ds_load_2addr_b32 v[2:3], v4 offset0:3 offset1:4
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
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_multi_ds_nostride64_end:
.size test_multi_ds_nostride64, .Ltest_multi_ds_nostride64_end-test_multi_ds_nostride64

// COM: Idempotency: rewriting the output again should produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.rodata
.p2align 8
.amdhsa_kernel test_multi_ds_nostride64
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
