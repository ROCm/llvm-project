// COM: Test the DS2 split drain when there is no downstream wait: a DS2
// COM: instruction followed directly by s_endpgm with no s_wait_dscnt in the
// COM: same basic block. The split still emits its own s_wait_dscnt 0x0 drain
// COM: after the two single-address ops, so both halves complete even though
// COM: the original kernel had no wait before s_endpgm.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The DS2 is expanded (replaced by s_branch to sled); the sled holds the
// COM: two single-address loads followed by the split's own s_wait_dscnt 0x0
// COM: drain. No downstream wait existed, so the only wait is the split drain.
// DISASM-LABEL: <test_ds_nowait>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_wait_dscnt 0x0
// DISASM: s_branch

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_nowait
.p2align 8
.type test_ds_nowait,@function
test_ds_nowait:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
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
.Ltest_ds_nowait_end:
.size test_ds_nowait, .Ltest_ds_nowait_end-test_ds_nowait

.rodata
.p2align 8
.amdhsa_kernel test_ds_nowait
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
