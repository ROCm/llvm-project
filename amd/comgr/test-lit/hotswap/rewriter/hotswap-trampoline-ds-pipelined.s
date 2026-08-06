// COM: Test HotSwap trampoline patch: each ds_*_2addr_* split is followed by
// COM: a local s_wait_dscnt 0x0 drain so both split halves complete before any
// COM: downstream consumer. An existing non-drain wait (s_wait_dscnt 0x1) is
// COM: left untouched -- the split's own drain provides the ordering, so the
// COM: downstream wait must NOT be relaxed (relaxing it would let a consumer
// COM: read a not-yet-landed half; see the patchDs2Addr rationale).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1 (single split): one DS2 -> two single-address loads followed
// COM: by an s_wait_dscnt 0x0 drain; the downstream s_wait_dscnt 0x1 is left
// COM: unchanged.
// DISASM-LABEL: <test_ds_pipelined_single>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_wait_dscnt 0x0
// DISASM: s_branch

// COM: Kernel 2 (two splits): each split lands its own s_wait_dscnt 0x0 drain;
// COM: the shared downstream s_wait_dscnt 0x1 is left unchanged.
// DISASM-LABEL: <test_ds_pipelined_multi>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b32 v2
// DISASM: ds_load_b32 v3
// DISASM: s_wait_dscnt 0x0

// COM: Idempotency: a second rewrite produces identical bytes (the 2-addr
// COM: forms are gone, so there is nothing left to split).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// ---- Kernel 1: single split + drain; downstream 0x1 wait preserved ----------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_pipelined_single
.p2align 8
.type test_ds_pipelined_single,@function
test_ds_pipelined_single:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x1
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
.Ltest_ds_pipelined_single_end:
.size test_ds_pipelined_single, .Ltest_ds_pipelined_single_end-test_ds_pipelined_single

// ---- Kernel 2: two splits, each + drain; downstream 0x1 wait preserved ------

.globl test_ds_pipelined_multi
.p2align 8
.type test_ds_pipelined_multi,@function
test_ds_pipelined_multi:
  ds_load_2addr_stride64_b32 v[0:1], v4 offset0:0 offset1:1
  ds_load_2addr_stride64_b32 v[2:3], v4 offset0:2 offset1:3
  s_wait_dscnt 0x1
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
.Ltest_ds_pipelined_multi_end:
.size test_ds_pipelined_multi, .Ltest_ds_pipelined_multi_end-test_ds_pipelined_multi

.rodata
.p2align 8
.amdhsa_kernel test_ds_pipelined_single
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_pipelined_multi
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
