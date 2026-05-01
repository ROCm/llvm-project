// COM: Test HotSwap trampoline patch: ds_*_2addr_stride64_* expansion
// COM: into two single-address DS instructions with s_wait_dscnt bump.
// COM: Covers b32 load, b64 load, b32 store, and multi-DS stacking paths
// COM: via the NOP sled emission mechanism. Verifies explicit s_branch
// COM: generation for the forward/back jumps.
// COM: See hotswap-trampoline-ds-nosled.s for the true trampoline fallback.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: --- Per-kernel checks ---

// COM: Kernel 1 (b32 load): s_branch forward to sled, bumped wait stays
// COM: at original position, expanded loads appear in sled area with
// COM: s_branch back to the wait instruction.
// DISASM-LABEL: <test_ds_load_b32>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_branch

// COM: Kernel 2 (b64 load): b64 register pairs formatted as v[X:Y]
// DISASM-LABEL: <test_ds_load_b64>:
// DISASM-NOT: ds_load_2addr_stride64_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_load_b64 v[0:1]
// DISASM: ds_load_b64 v[2:3]
// DISASM: s_branch

// COM: Kernel 3 (b32 store): store operand layout (addr, data0, data1)
// DISASM-LABEL: <test_ds_store_b32>:
// DISASM-NOT: ds_store_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_store_b32 v2, v0
// DISASM: ds_store_b32 v2, v1
// DISASM: s_branch

// COM: Kernel 4 (multi-DS stacking): two DS2 sites before one wait => 0x2
// DISASM-LABEL: <test_multi_ds>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x2

// COM: Idempotency: rewriting the output again should produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// ---- Kernel 1: ds_load_2addr_stride64_b32 (base case) -----------------------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_load_b32
.p2align 8
.type test_ds_load_b32,@function
test_ds_load_b32:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
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
.Ltest_ds_load_b32_end:
.size test_ds_load_b32, .Ltest_ds_load_b32_end-test_ds_load_b32

// ---- Kernel 2: ds_load_2addr_stride64_b64 (b64 element size) ----------------

.globl test_ds_load_b64
.p2align 8
.type test_ds_load_b64,@function
test_ds_load_b64:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
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
.Ltest_ds_load_b64_end:
.size test_ds_load_b64, .Ltest_ds_load_b64_end-test_ds_load_b64

// ---- Kernel 3: ds_store_2addr_stride64_b32 (store operand layout) -----------

.globl test_ds_store_b32
.p2align 8
.type test_ds_store_b32,@function
test_ds_store_b32:
  ds_store_2addr_stride64_b32 v2, v0, v1 offset0:1 offset1:3
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
.Ltest_ds_store_b32_end:
.size test_ds_store_b32, .Ltest_ds_store_b32_end-test_ds_store_b32

// ---- Kernel 4: multi-DS stacking (two DS2 sites, one wait) ------------------

.globl test_multi_ds
.p2align 8
.type test_multi_ds,@function
test_multi_ds:
  ds_load_2addr_stride64_b32 v[0:1], v4 offset0:0 offset1:1
  ds_load_2addr_stride64_b32 v[2:3], v4 offset0:2 offset1:3
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
.Ltest_multi_ds_end:
.size test_multi_ds, .Ltest_multi_ds_end-test_multi_ds

.rodata
.p2align 8
.amdhsa_kernel test_ds_load_b32
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b64
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_store_b32
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_multi_ds
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
