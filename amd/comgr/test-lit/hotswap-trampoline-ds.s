// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s

// API: RESULT: SUCCESS
// DISASM-LABEL: <test_ds_load_b32>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_branch
// DISASM-LABEL: <test_ds_load_b64>:
// DISASM-NOT: ds_load_2addr_stride64_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_load_b64 v[0:1]
// DISASM: ds_load_b64 v[2:3]
// DISASM: s_branch
// DISASM-LABEL: <test_ds_store_b32>:
// DISASM-NOT: ds_store_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_store_b32 v2, v0
// DISASM: ds_store_b32 v2, v1
// DISASM: s_branch
// DISASM-LABEL: <test_ds_xchg_b32>:
// DISASM-NOT: ds_storexchg_2addr_stride64_rtn_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x1
// DISASM: ds_storexchg_rtn_b32 v0
// DISASM: ds_storexchg_rtn_b32 v1
// DISASM: s_branch
// IDEM: IDEMPOTENT: YES

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

// ---- Kernel 4: ds_storexchg_2addr_stride64_rtn_b32 (exchange layout) --------

.globl test_ds_xchg_b32
.p2align 8
.type test_ds_xchg_b32,@function
test_ds_xchg_b32:
  ds_storexchg_2addr_stride64_rtn_b32 v[0:1], v2, v3, v4 offset0:1 offset1:3
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
.Ltest_ds_xchg_b32_end:
.size test_ds_xchg_b32, .Ltest_ds_xchg_b32_end-test_ds_xchg_b32

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

.amdhsa_kernel test_ds_xchg_b32
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
