// COM: A contiguous B32 DS2 store whose first data VGPR is odd cannot fold to
// COM: a single-address B64 store on gfx1250 A0. Keep it on the normal split
// COM: path, including when an adjacent DS2 site forces deferred trampolines.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf > %t.log 2>&1
// RUN: %FileCheck --check-prefix=API \
// RUN:   --implicit-check-not="invalid operand for instruction" \
// RUN:   --implicit-check-not="parser produced no instructions" %s < %t.log
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_odd_aligned_contiguous_store>:
// DISASM-NOT:  ds_store_2addr
// DISASM:      s_branch
// DISASM:      s_branch
// DISASM:      ds_store_b32 v27, v19 offset:20
// DISASM-NEXT: ds_store_b32 v27, v20 offset:24
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM:      ds_store_b64 v27, v[0:1] offset:8
// DISASM-NEXT: ds_store_b64 v27, v[36:37] offset:24
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NOT:  ds_store_b64 v27, v[19:20]

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_odd_aligned_contiguous_store
.p2align 8
.type test_odd_aligned_contiguous_store,@function
test_odd_aligned_contiguous_store:
  ds_store_2addr_b32 v27, v19, v20 offset0:5 offset1:6
  ds_store_2addr_b64 v27, v[0:1], v[36:37] offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_odd_aligned_contiguous_store_end:
.size test_odd_aligned_contiguous_store, .Ltest_odd_aligned_contiguous_store_end-test_odd_aligned_contiguous_store

.rodata
.p2align 8
.amdhsa_kernel test_odd_aligned_contiguous_store
  .amdhsa_next_free_vgpr 38
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
