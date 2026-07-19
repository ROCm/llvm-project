// COM: A far patch whose scratch pair must be globally allocated fails closed
// COM: when the code object has no metadata note to charge for that allocation.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK: hotswap: error:
// CHECK-SAME: metadata
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far_no_metadata
.p2align 8
.type test_far_no_metadata,@function
test_far_no_metadata:
  s_cbranch_scc1 .Lresume
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lresume:
  s_cmp_lg_u64 s[62:63], 0
  s_cmp_lg_u64 s[60:61], 0
  s_cmp_lg_u64 s[58:59], 0
  s_cmp_lg_u64 s[56:57], 0
  s_cmp_lg_u64 s[46:47], 0
  s_cmp_lg_u64 s[44:45], 0
  s_endpgm
.Ltest_far_no_metadata_end:
.size test_far_no_metadata, .Ltest_far_no_metadata_end-test_far_no_metadata

// Safe gateway space outside the function, followed by non-NOP far filler.
.rept 8
  s_nop 0
.endr
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_far_no_metadata
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
