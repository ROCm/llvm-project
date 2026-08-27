// A merge that disagrees only in src1 must lose the complete VGPR-MSB mode.
// This guards the four-field CFG lattice independently of the dst/src0 facts
// used by DS2 alignment.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --expect-status ERROR 2>&1 | %FileCheck %s
// CHECK-LABEL: WMMA split: cannot determine VGPR-MSB mode
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_src1_merge
.p2align 8
.type test_wmma_src1_merge,@function
test_wmma_src1_merge:
  s_cbranch_scc1 .Lzero
  s_set_vgpr_msb 4
  s_branch .Ljoin
.Lzero:
  s_set_vgpr_msb 0
.Ljoin:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_wmma_src1_merge, .-test_wmma_src1_merge
