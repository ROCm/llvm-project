// A direct call can enter the WMMA either with the function-entry mode or the
// fallthrough can reach it after selecting a different destination bank. The
// K-split sources do not cross v255, but the second half replaces immediate
// src2 with dst and therefore still needs the ambiguous mode. Fail closed.
// This also exercises s_call_i64's unusual target operand layout in the shared
// direct-control-flow target index.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=CHECK %s
// CHECK-LABEL: WMMA split: cannot determine VGPR-MSB mode
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_vgpr_msb_merge
.p2align 8
.type test_wmma_vgpr_msb_merge,@function
test_wmma_vgpr_msb_merge:
  s_call_i64 s[0:1], .Lwmma
  s_set_vgpr_msb 0x40
.Lwmma:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.Ltest_wmma_vgpr_msb_merge_end:
.size test_wmma_vgpr_msb_merge, .Ltest_wmma_vgpr_msb_merge_end-test_wmma_vgpr_msb_merge

.rodata
.p2align 8
.amdhsa_kernel test_wmma_vgpr_msb_merge
  .amdhsa_next_free_vgpr 266
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
