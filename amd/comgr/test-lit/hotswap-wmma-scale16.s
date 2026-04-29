// COM: Test WMMA Scale16 (block-16 -> block-32) decomposition patch.
// COM:
// COM: Creates minimal gfx1250 code objects containing v_wmma_scale16_f32_*
// COM: instructions, runs the hotswap rewrite, and verifies the replacement
// COM: sequence covers: byte-pair max scale reduction followed by the
// COM: rewritten v_wmma_scale_f32_* instruction (VOP3PX2).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: -----------------------------------------------------------------------
// COM: Verify: Scale16 16x16 instruction is patched with scale reduction
// COM: preamble followed by the rewritten v_wmma_scale_f32_* instruction.
// COM:
// COM: Scale reduction per operand (13 instructions):
// COM:   4 byte pairs x (v_and_b32/v_bfe_u32 + v_bfe_u32/v_lshrrev_b32 +
// COM:                    v_max_u32 + optional v_lshl_or_b32)
// COM:
// COM: The preamble reduces block-16 scales from the B64 VGPR pairs
// COM: (v48:v49 for scale A, v50:v51 for scale B) into B32 scratch VGPRs
// COM: via byte-pair max, then the rewritten WMMA uses those scratch VGPRs.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SCALE16 %s

// SCALE16-LABEL: <test_wmma_scale16_16x16>:
// SCALE16:       s_branch
// COM: --- scale A reduction: byte pair 0 ---
// SCALE16:       v_and_b32{{.*}}0xff, v48
// SCALE16-NEXT:  v_bfe_u32{{.*}}v48, 8, 8
// SCALE16-NEXT:  v_max_u32
// COM: --- scale A reduction: byte pair 1 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v48, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v48
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale A reduction: byte pair 2 ---
// SCALE16-NEXT:  v_and_b32{{.*}}0xff, v49
// SCALE16-NEXT:  v_bfe_u32{{.*}}v49, 8, 8
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale A reduction: byte pair 3 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v49, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v49
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale B reduction: byte pair 0 ---
// SCALE16-NEXT:  v_and_b32{{.*}}0xff, v50
// SCALE16-NEXT:  v_bfe_u32{{.*}}v50, 8, 8
// SCALE16-NEXT:  v_max_u32
// COM: --- scale B reduction: byte pair 1 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v50, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v50
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale B reduction: byte pair 2 ---
// SCALE16-NEXT:  v_and_b32{{.*}}0xff, v51
// SCALE16-NEXT:  v_bfe_u32{{.*}}v51, 8, 8
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale B reduction: byte pair 3 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v51, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v51
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- rewritten WMMA (VOP3PX2): regular scale with scratch VGPRs ---
// SCALE16-NEXT:  v_wmma_scale_f32_16x16x128_f8f6f4

// COM: -----------------------------------------------------------------------
// COM: Verify: regular Scale instruction is NOT patched.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOPATCH %s

// NOPATCH-LABEL: <test_wmma_scale_16x16>:
// NOPATCH-NEXT:  v_wmma_scale_f32_16x16x128_f8f6f4

// COM: -----------------------------------------------------------------------
// COM: Idempotency: running the rewrite a second time should succeed and
// COM: produce the same output.
// COM: -----------------------------------------------------------------------
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=IDEMP %s
// IDEMP: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// --- Kernel 1: Scale16 (should be patched) ---
.globl test_wmma_scale16_16x16
.p2align 8
.type test_wmma_scale16_16x16,@function
test_wmma_scale16_16x16:
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[16:31], v[32:47], v[0:7], v[48:49], v[50:51]
  s_endpgm
.Ltest_wmma_scale16_16x16_end:
.size test_wmma_scale16_16x16, .Ltest_wmma_scale16_16x16_end-test_wmma_scale16_16x16

// --- Kernel 2: regular Scale (should NOT be patched) ---
.globl test_wmma_scale_16x16
.p2align 8
.type test_wmma_scale_16x16,@function
test_wmma_scale_16x16:
  v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[16:31], v[32:47], v[0:7], v48, v50
  s_endpgm
.Ltest_wmma_scale_16x16_end:
.size test_wmma_scale_16x16, .Ltest_wmma_scale_16x16_end-test_wmma_scale_16x16

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_16x16
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_wmma_scale_16x16
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
