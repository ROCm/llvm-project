// Test HotSwap WMMA-split modifier and FP-inline-immediate preservation.
//
// The asm syntax forms exercised here are the splittable subset of
// llvm/test/MC/AMDGPU/gfx1250_asm_wmma_w32.s (the upstream source of
// truth for what the AMDGPU asm parser accepts on these opcodes):
//
//   - src2 = FP inline constant `1.0` (encodes via inline-const slot
//     242, distinct from integer `1`'s slot 1; the splitter must
//     preserve `1.0` verbatim through the printer rather than reformat
//     as `itostr(getImm())` which would mis-encode).
//   - `neg_lo:[0,0,1]` (negate src2 packed-modifier bit). On a K-split
//     this bit must appear on the FIRST half (which holds the original
//     src2) and be CLEARED on the SECOND half (whose src2 slot is the
//     dst register, the accumulator carry -- negating dst would
//     subtract the partial result, which is wrong).
//   - `neg_hi:[0,0,1]` (same shape as neg_lo, separate bit).
//   - `matrix_a_reuse` / `matrix_b_reuse` (HW data-reuse hints). The
//     splitter strips these on output: preserving them would assert
//     the reuse buffer holds valid data after the rewrite, which is
//     not true (the data lives in different VGPR slices in each half).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// -- Source kernels: each test function exercises one variant ---------------
//
// All trampolines land at the tail of .text after the last function body;
// the DISASM-LABEL / DISASM-NOT / DISASM blocks per function only assert
// that the original opcode is gone and an s_branch took its place. The
// trailing DISASM-DAG block (at the bottom of this file) is where
// trampoline emission is verified.

// DISASM-LABEL: <test_k_fp_imm>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_k_fp_imm
.p2align 8
.type test_k_fp_imm,@function
test_k_fp_imm:
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], 1.0
  s_endpgm
.size test_k_fp_imm, .-test_k_fp_imm

// DISASM-LABEL: <test_k_neg_lo>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_k_neg_lo
.p2align 8
.type test_k_neg_lo,@function
test_k_neg_lo:
  v_wmma_f32_16x16x128_bf8_bf8 v[24:31], v[0:15], v[8:23], v[24:31] neg_lo:[0,0,1]
  s_endpgm
.size test_k_neg_lo, .-test_k_neg_lo

// DISASM-LABEL: <test_k_neg_hi>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_k_neg_hi
.p2align 8
.type test_k_neg_hi,@function
test_k_neg_hi:
  v_wmma_f16_16x16x128_fp8_bf8 v[20:23], v[0:15], v[8:23], v[20:23] neg_hi:[0,0,1]
  s_endpgm
.size test_k_neg_hi, .-test_k_neg_hi

// DISASM-LABEL: <test_k_matrix_a_reuse>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_k_matrix_a_reuse
.p2align 8
.type test_k_matrix_a_reuse,@function
test_k_matrix_a_reuse:
  v_wmma_f32_16x16x128_fp8_bf8 v[40:47], v[0:15], v[8:23], v[40:47] matrix_a_reuse
  s_endpgm
.size test_k_matrix_a_reuse, .-test_k_matrix_a_reuse

// DISASM-LABEL: <test_k_matrix_b_reuse>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_fp8
// DISASM:       s_branch
.globl test_k_matrix_b_reuse
.p2align 8
.type test_k_matrix_b_reuse,@function
test_k_matrix_b_reuse:
  v_wmma_f32_16x16x128_bf8_fp8 v[48:55], v[0:15], v[8:23], v[48:55] matrix_b_reuse
  s_endpgm
.size test_k_matrix_b_reuse, .-test_k_matrix_b_reuse

// DISASM-LABEL: <test_m_fp_imm>:
// DISASM-NOT:   v_wmma_f32_32x16x128_f4
// DISASM:       s_branch
.globl test_m_fp_imm
.p2align 8
.type test_m_fp_imm,@function
test_m_fp_imm:
  v_wmma_f32_32x16x128_f4 v[64:79], v[0:15], v[2:9], 1.0
  s_endpgm
.size test_m_fp_imm, .-test_m_fp_imm

// DISASM-LABEL: <test_m_neg_lo>:
// DISASM-NOT:   v_wmma_f32_32x16x128_f4
// DISASM:       s_branch
.globl test_m_neg_lo
.p2align 8
.type test_m_neg_lo,@function
test_m_neg_lo:
  v_wmma_f32_32x16x128_f4 v[80:95], v[0:15], v[2:9], v[80:95] neg_lo:[0,0,1]
  s_endpgm
.size test_m_neg_lo, .-test_m_neg_lo

// -- Trampoline region: assertions on the rewritten output ------------------
//
// All DAG matches operate on the .text tail (the trampolines appended
// by the rewriter); DAG order is unconstrained.

// COM: K-split with FP imm src2: first-half preserves `1.0`, second-half
// COM: src2 becomes the dst register `v[16:23]` (the accumulator carry).
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], 1.0
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[8:15], v[16:23], v[16:23]

// COM: K-split with neg_lo:[0,0,1] on src2: first half preserves the
// COM: modifier; second half drops it (src2 bit cleared, [0,0,0] is the
// COM: printer's omitted-default form so the second half has no
// COM: neg_lo suffix at all).
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_bf8 v[24:31], v[0:7], v[8:15], v[24:31] neg_lo:[0,0,1]
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_bf8 v[24:31], v[8:15], v[16:23], v[24:31]{{[[:space:]]*\/\/}}

// COM: Same shape for neg_hi:[0,0,1].
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_bf8 v[20:23], v[0:7], v[8:15], v[20:23] neg_hi:[0,0,1]
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_bf8 v[20:23], v[8:15], v[16:23], v[20:23]{{[[:space:]]*\/\/}}

// COM: matrix_a_reuse stripped on both halves.
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_bf8 v[40:47], v[0:7], v[8:15], v[40:47]{{[[:space:]]*\/\/}}
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_bf8 v[40:47], v[8:15], v[16:23], v[40:47]{{[[:space:]]*\/\/}}
// DISASM-NOT: v_wmma_f32_16x16x64_fp8_bf8 v[40:47]{{.*}}matrix_a_reuse

// COM: matrix_b_reuse stripped on both halves.
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_fp8 v[48:55], v[0:7], v[8:15], v[48:55]{{[[:space:]]*\/\/}}
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_fp8 v[48:55], v[8:15], v[16:23], v[48:55]{{[[:space:]]*\/\/}}
// DISASM-NOT: v_wmma_f32_16x16x64_bf8_fp8 v[48:55]{{.*}}matrix_b_reuse

// COM: M-split with FP imm: both halves carry 1.0 plus the splitter-added
// COM: MATRIX_FMT_FP4 modifiers (required by the f8f6f4 destination opcode
// COM: to interpret the data as f4).
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[64:71], v[0:7], v[2:9], 1.0{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[72:79], v[8:15], v[2:9], 1.0{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4

// COM: M-split with neg_lo on src2: both halves preserve the modifier
// COM: (no carry on the M axis).
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[80:87], v[0:7], v[2:9], v[80:87]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4{{.*}}neg_lo:[0,0,1]
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[88:95], v[8:15], v[2:9], v[88:95]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4{{.*}}neg_lo:[0,0,1]

// -- Idempotency across all variants ----------------------------------------
//
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
