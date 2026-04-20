// Test WMMA split when $src2 (the accumulator input) is an inline immediate
// instead of a VGPR range.
//
// VOP3P WMMA's $src2 accepts VSrc, so when the C operand is statically zero
// clang folds it to inline immediate 0 — the canonical case for kernels that
// use the WMMA intrinsics with `acc = {0,…,0}`. The splitter must accept
// this 3-VGPR-plus-immediate form: it propagates the immediate into the
// first split's $src2 and uses the dst register as the carry into the
// second split. Without this, the splitter bails (`could not extract 4 VGPR
// operands`), the rewrite is refused, and the kernel keeps the original
// K=128 opcode that doesn't exist on A0 silicon.
//
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj -o %t.o %s
// RUN: hotswap-rewrite %t.o amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 -o %t.patched.o
// RUN: llvm-objdump -d --mcpu=gfx1250 %t.patched.o | FileCheck %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// ── Test 1: 16x16x128_fp8_fp8 with $src2 = inline imm 0 ─────────────────────
//
// CHECK-LABEL: <test_f32_16x16x128_fp8_fp8_imm0>:
// CHECK-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// CHECK:       s_branch
.globl test_f32_16x16x128_fp8_fp8_imm0
.p2align 8
.type test_f32_16x16x128_fp8_fp8_imm0,@function
test_f32_16x16x128_fp8_fp8_imm0:
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], 0
  s_endpgm
.size test_f32_16x16x128_fp8_fp8_imm0, .-test_f32_16x16x128_fp8_fp8_imm0

// ── Test 2: 16x16x128_bf8_bf8 (f16 dest) with $src2 = inline imm 0 ──────────
//
// CHECK-LABEL: <test_f16_16x16x128_bf8_bf8_imm0>:
// CHECK-NOT:   v_wmma_f16_16x16x128_bf8_bf8
// CHECK:       s_branch
.globl test_f16_16x16x128_bf8_bf8_imm0
.p2align 8
.type test_f16_16x16x128_bf8_bf8_imm0,@function
test_f16_16x16x128_bf8_bf8_imm0:
  v_wmma_f16_16x16x128_bf8_bf8 v[16:19], v[0:15], v[8:23], 0
  s_endpgm
.size test_f16_16x16x128_bf8_bf8_imm0, .-test_f16_16x16x128_bf8_bf8_imm0

// ── Test 3: 32x16x128_f4 (FP4 M-split) with $src2 = inline imm 0 ────────────
//
// CHECK-LABEL: <test_f32_32x16x128_f4_imm0>:
// CHECK-NOT:   v_wmma_f32_32x16x128_f4
// CHECK:       s_branch
.globl test_f32_32x16x128_f4_imm0
.p2align 8
.type test_f32_32x16x128_f4_imm0,@function
test_f32_32x16x128_f4_imm0:
  v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], 0
  s_endpgm
.size test_f32_32x16x128_f4_imm0, .-test_f32_32x16x128_f4_imm0

// ── Test 4: mixed-format 16x16x128_fp8_bf8 with $src2 = inline imm 0 ────────
//
// CHECK-LABEL: <test_f32_16x16x128_fp8_bf8_imm0>:
// CHECK-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// CHECK:       s_branch
.globl test_f32_16x16x128_fp8_bf8_imm0
.p2align 8
.type test_f32_16x16x128_fp8_bf8_imm0,@function
test_f32_16x16x128_fp8_bf8_imm0:
  v_wmma_f32_16x16x128_fp8_bf8 v[16:23], v[0:15], v[8:23], 0
  s_endpgm
.size test_f32_16x16x128_fp8_bf8_imm0, .-test_f32_16x16x128_fp8_bf8_imm0

// ── Trampoline region ───────────────────────────────────────────────────────
//
// Each rewritten instruction lands in a trampoline at the tail of .text.
// The K-split form emits:
//   <repl> dst, A_lo, B_lo, <c>      ; first split — $src2 = original imm
//   <repl> dst, A_hi, B_hi, dst      ; second split — $src2 = dst (carry)
// The M-split (FP4) form emits two halves of dst, both with $src2 = the
// original imm (no carry between them on the M axis).
//
// Each CHECK-DAG below asserts both the replacement mnemonic AND that the
// assembler emitted at least one form with the inline immediate 0 as $src2
// — the trailing literal `, 0` is what we fed in. For K-splits the second
// half uses dst as $src2 (the accumulator carry), so each pattern matches
// exactly the first half of its split. For the M-split (FP4) both halves
// use the same imm, but one match per mnemonic is enough to prove the
// splitter propagated the imm into the rewritten code.
//
// CHECK-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
// CHECK-DAG: v_wmma_f16_16x16x64_bf8_bf8 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
// CHECK-DAG: v_wmma_f32_16x16x128_f8f6f4 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
// CHECK-DAG: v_wmma_f32_16x16x64_fp8_bf8 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
