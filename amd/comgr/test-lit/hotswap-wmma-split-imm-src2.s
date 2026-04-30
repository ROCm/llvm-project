// Test HotSwap WMMA-split when $src2 (accumulator) is an inline immediate.
//
// VOP3P WMMA's $src2 accepts VSrc, so when the C operand is statically zero
// clang folds it to inline immediate 0 -- the canonical case for kernels
// that use the WMMA intrinsics with `acc = {0,...,0}`. The splitter must
// accept this 3-VGPR-plus-immediate form: it propagates the immediate into
// the first split's $src2 and uses the dst register as the carry into the
// second split. Without this, the splitter bails (`could not extract 4 VGPR
// operands`), the rewrite is refused, and the kernel keeps the original
// K=128 opcode that does not exist on A0 silicon.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Verify .text grew on the wire (see hotswap-wmma-split.s for rationale).
// RUN: SIZE_IN=$(%llvm-readelf -S %t.elf | awk '/\.text /{print $7; exit}') && \
// RUN:   SIZE_OUT=$(%llvm-readelf -S %t.out.elf | awk '/\.text /{print $7; exit}') && \
// RUN:   test $((16#$SIZE_OUT)) -gt $((16#$SIZE_IN))

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// -- Test 1: 16x16x128_fp8_fp8 with $src2 = inline imm 0 ---------------------
//
// DISASM-LABEL: <test_f32_16x16x128_fp8_fp8_imm0>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_fp8_imm0
.p2align 8
.type test_f32_16x16x128_fp8_fp8_imm0,@function
test_f32_16x16x128_fp8_fp8_imm0:
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], 0
  s_endpgm
.size test_f32_16x16x128_fp8_fp8_imm0, .-test_f32_16x16x128_fp8_fp8_imm0

// -- Test 2: 16x16x128_bf8_bf8 (f16 dest) with $src2 = inline imm 0 ----------
//
// DISASM-LABEL: <test_f16_16x16x128_bf8_bf8_imm0>:
// DISASM-NOT:   v_wmma_f16_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_f16_16x16x128_bf8_bf8_imm0
.p2align 8
.type test_f16_16x16x128_bf8_bf8_imm0,@function
test_f16_16x16x128_bf8_bf8_imm0:
  v_wmma_f16_16x16x128_bf8_bf8 v[16:19], v[0:15], v[8:23], 0
  s_endpgm
.size test_f16_16x16x128_bf8_bf8_imm0, .-test_f16_16x16x128_bf8_bf8_imm0

// -- Test 3: 32x16x128_f4 (FP4 M-split) with $src2 = inline imm 0 ------------
//
// DISASM-LABEL: <test_f32_32x16x128_f4_imm0>:
// DISASM-NOT:   v_wmma_f32_32x16x128_f4
// DISASM:       s_branch
.globl test_f32_32x16x128_f4_imm0
.p2align 8
.type test_f32_32x16x128_f4_imm0,@function
test_f32_32x16x128_f4_imm0:
  v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], 0
  s_endpgm
.size test_f32_32x16x128_f4_imm0, .-test_f32_32x16x128_f4_imm0

// -- Test 4: mixed-format 16x16x128_fp8_bf8 with $src2 = inline imm 0 --------
//
// DISASM-LABEL: <test_f32_16x16x128_fp8_bf8_imm0>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_bf8_imm0
.p2align 8
.type test_f32_16x16x128_fp8_bf8_imm0,@function
test_f32_16x16x128_fp8_bf8_imm0:
  v_wmma_f32_16x16x128_fp8_bf8 v[16:23], v[0:15], v[8:23], 0
  s_endpgm
.size test_f32_16x16x128_fp8_bf8_imm0, .-test_f32_16x16x128_fp8_bf8_imm0

// -- Trampoline region -------------------------------------------------------
//
// Each rewritten instruction lands in a trampoline at the tail of .text.
// The K-split form emits:
//   <repl> dst, A_lo, B_lo, <c>      ; first split, $src2 = original imm
//   <repl> dst, A_hi, B_hi, dst      ; second split, $src2 = dst (carry)
// The M-split (FP4) form emits two halves of dst, both with $src2 = the
// original imm (no carry between them on the M axis).
//
// Each CHECK-DAG below asserts both the replacement mnemonic AND that the
// assembler emitted at least one form with the inline immediate 0 as $src2
// -- the trailing literal `, 0` is what we fed in. For K-splits the second
// half uses dst as $src2 (the accumulator carry), so each pattern matches
// exactly the first half of its split. For the M-split (FP4) both halves
// use the same imm, but one match per mnemonic is enough to prove the
// splitter propagated the imm into the rewritten code.
//
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
// DISASM-DAG: v_wmma_f16_16x16x64_bf8_bf8 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_bf8 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0

// Idempotency: rewriting the patched output should produce identical bytes.
//
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
