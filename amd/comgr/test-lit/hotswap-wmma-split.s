// Test HotSwap WMMA-split patches for GFX1250 B0-to-A0.
//
// The splitter replaces every K=128 fp8/bf8 WMMA with an s_branch into a
// trampoline at the tail of .text containing two K=64 halves followed by a
// branch back. The 32x16x128_f4 variant becomes two 16x16x128_f8f6f4 halves
// with both matrix-format modifiers forced to MATRIX_FMT_FP4. This test
// disassembles the patched ELF and checks that the original mnemonics are
// gone, the narrower replacement mnemonics appear in the trampoline region,
// and non-split instructions round-trip unchanged.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf --allow-grow \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// -- Test 1: 16x16x128_fp8_fp8 -> two 16x16x64_fp8_fp8 -----------------------
//
// DISASM-LABEL: <test_f32_16x16x128_fp8_fp8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_fp8
.p2align 8
.type test_f32_16x16x128_fp8_fp8,@function
test_f32_16x16x128_fp8_fp8:
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size test_f32_16x16x128_fp8_fp8, .-test_f32_16x16x128_fp8_fp8

// -- Test 2: 16x16x128_bf8_bf8 (f16 dest, 4-wide) -> two 16x16x64_bf8_bf8 ----
//
// DISASM-LABEL: <test_f16_16x16x128_bf8_bf8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_f16_16x16x128_bf8_bf8
.p2align 8
.type test_f16_16x16x128_bf8_bf8,@function
test_f16_16x16x128_bf8_bf8:
  v_wmma_f16_16x16x128_bf8_bf8 v[16:19], v[0:15], v[8:23], v[16:19]
  s_endpgm
.size test_f16_16x16x128_bf8_bf8, .-test_f16_16x16x128_bf8_bf8

// -- Test 3: 32x16x128_f4 -> two 16x16x128_f8f6f4 ----------------------------
//
// DISASM-LABEL: <test_f32_32x16x128_f4>:
// DISASM-NOT:   v_wmma_f32_32x16x128_f4
// DISASM:       s_branch
.globl test_f32_32x16x128_f4
.p2align 8
.type test_f32_32x16x128_f4,@function
test_f32_32x16x128_f4:
  v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19]
  s_endpgm
.size test_f32_32x16x128_f4, .-test_f32_32x16x128_f4

// -- Test 4: mixed-format 16x16x128_fp8_bf8 -> two 16x16x64_fp8_bf8 ----------
//
// DISASM-LABEL: <test_f32_16x16x128_fp8_bf8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_bf8
.p2align 8
.type test_f32_16x16x128_fp8_bf8,@function
test_f32_16x16x128_fp8_bf8:
  v_wmma_f32_16x16x128_fp8_bf8 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size test_f32_16x16x128_fp8_bf8, .-test_f32_16x16x128_fp8_bf8

// -- Test 5: 16x16x128_bf8_fp8 (f32) -> two 16x16x64_bf8_fp8 -----------------
//
// DISASM-LABEL: <test_f32_16x16x128_bf8_fp8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_fp8
// DISASM:       s_branch
.globl test_f32_16x16x128_bf8_fp8
.p2align 8
.type test_f32_16x16x128_bf8_fp8,@function
test_f32_16x16x128_bf8_fp8:
  v_wmma_f32_16x16x128_bf8_fp8 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size test_f32_16x16x128_bf8_fp8, .-test_f32_16x16x128_bf8_fp8

// -- Test 6: 16x16x128_bf8_bf8 (f32) -> two 16x16x64_bf8_bf8 -----------------
//
// DISASM-LABEL: <test_f32_16x16x128_bf8_bf8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_f32_16x16x128_bf8_bf8
.p2align 8
.type test_f32_16x16x128_bf8_bf8,@function
test_f32_16x16x128_bf8_bf8:
  v_wmma_f32_16x16x128_bf8_bf8 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size test_f32_16x16x128_bf8_bf8, .-test_f32_16x16x128_bf8_bf8

// -- Test 7: 16x16x128_fp8_fp8 (f16 dest) -> two 16x16x64_fp8_fp8 ------------
//
// DISASM-LABEL: <test_f16_16x16x128_fp8_fp8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_f16_16x16x128_fp8_fp8
.p2align 8
.type test_f16_16x16x128_fp8_fp8,@function
test_f16_16x16x128_fp8_fp8:
  v_wmma_f16_16x16x128_fp8_fp8 v[16:19], v[0:15], v[8:23], v[16:19]
  s_endpgm
.size test_f16_16x16x128_fp8_fp8, .-test_f16_16x16x128_fp8_fp8

// -- Test 8: 16x16x128_fp8_bf8 (f16 dest) -> two 16x16x64_fp8_bf8 ------------
//
// DISASM-LABEL: <test_f16_16x16x128_fp8_bf8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_f16_16x16x128_fp8_bf8
.p2align 8
.type test_f16_16x16x128_fp8_bf8,@function
test_f16_16x16x128_fp8_bf8:
  v_wmma_f16_16x16x128_fp8_bf8 v[16:19], v[0:15], v[8:23], v[16:19]
  s_endpgm
.size test_f16_16x16x128_fp8_bf8, .-test_f16_16x16x128_fp8_bf8

// -- Test 9: 16x16x128_bf8_fp8 (f16 dest) -> two 16x16x64_bf8_fp8 ------------
//
// DISASM-LABEL: <test_f16_16x16x128_bf8_fp8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_bf8_fp8
// DISASM:       s_branch
.globl test_f16_16x16x128_bf8_fp8
.p2align 8
.type test_f16_16x16x128_bf8_fp8,@function
test_f16_16x16x128_bf8_fp8:
  v_wmma_f16_16x16x128_bf8_fp8 v[16:19], v[0:15], v[8:23], v[16:19]
  s_endpgm
.size test_f16_16x16x128_bf8_fp8, .-test_f16_16x16x128_bf8_fp8

// -- Test 10: non-splittable instructions round-trip unchanged ---------------
//
// DISASM-LABEL: <test_no_split_required>:
// DISASM:       v_wmma_f32_16x16x32_f16
// DISASM:       v_add_f32
.globl test_no_split_required
.p2align 8
.type test_no_split_required,@function
test_no_split_required:
  v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
  v_add_f32_e32 v0, v1, v2
  s_endpgm
.size test_no_split_required, .-test_no_split_required

// -- Trampoline region: the splits land after the last original function. The
//    grown .text has no distinct symbol for the trampolines, so the
//    disassembly lists them under the <test_no_split_required> label
//    (anchored above). Assert each replacement mnemonic appears within that
//    region; CHECK-DAG lets the emission order change without breaking the
//    test. Eight K=64 fp8/bf8 replacement mnemonics (4 sign combinations x
//    {f16,f32} dest) plus the f4-split's f8f6f4 product cover the full
//    splitter table.
//
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_bf8
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_fp8
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_bf8
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_fp8
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_bf8
// DISASM-DAG: v_wmma_f16_16x16x64_bf8_fp8
// DISASM-DAG: v_wmma_f16_16x16x64_bf8_bf8
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4

// Idempotency: rewriting the patched output again should produce identical
// bytes (the splitter only fires on K=128 mnemonics, which no longer exist
// in the rewritten ELF).
//
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf --allow-grow \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
