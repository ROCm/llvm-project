// Test WMMA split patches for GFX1250 B0-to-A0 hotswap.
//
// The hotswap rewriter replaces every splittable WMMA with an s_branch
// into a trampoline at the tail of .text that contains two narrower WMMAs
// followed by an s_branch back.  This test disassembles the patched ELF
// and checks that:
//   - the original mnemonics no longer appear (overwritten with s_branch)
//   - the narrower replacement mnemonics appear in the trampoline region
//   - non-split instructions round-trip unchanged.
//
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj -o %t.o %s
// RUN: hotswap-rewrite %t.o amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 -o %t.patched.o
// RUN: llvm-objdump -d --mcpu=gfx1250 %t.patched.o | FileCheck %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// ── Test 1: 16x16x128_fp8_fp8 → two 16x16x64_fp8_fp8 ─────────────────────────
//
// CHECK-LABEL: <test_f32_16x16x128_fp8_fp8>:
// CHECK-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// CHECK:       s_branch
.globl test_f32_16x16x128_fp8_fp8
.p2align 8
.type test_f32_16x16x128_fp8_fp8,@function
test_f32_16x16x128_fp8_fp8:
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size test_f32_16x16x128_fp8_fp8, .-test_f32_16x16x128_fp8_fp8

// ── Test 2: 16x16x128_bf8_bf8 (f16 dest, 4-wide) → two 16x16x64_bf8_bf8 ──────
//
// CHECK-LABEL: <test_f16_16x16x128_bf8_bf8>:
// CHECK-NOT:   v_wmma_f16_16x16x128_bf8_bf8
// CHECK:       s_branch
.globl test_f16_16x16x128_bf8_bf8
.p2align 8
.type test_f16_16x16x128_bf8_bf8,@function
test_f16_16x16x128_bf8_bf8:
  v_wmma_f16_16x16x128_bf8_bf8 v[16:19], v[0:15], v[8:23], v[16:19]
  s_endpgm
.size test_f16_16x16x128_bf8_bf8, .-test_f16_16x16x128_bf8_bf8

// ── Test 3: 32x16x128_f4 → two 16x16x128_f8f6f4 ─────────────────────────────
//
// CHECK-LABEL: <test_f32_32x16x128_f4>:
// CHECK-NOT:   v_wmma_f32_32x16x128_f4
// CHECK:       s_branch
.globl test_f32_32x16x128_f4
.p2align 8
.type test_f32_32x16x128_f4,@function
test_f32_32x16x128_f4:
  v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19]
  s_endpgm
.size test_f32_32x16x128_f4, .-test_f32_32x16x128_f4

// ── Test 4: mixed-format 16x16x128_fp8_bf8 → two 16x16x64_fp8_bf8 ───────────
//
// CHECK-LABEL: <test_f32_16x16x128_fp8_bf8>:
// CHECK-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// CHECK:       s_branch
.globl test_f32_16x16x128_fp8_bf8
.p2align 8
.type test_f32_16x16x128_fp8_bf8,@function
test_f32_16x16x128_fp8_bf8:
  v_wmma_f32_16x16x128_fp8_bf8 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size test_f32_16x16x128_fp8_bf8, .-test_f32_16x16x128_fp8_bf8

// ── Test 5: non-splittable instructions round-trip unchanged ────────────────
//
// CHECK-LABEL: <test_no_split_required>:
// CHECK:       v_wmma_f32_16x16x32_f16
// CHECK:       v_add_f32
.globl test_no_split_required
.p2align 8
.type test_no_split_required,@function
test_no_split_required:
  v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
  v_add_f32_e32 v0, v1, v2
  s_endpgm
.size test_no_split_required, .-test_no_split_required

// ── Trampoline region: the splits land after the last original function.
//    The grown .text has no distinct symbol for the trampolines, so the
//    disassembly lists them under the <test_no_split_required> label
//    (anchored above).  Assert each replacement mnemonic appears within
//    that region; CHECK-DAG lets the emission order change without
//    breaking the test.
//
// CHECK-DAG: v_wmma_f32_16x16x64_fp8_fp8
// CHECK-DAG: v_wmma_f16_16x16x64_bf8_bf8
// CHECK-DAG: v_wmma_f32_16x16x128_f8f6f4
// CHECK-DAG: v_wmma_f32_16x16x64_fp8_bf8
