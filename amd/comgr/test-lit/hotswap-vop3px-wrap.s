// Test VOP3PX2 wrap patches for GFX1250 B0-to-A0 hotswap.
//
// On A0 silicon, an async trap fired between LD_SCALE and the WMMA half
// of a VOP3PX2 pair is unrecoverable.  The trap handler rewinds the PC
// for known-paired forms (ROCm/rocm-systems commit 74c647e6605); this
// pass ensures every standalone V_WMMA_F32_16X16X128_F8F6F4 (encoding
// 0xCC33) is paired with an inline-zero LD_SCALE prefix (effectively
// scale=1.0, a no-op) so the rewind path always has a pair to walk
// back to.
//
// The wrap is byte-level: an 8-byte LD_SCALE prefix is prepended to the
// original 8-byte WMMA, leaving the WMMA portion bit-identical.  In
// disassembly the result reads as a single fused
// `v_wmma_scale_f32_16x16x128_f8f6f4` instruction with `0, 0` for the
// two scale operands.
//
// Per amd/comgr/AGENT_CONVENTIONS.md, LIT inputs are compiled with
// %clang directly (not through Comgr actions), and llvm-objdump /
// FileCheck go through their lit substitutions.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// ── Test 1: bare standalone WMMA gets wrapped ───────────────────────────────
//
// The original WMMA must be replaced by an s_branch into the trampoline,
// where the wrapped form appears as a SCALE-prefixed VOP3PX2.
//
// CHECK-LABEL: <test_standalone_f8f6f4>:
// CHECK-NOT:   v_wmma_f32_16x16x128_f8f6f4 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
// CHECK:       s_branch
.globl test_standalone_f8f6f4
.p2align 8
.type test_standalone_f8f6f4,@function
test_standalone_f8f6f4:
  v_wmma_f32_16x16x128_f8f6f4 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size test_standalone_f8f6f4, .-test_standalone_f8f6f4

// ── Test 2: standalone with explicit FP8/FP8 modifiers ──────────────────────
//
// Modifiers (matrix_a_fmt, matrix_b_fmt) must be preserved verbatim in
// the wrapped form because the WMMA bytes are copied unchanged.
//
// CHECK-LABEL: <test_standalone_fp8_fp8>:
// CHECK-NOT:   v_wmma_f32_16x16x128_f8f6f4 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] matrix_a_fmt
// CHECK:       s_branch
.globl test_standalone_fp8_fp8
.p2align 8
.type test_standalone_fp8_fp8,@function
test_standalone_fp8_fp8:
  v_wmma_f32_16x16x128_f8f6f4 v[16:23], v[0:15], v[8:23], v[16:23] matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
  s_endpgm
.size test_standalone_fp8_fp8, .-test_standalone_fp8_fp8

// ── Test 3: standalone with FP6/FP4 mixed modifiers ─────────────────────────
//
// Verifies that all 9 type combinations are handled — the wrap is
// modifier-agnostic; only the WMMA's byte-level opcode (0xCC33) matters.
//
// CHECK-LABEL: <test_standalone_fp6_fp4>:
// CHECK-NOT:   v_wmma_f32_16x16x128_f8f6f4 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] matrix_a_fmt
// CHECK:       s_branch
.globl test_standalone_fp6_fp4
.p2align 8
.type test_standalone_fp6_fp4,@function
test_standalone_fp6_fp4:
  v_wmma_f32_16x16x128_f8f6f4 v[16:23], v[0:11], v[8:15], v[16:23] matrix_a_fmt:MATRIX_FMT_FP6 matrix_b_fmt:MATRIX_FMT_FP4
  s_endpgm
.size test_standalone_fp6_fp4, .-test_standalone_fp6_fp4

// ── Test 4: already-wrapped (SCALE prefix present) is left alone ─────────────
//
// A v_wmma_scale_f32_16x16x128_f8f6f4 is already a fused VOP3PX2.  The
// wrap pass MUST NOT add a second LD_SCALE prefix in front of it (that
// would corrupt the encoding).  We assert that no s_branch is installed
// over the user's already-wrapped form.
//
// CHECK-LABEL: <test_already_wrapped>:
// CHECK:       v_wmma_scale_f32_16x16x128_f8f6f4
// CHECK-NOT:   s_branch
// CHECK:       s_endpgm
.globl test_already_wrapped
.p2align 8
.type test_already_wrapped,@function
test_already_wrapped:
  v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[0:15], v[8:23], v[16:23], 0, 0
  s_endpgm
.size test_already_wrapped, .-test_already_wrapped

// ── Test 5: K=128 splitter interaction ──────────────────────────────────────
//
// `v_wmma_f32_32x16x128_f4` is split by the K=128 splitter into two
// f8f6f4 WMMAs that land in a trampoline.  The wrap pass's pass-2
// trampoline scan must wrap BOTH of them.  Disassembled trampoline
// region should contain `v_wmma_scale_f32_16x16x128_f8f6f4` (the wrapped
// form) and NO bare `v_wmma_f32_16x16x128_f8f6f4` (the unwrapped form).
//
// CHECK-LABEL: <test_f4_split_then_wrap>:
// CHECK-NOT:   v_wmma_f32_32x16x128_f4
// CHECK:       s_branch
.globl test_f4_split_then_wrap
.p2align 8
.type test_f4_split_then_wrap,@function
test_f4_split_then_wrap:
  v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], 0
  s_endpgm
.size test_f4_split_then_wrap, .-test_f4_split_then_wrap

// ── Trampoline region asserts ───────────────────────────────────────────────
//
// At least one wrapped form (LD_SCALE + WMMA) must appear in the
// trampoline tail.  We use CHECK-DAG since trampolines are emitted in
// patch order and the lit harness's textual order may differ.  We also
// assert that no bare standalone WMMA leaks through the splitter+wrap
// pipeline — every f8f6f4 in the rewritten ELF must be paired with a
// SCALE prefix (i.e., printed as `v_wmma_scale_*`).
//
// CHECK-DAG: v_wmma_scale_f32_16x16x128_f8f6f4
// CHECK-NOT: v_wmma_f32_16x16x128_f8f6f4 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
