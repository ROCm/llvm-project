// Test sethalt-fix patches for GFX1250 A0 LD_SCALE+WMMA clause-break
// triggered by in-shader s_sethalt (FPV ticket: sethalt before
// LD_SCALE breaks the implicit clause).
//
// The hotswap sethalt-fix pass detects every `s_sethalt` instruction
// in the input ELF and replaces it with `s_nop 0` in-place.  The
// shader proceeds without the halt; if the halt was for an in-shader
// debug breakpoint, that's the correct trade-off on A0 (debugger
// should switch to trap-based breakpoints, which lives outside the
// code object).
//
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// ── Test 1: bare s_sethalt is neutralized to s_nop ─────────────────────────
//
// The single s_sethalt in this kernel must disappear; an s_nop must
// appear in its place.  No other instructions are affected.
//
// CHECK-LABEL: <test_bare_sethalt>:
// CHECK-NOT:   s_sethalt
// CHECK:       s_nop
// CHECK:       s_endpgm
.globl test_bare_sethalt
.p2align 8
.type test_bare_sethalt,@function
test_bare_sethalt:
  s_sethalt 1
  s_endpgm
.size test_bare_sethalt, .-test_bare_sethalt

// ── Test 2: s_sethalt before VOP3PX2 (the FPV scenario) ────────────────────
//
// This is the exact FPV-ticket repro shape: s_sethalt halts the wave
// just before a v_wmma_scale_* (VOP3PX2) instruction would issue.
// On A0, the halt would break the implicit LD_SCALE+WMMA clause and
// allow scale-factor leakage.  After hotswap, the s_sethalt is gone
// and the VOP3PX2 runs without the halt-induced race.
//
// CHECK-LABEL: <test_sethalt_before_vop3px2>:
// CHECK-NOT:   s_sethalt
// CHECK:       s_nop
// CHECK:       v_wmma_scale_f32_16x16x128_f8f6f4
// CHECK:       s_endpgm
.globl test_sethalt_before_vop3px2
.p2align 8
.type test_sethalt_before_vop3px2,@function
test_sethalt_before_vop3px2:
  s_sethalt 1
  v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[0:15], v[8:23], v[16:23], 0, 0
  s_endpgm
.size test_sethalt_before_vop3px2, .-test_sethalt_before_vop3px2

// ── Test 3: multiple s_sethalt instances all neutralized ───────────────────
//
// The pass walks decoded[] linearly, so all s_sethalt occurrences
// should be neutralized regardless of position or simm operand.
//
// CHECK-LABEL: <test_multiple_sethalt>:
// CHECK-NOT:   s_sethalt
// CHECK:       s_nop
// CHECK:       v_add_f32
// CHECK:       s_nop
// CHECK:       s_endpgm
.globl test_multiple_sethalt
.p2align 8
.type test_multiple_sethalt,@function
test_multiple_sethalt:
  s_sethalt 0
  v_add_f32_e32 v0, v1, v2
  s_sethalt 0x42
  s_endpgm
.size test_multiple_sethalt, .-test_multiple_sethalt

// ── Test 4: kernel without s_sethalt is not modified ───────────────────────
//
// Kernels that don't contain s_sethalt must round-trip unchanged.
// In particular, no s_nop should be inserted where there wasn't an
// s_sethalt to replace.
//
// CHECK-LABEL: <test_no_sethalt>:
// CHECK-NOT:   s_nop
// CHECK:       v_add_f32
// CHECK:       s_endpgm
.globl test_no_sethalt
.p2align 8
.type test_no_sethalt,@function
test_no_sethalt:
  v_add_f32_e32 v0, v1, v2
  s_endpgm
.size test_no_sethalt, .-test_no_sethalt
