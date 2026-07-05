// COM: Test v_cvt_pk_fp8_f32 CLAMP=1 (E5M3) full conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_pk_fp8_f32
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: NaN detection, base F32->F16->UE5M3
// COM: conversion, RTE rounding, overflow clamping, and NaN override.
// COM:
// COM: Companion tests:
// COM:   hotswap-cvt-fp8-modifiers.s — source modifier variants
// COM:   hotswap-cvt-fp8-nosled.s    — trampoline fallback path
// COM:   hotswap-cvt-fp8-multi.s     — multi-site stacking

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=LOW %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=HIGH %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOCLAMP %s

// ---- Kernel 1: CLAMP=1, low half (should be patched) --------------------------
//
// COM: Original site is replaced with s_branch. Trampoline body: VCC save,
// COM: two per-source F32→UE5M3 conversions (15 instructions each), pack
// COM: into 16-bit pair, merge into low half of vdst via v_bfi_b32, VCC restore.

// LOW-LABEL: <test_cvt_pk_fp8_low>:
// LOW:       s_branch
// COM: --- VCC save ---
// LOW:       s_mov_b32
// COM: --- src0 conversion ---
// LOW-NEXT:  v_and_b32{{.*}}0x7fffffff, v1
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_max_num_f32{{.*}}, 0, v1
// LOW-NEXT:  v_cvt_f16_f32
// LOW-NEXT:  v_and_b32
// LOW-NEXT:  v_lshrrev_b32
// LOW-NEXT:  v_lshlrev_b32
// LOW-NEXT:  v_bfi_b32
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x80
// LOW-NEXT:  v_add_co_ci_u32
// LOW-NEXT:  v_min_u32
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_mov_b32
// LOW-NEXT:  v_cndmask_b32
// COM: --- src1 conversion ---
// LOW-NEXT:  v_and_b32{{.*}}0x7fffffff, v2
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_max_num_f32{{.*}}, 0, v2
// LOW-NEXT:  v_cvt_f16_f32
// LOW-NEXT:  v_and_b32
// LOW-NEXT:  v_lshrrev_b32
// LOW-NEXT:  v_lshlrev_b32
// LOW-NEXT:  v_bfi_b32
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x80
// LOW-NEXT:  v_add_co_ci_u32
// LOW-NEXT:  v_min_u32
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_mov_b32
// LOW-NEXT:  v_cndmask_b32
// COM: --- pack + merge (low half) ---
// LOW-NEXT:  v_lshl_or_b32
// LOW-NEXT:  v_bfi_b32 v0,
// COM: --- VCC restore ---
// LOW-NEXT:  s_mov_b32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cvt_pk_fp8_low
.p2align 8
.type test_cvt_pk_fp8_low,@function
test_cvt_pk_fp8_low:
  v_cvt_pk_fp8_f32 v0, v1, v2 clamp
  s_endpgm
.Ltest_cvt_pk_fp8_low_end:
.size test_cvt_pk_fp8_low, .Ltest_cvt_pk_fp8_low_end-test_cvt_pk_fp8_low

// ---- Kernel 2: CLAMP=1, high half (raw encoding for op_sel[3]=1) --------------
//
// COM: Same conversion sequence as low, but final merge uses shift + bfi to
// COM: write the packed bytes into the upper 16 bits of vdst.

// HIGH-LABEL: <test_cvt_pk_fp8_high>:
// HIGH:       s_branch
// COM: --- VCC save + src0 conversion (anchor on unique src v6) ---
// HIGH:       v_and_b32{{.*}}0x7fffffff, v6
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_max_num_f32{{.*}}, 0, v6
// HIGH-NEXT:  v_cvt_f16_f32
// HIGH-NEXT:  v_and_b32
// HIGH-NEXT:  v_lshrrev_b32
// HIGH-NEXT:  v_lshlrev_b32
// HIGH-NEXT:  v_bfi_b32
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x80
// HIGH-NEXT:  v_add_co_ci_u32
// HIGH-NEXT:  v_min_u32
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_mov_b32
// HIGH-NEXT:  v_cndmask_b32
// COM: --- src1 conversion ---
// HIGH-NEXT:  v_and_b32{{.*}}0x7fffffff, v7
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_max_num_f32{{.*}}, 0, v7
// HIGH-NEXT:  v_cvt_f16_f32
// HIGH-NEXT:  v_and_b32
// HIGH-NEXT:  v_lshrrev_b32
// HIGH-NEXT:  v_lshlrev_b32
// HIGH-NEXT:  v_bfi_b32
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x80
// HIGH-NEXT:  v_add_co_ci_u32
// HIGH-NEXT:  v_min_u32
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_mov_b32
// HIGH-NEXT:  v_cndmask_b32
// COM: --- pack + merge (high half: shift + bfi) ---
// HIGH-NEXT:  v_lshl_or_b32
// HIGH-NEXT:  v_lshlrev_b32
// HIGH-NEXT:  v_bfi_b32 v5,
// COM: --- VCC restore ---
// HIGH-NEXT:  s_mov_b32

.globl test_cvt_pk_fp8_high
.p2align 8
.type test_cvt_pk_fp8_high,@function
test_cvt_pk_fp8_high:
  // v_cvt_pk_fp8_f32 v5, v6, v7 clamp op_sel:[0,0,0,1]
  // dword0 = 0xD769C005 (bit14=1 op_sel[3], bit15=1 CLAMP, vdst=v5)
  // dword1 = 0x02020F06 (src0=v6, src1=v7, no modifiers)
  .long 0xD769C005
  .long 0x02020F06
  s_endpgm
.Ltest_cvt_pk_fp8_high_end:
.size test_cvt_pk_fp8_high, .Ltest_cvt_pk_fp8_high_end-test_cvt_pk_fp8_high

// ---- Kernel 3: no clamp (should NOT be patched) -------------------------------

// NOCLAMP-LABEL: <test_cvt_pk_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_pk_fp8_f32

.globl test_cvt_pk_fp8_noclamp
.p2align 8
.type test_cvt_pk_fp8_noclamp,@function
test_cvt_pk_fp8_noclamp:
  v_cvt_pk_fp8_f32 v10, v11, v12
  s_endpgm
.Ltest_cvt_pk_fp8_noclamp_end:
.size test_cvt_pk_fp8_noclamp, .Ltest_cvt_pk_fp8_noclamp_end-test_cvt_pk_fp8_noclamp

.rodata
.p2align 8
.amdhsa_kernel test_cvt_pk_fp8_low
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_high
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_noclamp
  .amdhsa_next_free_vgpr 13
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
