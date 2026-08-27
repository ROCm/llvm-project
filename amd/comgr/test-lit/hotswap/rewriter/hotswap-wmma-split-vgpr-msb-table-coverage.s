// Table coverage: exercise every entry in the WMMA split table through the
// VGPR-MSB-aware split + bracket path. Each kernel holds one splittable opcode
// whose upper source half crosses v255, so the split runs and brackets the
// crossing half. The per-SplitKind bracket bytes are checked in detail by
// hotswap-wmma-split-vgpr-msb.s (Split128to64FP8BF8) and
// hotswap-wmma-split-vgpr-msb-m-roles.s (Split32x16to16x16F4); this test
// guarantees no table entry is left unsplit.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: error:
// API: WMMA split: patched v_wmma_f16_16x16x128_fp8_fp8
// API: WMMA split: patched v_wmma_f16_16x16x128_fp8_bf8
// API: WMMA split: patched v_wmma_f16_16x16x128_bf8_fp8
// API: WMMA split: patched v_wmma_f16_16x16x128_bf8_bf8
// API: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API: WMMA split: patched v_wmma_f32_16x16x128_fp8_bf8
// API: WMMA split: patched v_wmma_f32_16x16x128_bf8_fp8
// API: WMMA split: patched v_wmma_f32_16x16x128_bf8_bf8
// API: WMMA split: patched v_wmma_f32_32x16x128_f4
// API: RESULT: SUCCESS

// A second rewrite must be byte-identical.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl k_f16_fp8_fp8
.p2align 8
.type k_f16_fp8_fp8,@function
k_f16_fp8_fp8:
  s_set_vgpr_msb 0
  v_wmma_f16_16x16x128_fp8_fp8 v[16:19], v[250:265], v[100:115], v[16:19]
  s_endpgm
.size k_f16_fp8_fp8, .-k_f16_fp8_fp8

.globl k_f16_fp8_bf8
.p2align 8
.type k_f16_fp8_bf8,@function
k_f16_fp8_bf8:
  s_set_vgpr_msb 0
  v_wmma_f16_16x16x128_fp8_bf8 v[16:19], v[250:265], v[100:115], v[16:19]
  s_endpgm
.size k_f16_fp8_bf8, .-k_f16_fp8_bf8

.globl k_f16_bf8_fp8
.p2align 8
.type k_f16_bf8_fp8,@function
k_f16_bf8_fp8:
  s_set_vgpr_msb 0
  v_wmma_f16_16x16x128_bf8_fp8 v[16:19], v[250:265], v[100:115], v[16:19]
  s_endpgm
.size k_f16_bf8_fp8, .-k_f16_bf8_fp8

.globl k_f16_bf8_bf8
.p2align 8
.type k_f16_bf8_bf8,@function
k_f16_bf8_bf8:
  s_set_vgpr_msb 0
  v_wmma_f16_16x16x128_bf8_bf8 v[16:19], v[250:265], v[100:115], v[16:19]
  s_endpgm
.size k_f16_bf8_bf8, .-k_f16_bf8_bf8

.globl k_f32_fp8_fp8
.p2align 8
.type k_f32_fp8_fp8,@function
k_f32_fp8_fp8:
  s_set_vgpr_msb 0
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[250:265], v[100:115], v[16:23]
  s_endpgm
.size k_f32_fp8_fp8, .-k_f32_fp8_fp8

.globl k_f32_fp8_bf8
.p2align 8
.type k_f32_fp8_bf8,@function
k_f32_fp8_bf8:
  s_set_vgpr_msb 0
  v_wmma_f32_16x16x128_fp8_bf8 v[16:23], v[250:265], v[100:115], v[16:23]
  s_endpgm
.size k_f32_fp8_bf8, .-k_f32_fp8_bf8

.globl k_f32_bf8_fp8
.p2align 8
.type k_f32_bf8_fp8,@function
k_f32_bf8_fp8:
  s_set_vgpr_msb 0
  v_wmma_f32_16x16x128_bf8_fp8 v[16:23], v[250:265], v[100:115], v[16:23]
  s_endpgm
.size k_f32_bf8_fp8, .-k_f32_bf8_fp8

.globl k_f32_bf8_bf8
.p2align 8
.type k_f32_bf8_bf8,@function
k_f32_bf8_bf8:
  s_set_vgpr_msb 0
  v_wmma_f32_16x16x128_bf8_bf8 v[16:23], v[250:265], v[100:115], v[16:23]
  s_endpgm
.size k_f32_bf8_bf8, .-k_f32_bf8_bf8

.globl k_f32_32x16x128_f4
.p2align 8
.type k_f32_32x16x128_f4,@function
k_f32_32x16x128_f4:
  s_set_vgpr_msb 0
  v_wmma_f32_32x16x128_f4 v[250:265], v[248:263], v[100:107], v[252:267]
  s_endpgm
.size k_f32_32x16x128_f4, .-k_f32_32x16x128_f4
