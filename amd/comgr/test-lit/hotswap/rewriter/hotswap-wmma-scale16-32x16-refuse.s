// COM: Fail-closed overlap gates for the staged M+K lowering. Each variant is
// COM: assembled into its own object so every rejection is exercised
// COM: independently.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wa,-defsym,CASE=1 %s -o %t.da.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.da.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=D-A %s
// D-A: destination overlaps matrix A
// D-A: RESULT: ERROR

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wa,-defsym,CASE=2 %s -o %t.db.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.db.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=D-B %s
// D-B: destination overlaps matrix B
// D-B: RESULT: ERROR

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wa,-defsym,CASE=3 %s -o %t.ab.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.ab.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=A-B %s
// A-B: matrix A overlaps matrix B
// A-B: RESULT: ERROR

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wa,-defsym,CASE=4 %s -o %t.dc.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.dc.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=D-C %s
// D-C: partial destination/src2 overlap
// D-C: RESULT: ERROR

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wa,-defsym,CASE=5 %s -o %t.ac.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.ac.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=A-C %s
// A-C: matrix A overlaps src2
// A-C: RESULT: ERROR

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wa,-defsym,CASE=6 %s -o %t.scale-matrix.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.scale-matrix.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=SCALE %s
// SCALE: scale pair overlaps a staged matrix operand
// SCALE: RESULT: ERROR

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wa,-defsym,CASE=7 %s -o %t.scale-scale.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.scale-scale.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=SCALE %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl test_wmma_scale16_32x16_refuse
.p2align 8
.type test_wmma_scale16_32x16_refuse,@function
test_wmma_scale16_32x16_refuse:
.if CASE == 1
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[8:23], v[32:39], v[0:15], v[40:41], v[42:43]
.elseif CASE == 2
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[8:15], v[0:15], v[40:41], v[42:43]
.elseif CASE == 3
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[24:31], v[0:15], v[40:41], v[42:43]
.elseif CASE == 4
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[24:39], v[40:47], v[8:23], v[48:49], v[50:51]
.elseif CASE == 5
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[16:31], v[40:41], v[42:43]
.elseif CASE == 6
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[0:15], v[16:17], v[42:43]
.elseif CASE == 7
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[0:15], v[40:41], v[40:41]
.else
  .error "unknown CASE"
.endif
  s_endpgm
.Ltest_wmma_scale16_32x16_refuse_end:
.size test_wmma_scale16_32x16_refuse, .Ltest_wmma_scale16_32x16_refuse_end-test_wmma_scale16_32x16_refuse

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_32x16_refuse
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale16_32x16_refuse
      .symbol: test_wmma_scale16_32x16_refuse.kd
      .sgpr_count: 2
      .vgpr_count: 52
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
