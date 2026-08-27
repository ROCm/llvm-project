// A dynamic write to WAVE_MODE does not reveal the incoming VGPR-MSB fields.
// Keep the required WMMA split fail-closed instead of assuming ABI entry mode.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck %s
// RUN: test ! -e %t.out.elf
// CHECK-LABEL: WMMA split: cannot determine VGPR-MSB mode
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_dynamic_mode
.p2align 8
.type test_wmma_dynamic_mode,@function
test_wmma_dynamic_mode:
  s_setreg_b32 hwreg(HW_REG_WAVE_MODE), s0
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_wmma_dynamic_mode, .-test_wmma_dynamic_mode
