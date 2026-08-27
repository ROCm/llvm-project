// On gfx1250, an immediate MODE write establishes all four VGPR-MSB fields.
// Immediate bits [19:12] are rotated into s_set_vgpr_msb order; 0x81 becomes
// mode 0x60 and must be restored around the split WMMA's upper half.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_setreg_mode>:
// DISASM:       s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 0x81000
// DISASM:       s_branch
// DISASM:       s_set_vgpr_msb 0x6050
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  s_set_vgpr_msb 0x5060

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_setreg_mode
.p2align 8
.type test_wmma_setreg_mode,@function
test_wmma_setreg_mode:
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 0x81000
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_wmma_setreg_mode, .-test_wmma_setreg_mode
