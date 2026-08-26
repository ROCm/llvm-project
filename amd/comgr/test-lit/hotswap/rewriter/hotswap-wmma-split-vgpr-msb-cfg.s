// Recover exact VGPR-MSB state through CFG joins instead of rejecting every
// direct target. This covers a backedge loop, a nonzero fixed point, and a
// non-MODE SETREG that must preserve the tracked fields.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-COUNT-3: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_zero_loop>:
// DISASM:       s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
// DISASM:       s_branch
// DISASM:       s_cbranch_scc1
// DISASM-LABEL: <test_wmma_nonzero_loop>:
// DISASM:       s_set_vgpr_msb 0x60
// DISASM:       s_branch
// DISASM:       s_cbranch_scc1
// DISASM-LABEL: <test_wmma_nonmode_setreg>:
// DISASM:       s_set_vgpr_msb 0x60
// DISASM-NEXT:  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_STATUS), 7
// DISASM-NEXT:  s_branch
// DISASM:       Disassembly of section :
// DISASM:       v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  s_branch
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  s_set_vgpr_msb 0x6050
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  s_set_vgpr_msb 0x5060
// DISASM-NEXT:  s_branch
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  s_set_vgpr_msb 0x6050
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  s_set_vgpr_msb 0x5060
// DISASM-NEXT:  s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl test_wmma_zero_loop
.p2align 8
.type test_wmma_zero_loop,@function
test_wmma_zero_loop:
  // gfx1250's SETREG fixup writes VGPR-MSB fields from imm32[19:12], which
  // are zero here even though the selected WAVE_MODE bit is set to one.
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
.Lzero_loop:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_cbranch_scc1 .Lzero_loop
  s_endpgm
.size test_wmma_zero_loop, .-test_wmma_zero_loop

.globl test_wmma_nonzero_loop
.p2align 8
.type test_wmma_nonzero_loop,@function
test_wmma_nonzero_loop:
  s_set_vgpr_msb 0x60
.Lnonzero_loop:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_cbranch_scc1 .Lnonzero_loop
  s_endpgm
.size test_wmma_nonzero_loop, .-test_wmma_nonzero_loop

.globl test_wmma_nonmode_setreg
.p2align 8
.type test_wmma_nonmode_setreg,@function
test_wmma_nonmode_setreg:
  s_set_vgpr_msb 0x60
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_STATUS), 7
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_wmma_nonmode_setreg, .-test_wmma_nonmode_setreg
