// A K-split's second half replaces src2 with dst. Even when neither source
// slice crosses v255, the temporary src2 bank must therefore match the
// incoming dst bank. Preserve and restore every other incoming mode field.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-COUNT-2: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_k_src2_role>:
// DISASM-NEXT:  s_set_vgpr_msb 0x60
// DISASM-NEXT:  s_branch
// DISASM-LABEL: <test_wmma_k_vgpr_src2_role>:
// DISASM-NEXT:  s_set_vgpr_msb 0x60
// DISASM-NEXT:  s_branch
// DISASM:      v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], 0
// DISASM-NEXT: s_set_vgpr_msb 0x6050
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[32:39]{{.*}}, v[8:15], v[24:31], v[32:39]
// DISASM-NEXT: s_set_vgpr_msb 0x5060
// DISASM-NEXT: s_branch
// DISASM:      v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], v[40:47]
// DISASM-NEXT: s_set_vgpr_msb 0x6050
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[32:39]{{.*}}, v[8:15], v[24:31], v[32:39]
// DISASM-NEXT: s_set_vgpr_msb 0x5060
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_k_src2_role
.p2align 8
.type test_wmma_k_src2_role,@function
test_wmma_k_src2_role:
  // Incoming mode: dst=1, src2=2, src0=src1=0.
  s_set_vgpr_msb 0x60
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_wmma_k_src2_role, .-test_wmma_k_src2_role

.globl test_wmma_k_vgpr_src2_role
.p2align 8
.type test_wmma_k_vgpr_src2_role,@function
test_wmma_k_vgpr_src2_role:
  // The first half must read src2 from bank 2. The second half replaces that
  // operand with dst and must therefore select bank 1 before restoring bank 2.
  s_set_vgpr_msb 0x60
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[40:47]
  s_endpgm
.size test_wmma_k_vgpr_src2_role, .-test_wmma_k_vgpr_src2_role
