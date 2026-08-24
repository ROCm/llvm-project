// An M-split slices dst, src0, and VGPR src2 independently. Normalize every
// upper-half base into its operand's mode field, keep broadcast src1 unchanged,
// and restore the exact nonzero incoming mode after the upper half.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-COUNT-2: WMMA split: patched v_wmma_f32_32x16x128_f4
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_m_operand_roles>:
// DISASM-NEXT:  s_set_vgpr_msb 9
// DISASM-NEXT:  s_branch
// DISASM-LABEL: <test_wmma_m_immediate_src2_role>:
// DISASM-NEXT:  s_set_vgpr_msb 57
// DISASM-NEXT:  s_branch
// DISASM:      v_wmma_f32_16x16x128_f8f6f4 v[250:257], v[248:255], v[100:107], v[252:259]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-NEXT: s_set_vgpr_msb 0x95a
// DISASM-NEXT: v_wmma_f32_16x16x128_f8f6f4 v[2:9]{{.*}}, v[0:7]{{.*}}, v[100:107]{{.*}}, v[4:11]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-NEXT: s_set_vgpr_msb 0x5a09
// DISASM-NEXT: s_branch
// DISASM:      v_wmma_f32_16x16x128_f8f6f4 v[250:257], v[248:255], v[100:107], 0{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-NEXT: s_set_vgpr_msb 0x397a
// DISASM-NEXT: v_wmma_f32_16x16x128_f8f6f4 v[2:9]{{.*}}, v[0:7]{{.*}}, v[100:107]{{.*}}, 0{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-NEXT: s_set_vgpr_msb 0x7a39
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_m_operand_roles
.p2align 8
.type test_wmma_m_operand_roles,@function
test_wmma_m_operand_roles:
  // Incoming mode: src0=1, src1=2, src2=0, dst=0.
  s_set_vgpr_msb 9
  v_wmma_f32_32x16x128_f4 v[250:265], v[248:263], v[100:107], v[252:267]
  s_endpgm
.size test_wmma_m_operand_roles, .-test_wmma_m_operand_roles

.globl test_wmma_m_immediate_src2_role
.p2align 8
.type test_wmma_m_immediate_src2_role,@function
test_wmma_m_immediate_src2_role:
  // Incoming mode: src0=1, src1=2, src2=3, dst=0. Immediate src2 does not
  // consume its role, but the temporary mode must preserve it exactly.
  s_set_vgpr_msb 0x39
  v_wmma_f32_32x16x128_f4 v[250:265], v[248:263], v[100:107], 0
  s_endpgm
.size test_wmma_m_immediate_src2_role, .-test_wmma_m_immediate_src2_role
