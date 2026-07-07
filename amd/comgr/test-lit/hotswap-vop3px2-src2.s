// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf -x .text %t.out.elf | %FileCheck --check-prefix=ENCODING %s
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// RUN: cmp %t.out.elf %t.out2.elf

// API: RESULT: SUCCESS
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4
// DISASM: s_endpgm
// ENCODING-LABEL: Hex dump of section '.text':
// ENCODING-NEXT: 0x{{[0-9a-f]+}} 000835cc 0105020c 000833cc 0831a214
// API2: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_vop3px2_src2
.p2align 8
.type test_vop3px2_src2,@function
test_vop3px2_src2:
  v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[40:47], v1, v2 matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
  s_endpgm
.Ltest_vop3px2_src2_end:
.size test_vop3px2_src2, .Ltest_vop3px2_src2_end-test_vop3px2_src2

.rodata
.p2align 8
.amdhsa_kernel test_vop3px2_src2
  .amdhsa_next_free_vgpr 48
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
