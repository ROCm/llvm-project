// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// RUN: SIZE_IN=$(%llvm-readelf -S %t.elf | awk '/\.text /{print $7}') && \
// RUN:   SIZE_OUT=$(%llvm-readelf -S %t.out.elf | awk '/\.text /{print $7}') && \
// RUN:   test $((16#$SIZE_OUT)) -gt $((16#$SIZE_IN))

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <test_f32_16x16x128_fp8_fp8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_fp8
.p2align 8
.type test_f32_16x16x128_fp8_fp8,@function
test_f32_16x16x128_fp8_fp8:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_fp8_fp8, .-test_f32_16x16x128_fp8_fp8

// DISASM-LABEL: <test_f16_16x16x128_bf8_bf8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_f16_16x16x128_bf8_bf8
.p2align 8
.type test_f16_16x16x128_bf8_bf8,@function
test_f16_16x16x128_bf8_bf8:
  v_wmma_f16_16x16x128_bf8_bf8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_bf8_bf8, .-test_f16_16x16x128_bf8_bf8

// DISASM-LABEL: <test_f32_32x16x128_f4>:
// DISASM-NOT:   v_wmma_f32_32x16x128_f4
// DISASM:       s_branch
.globl test_f32_32x16x128_f4
.p2align 8
.type test_f32_32x16x128_f4,@function
test_f32_32x16x128_f4:
  v_wmma_f32_32x16x128_f4 v[32:47], v[0:15], v[16:23], v[32:47]
  s_endpgm
.size test_f32_32x16x128_f4, .-test_f32_32x16x128_f4

// DISASM-LABEL: <test_f32_16x16x128_fp8_bf8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_f32_16x16x128_fp8_bf8
.p2align 8
.type test_f32_16x16x128_fp8_bf8,@function
test_f32_16x16x128_fp8_bf8:
  v_wmma_f32_16x16x128_fp8_bf8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_fp8_bf8, .-test_f32_16x16x128_fp8_bf8

// DISASM-LABEL: <test_f32_16x16x128_bf8_fp8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_fp8
// DISASM:       s_branch
.globl test_f32_16x16x128_bf8_fp8
.p2align 8
.type test_f32_16x16x128_bf8_fp8,@function
test_f32_16x16x128_bf8_fp8:
  v_wmma_f32_16x16x128_bf8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_bf8_fp8, .-test_f32_16x16x128_bf8_fp8

// DISASM-LABEL: <test_f32_16x16x128_bf8_bf8>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_bf8
// DISASM:       s_branch
.globl test_f32_16x16x128_bf8_bf8
.p2align 8
.type test_f32_16x16x128_bf8_bf8,@function
test_f32_16x16x128_bf8_bf8:
  v_wmma_f32_16x16x128_bf8_bf8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_f32_16x16x128_bf8_bf8, .-test_f32_16x16x128_bf8_bf8

// DISASM-LABEL: <test_f16_16x16x128_fp8_fp8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_fp8
// DISASM:       s_branch
.globl test_f16_16x16x128_fp8_fp8
.p2align 8
.type test_f16_16x16x128_fp8_fp8,@function
test_f16_16x16x128_fp8_fp8:
  v_wmma_f16_16x16x128_fp8_fp8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_fp8_fp8, .-test_f16_16x16x128_fp8_fp8

// DISASM-LABEL: <test_f16_16x16x128_fp8_bf8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_bf8
// DISASM:       s_branch
.globl test_f16_16x16x128_fp8_bf8
.p2align 8
.type test_f16_16x16x128_fp8_bf8,@function
test_f16_16x16x128_fp8_bf8:
  v_wmma_f16_16x16x128_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_fp8_bf8, .-test_f16_16x16x128_fp8_bf8

// DISASM-LABEL: <test_f16_16x16x128_bf8_fp8>:
// DISASM-NOT:   v_wmma_f16_16x16x128_bf8_fp8
// DISASM:       s_branch
.globl test_f16_16x16x128_bf8_fp8
.p2align 8
.type test_f16_16x16x128_bf8_fp8,@function
test_f16_16x16x128_bf8_fp8:
  v_wmma_f16_16x16x128_bf8_fp8 v[32:35], v[0:15], v[16:31], v[32:35]
  s_endpgm
.size test_f16_16x16x128_bf8_fp8, .-test_f16_16x16x128_bf8_fp8

// DISASM-LABEL: <test_no_split_required>:
// DISASM:       v_wmma_f32_16x16x32_f16
// DISASM:       v_add_f32
.globl test_no_split_required
.p2align 8
.type test_no_split_required,@function
test_no_split_required:
  v_wmma_f32_16x16x32_f16 v[32:39], v[0:7], v[8:15], v[32:39]
  v_add_f32_e32 v0, v1, v2
  s_endpgm
.size test_no_split_required, .-test_no_split_required

// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], v[32:39]
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]
// DISASM-DAG: v_wmma_f32_16x16x64_fp8_bf8
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_fp8
// DISASM-DAG: v_wmma_f32_16x16x64_bf8_bf8
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_fp8
// DISASM-DAG: v_wmma_f16_16x16x64_fp8_bf8
// DISASM-DAG: v_wmma_f16_16x16x64_bf8_fp8
// DISASM-DAG: v_wmma_f16_16x16x64_bf8_bf8
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[32:39], v[0:7], v[16:23], v[32:39]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-DAG: v_wmma_f32_16x16x128_f8f6f4 v[40:47], v[8:15], v[16:23], v[40:47]{{.*}}matrix_a_fmt:MATRIX_FMT_FP4{{.*}}matrix_b_fmt:MATRIX_FMT_FP4

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
