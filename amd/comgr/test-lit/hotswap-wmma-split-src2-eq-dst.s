// Test K-split with src2 == dst (the C += A*B accumulator-reuse pattern,
// the most common WMMA shape in real kernels).
//
// At the source level, vdst and src2 share the same VGPR range. The
// splitter's K-split second half uses dst as src2 (the carry from the
// first half) regardless of the input's src2 -- which means for this
// shape the second half's src2 is identical to what the source already
// had, but the splitter still has to emit it correctly via the
// transformation rather than blindly reusing the printed src2.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// COM: Source: original K=128 opcode replaced by s_branch into trampoline.
// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
// DISASM:       s_endpgm

// COM: Trampoline: K=64 first half uses (A_lo, B_lo, original_src2 == dst);
// COM: K=64 second half uses (A_hi, B_hi, dst as carry). Both halves
// COM: write back to dst v[16:23], so on this shape the visible operand
// COM: list is identical between halves -- only the sliced A/B differ.
// COM: The two halves are emitted back-to-back in the trampoline body
// COM: with the s_branch-back appended once at the end.
// DISASM:       v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23]
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[8:15], v[16:23], v[16:23]
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], v[16:23]
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
