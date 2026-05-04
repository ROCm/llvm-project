// Test the WMMA splitter's refusal paths: instructions that match the
// splittable mnemonic table but whose operands fall outside the supported
// shape must be left in place rather than silently rewritten to wrong
// asm. The hotswap rewriter still returns SUCCESS at the API level
// because no patches were applied -- the original A0-incompatible opcode
// remains, and the runtime will report a load-time error rather than
// run a miscompiled kernel.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// -- Refuse: non-zero src2 inline immediate ---------------------------------
//
// extractWmmaOps accepts only inline imm 0 (the compiler-folded zero
// accumulator). Other integer literals or FP inline constants would need
// printer-mediated formatting to emit canonically -- the splitter
// currently lacks the public-API access for that, so it refuses rather
// than risk a wrong encoding. The original opcode must therefore appear
// unchanged in the output disassembly.
//
// DISASM-LABEL: <test_refuse_nonzero_src2_imm>:
// DISASM:       v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], 1
.globl test_refuse_nonzero_src2_imm
.p2align 8
.type test_refuse_nonzero_src2_imm,@function
test_refuse_nonzero_src2_imm:
  v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], 1
  s_endpgm
.size test_refuse_nonzero_src2_imm, .-test_refuse_nonzero_src2_imm

// COM: No replacement K=64 mnemonic should appear -- the only WMMA in
// COM: this test refused to split, so the trampoline append path was
// COM: never taken.
// DISASM-NOT: v_wmma_f32_16x16x64_fp8_fp8

// COM: Idempotency: a refused split is still bytewise-stable across
// COM: rewrites.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
