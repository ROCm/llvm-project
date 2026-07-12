// COM: HSV-009 / PLAT-205406: WMMA-split shares emitToTrampoline with the other
// COM: patch families. A split site beyond s_branch's +-128 KB reach uses the
// COM: same sign-extended literal32 return as DS and tensor patches, avoiding
// COM: the 64-bit-literal form that corrupts wave state on gfx1250 A0. A large
// COM: .rept filler (~160 KB, non-NOP) forces the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The site redirects to two K=64 halves and returns with the 8-byte
// COM: negative literal32 form.
// DISASM-LABEL: <test_wsplit_far>:
// DISASM-NEXT: s_add_pc_i64
// DISASM: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: s_add_pc_i64 0xffff{{[0-9a-f]+}}

// COM: Idempotency: rewriting the output again must be a no-op.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wsplit_far
.p2align 8
.type test_wsplit_far,@function
test_wsplit_far:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the WMMA above (forces the long-branch path).
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.size test_wsplit_far, .-test_wsplit_far
