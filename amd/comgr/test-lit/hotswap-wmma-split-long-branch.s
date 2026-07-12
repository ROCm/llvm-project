// COM: HSV-009 / PLAT-205406: WMMA-split shares emitToTrampoline with the other
// COM: patch families. A split site beyond s_branch's +-128 KB reach uses the
// COM: same scratch-backed set-PC return as DS and tensor patches, avoiding a
// COM: second s_add_pc_i64 on the hot edge. A large .rept filler (~160 KB,
// COM: non-NOP) forces the far case.

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
// DISASM-NEXT: s_cselect_b32 s4, 1, 0
// DISASM-NEXT: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_co_u32 s2, s2,
// DISASM-NEXT: s_add_co_ci_u32 s3, s3,
// DISASM-NEXT: s_cmp_lg_u32 s4, 0
// DISASM-NEXT: s_set_pc_i64 s[2:3]

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

.rodata
.p2align 8
.amdhsa_kernel test_wsplit_far
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wsplit_far
      .symbol: test_wsplit_far.kd
      .sgpr_count: 0
      .vgpr_count: 40
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
