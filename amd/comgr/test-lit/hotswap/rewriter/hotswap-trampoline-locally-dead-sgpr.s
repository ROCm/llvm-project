// COM: A far trampoline needs an aligned scratch SGPR pair for its set-PC
// COM: return. When a kernel declares the full numbered SGPR file (s0-s105),
// COM: no aligned pair fits above the high-water mark, so the watermark search
// COM: fails. The dead-pair fallback then scans every aligned pair from the
// COM: top down and reuses one whose incoming value is dead at the site's
// COM: continuation, rescuing a rewrite that would otherwise fail closed.
// COM:
// COM: The kernel has two far sites in one function. The first queues a
// COM: deferred trampoline; the second must still prove its own continuation
// COM: liveness while that trampoline is pending, so both reuse a locally dead
// COM: pair instead of the second failing closed.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Both far edges route through an aligned locally dead pair via an
// COM: SGPR-backed set-PC sequence, never gfx1250's broken s_add_pc_i64.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// DISASM-LABEL: <test_far>:
// DISASM: s_get_pc_i64 s[{{[0-9]+}}:{{[0-9]+}}]
// DISASM-NEXT: s_add_nc_u64 s[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}],
// DISASM-NEXT: s_set_pc_i64 s[{{[0-9]+}}:{{[0-9]+}}]

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far
.p2align 8
.type test_far,@function
test_far:
  s_mov_b32 s105, 0
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  // Interior filler keeps the second WMMA beyond s_branch reach of the first
  // site's appended pool, so it too takes the far path while the first
  // trampoline is still pending.
  .rept 40000
    s_mov_b32 s0, s1
  .endr
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_far, .-test_far

.rept 8
  s_nop 0
.endr

  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the WMMA sites above (forces far).
  .rept 40000
    s_mov_b32 s0, s1
  .endr

.rodata
.p2align 8
.amdhsa_kernel test_far
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_far
      .symbol: test_far.kd
      .sgpr_count: 106
      .vgpr_count: 40
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
