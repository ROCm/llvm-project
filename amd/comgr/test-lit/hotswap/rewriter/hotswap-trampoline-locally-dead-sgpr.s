// COM: A far trampoline needs an aligned scratch SGPR pair for its set-PC
// COM: return. When a kernel declares the full numbered SGPR file (s0-s105),
// COM: no aligned pair fits above the high-water mark, so the watermark search
// COM: fails. The dead-pair fallback then scans every aligned pair from the
// COM: top down and reuses one whose incoming value is dead at the site's
// COM: continuation, rescuing a rewrite that would otherwise fail closed.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: The reused pair is the highest aligned pair proven dead at the
// COM: continuation; s105 is written before the patched site, leaving
// COM: s[104:105] dead, so both far edges route through it.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// DISASM-LABEL: <test_far>:
// DISASM: s_get_pc_i64 s[104:105]
// DISASM-NEXT: s_add_nc_u64 s[104:105], s[104:105],
// DISASM-NEXT: s_set_pc_i64 s[104:105]

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
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.size test_far, .-test_far

.type gateway_barrier,@function
gateway_barrier:
  s_endpgm
.size gateway_barrier, .-gateway_barrier
.fill 32, 1, 0

  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the tensor_load above (forces the
  // long-branch path).
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.Ltest_far_end:

.rodata
.p2align 8
.amdhsa_kernel test_far
  .amdhsa_next_free_vgpr 1
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
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
