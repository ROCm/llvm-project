// COM: A0 cannot execute s_add_pc_i64. Both edges use an SGPR-backed set-PC
// COM: sequence. The 20-byte source window carries the forward sequence
// COM: directly. A large non-NOP .rept filler (~160 KB) pushes the pool past
// COM: s_branch's reach to force the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_far>:
// DISASM-NEXT: s_get_pc_i64 s[12:13]
// DISASM-NEXT: s_add_nc_u64 s[12:13], s[12:13],
// DISASM-NEXT: s_set_pc_i64 s[12:13]
// DISASM-NEXT: s_endpgm
// DISASM-LABEL: <gateway_barrier>:
// DISASM-NEXT: s_endpgm
// DISASM: s_mov_b64 vcc, -1
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_get_pc_i64 s[12:13]
// DISASM-NEXT: s_add_nc_u64 s[12:13], s[12:13],
// DISASM-NEXT: s_set_pc_i64 s[12:13]

// METADATA: .name:           test_far
// METADATA: .sgpr_count:     16

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: A kernel with no globally unused aligned two-SGPR block can reuse a
// COM: pair whose incoming values are dead in both the replacement and every
// COM: continuation path.
// RUN: sed -e 's/s_mov_b64 vcc, -1/s_mov_b32 s105, 0/' \
// RUN:   -e 's/\.amdhsa_next_free_sgpr 12/.amdhsa_next_free_sgpr 106/' \
// RUN:   -e 's/\.sgpr_count: 14/.sgpr_count: 106/' %s > %t.full-sgpr.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.full-sgpr.s -o %t.full-sgpr.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.full-sgpr.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.full-sgpr.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=FULL-PAIR-LOG %s
// FULL-PAIR-LOG: hotswap: safe far return: reusing locally dead s[104:105]
// FULL-PAIR-LOG: RESULT: SUCCESS

// COM: Search every aligned pair, not just the highest eight. Every pair above
// COM: s[30:31] has a reachable incoming-value read; s[30:31] is overwritten
// COM: first and is therefore the highest locally dead pair.
// RUN: sed -e 's|^// LOW-PAIR-ONLY:|  |' %t.full-sgpr.s > %t.low-pair.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.low-pair.s -o %t.low-pair.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.low-pair.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.low-pair.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=LOW-PAIR-LOG %s
// LOW-PAIR-LOG: hotswap: safe far return: reusing locally dead s[30:31]
// LOW-PAIR-LOG: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.low-pair.out.elf \
// RUN:   | %FileCheck --check-prefix=LOW-PAIR-DISASM %s
// LOW-PAIR-DISASM: s_get_pc_i64 s[30:31]
// LOW-PAIR-DISASM-NEXT: s_add_nc_u64 s[30:31], s[30:31],
// LOW-PAIR-DISASM-NEXT: s_set_pc_i64 s[30:31]

// COM: If every aligned pair is live, the rewrite still fails closed.
// RUN: sed -e 's|^// ALL-PAIRS-LIVE:|  |' %t.full-sgpr.s \
// RUN:   > %t.all-pairs-live.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.all-pairs-live.s -o %t.all-pairs-live.elf
// RUN: hotswap-rewrite %t.all-pairs-live.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR | %FileCheck --check-prefix=FAIL %s
// FAIL: RESULT: ERROR

// COM: A metadata-less object also fails closed because scratch usage cannot
// COM: be charged to its owning kernel.
// RUN: sed '/^.amdgpu_metadata$/,/^.end_amdgpu_metadata$/d' %s > %t.nometa.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.nometa.s -o %t.nometa.elf
// RUN: hotswap-rewrite %t.nometa.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR | %FileCheck --check-prefix=FAIL %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far
.p2align 8
.type test_far,@function
test_far:
  s_mov_b64 vcc, -1
  tensor_load_to_lds s[0:3], s[4:11]
// LOW-PAIR-ONLY:s_mov_b64 s[30:31], 0
// LOW-PAIR-ONLY:.irp live_reg, s32, s34, s36, s38, s40, s42, s44, s46, s48, s50, s52, s54, s56, s58, s60, s62, s64, s66, s68, s70, s72, s74, s76, s78, s80, s82, s84, s86, s88, s90, s92, s94, s96, s98, s100, s102, s104
// LOW-PAIR-ONLY:s_mov_b32 s1, \live_reg
// LOW-PAIR-ONLY:.endr
// ALL-PAIRS-LIVE:.irp live_reg, s0, s2, s4, s6, s8, s10, s12, s14, s16, s18, s20, s22, s24, s26, s28, s30, s32, s34, s36, s38, s40, s42, s44, s46, s48, s50, s52, s54, s56, s58, s60, s62, s64, s66, s68, s70, s72, s74, s76, s78, s80, s82, s84, s86, s88, s90, s92, s94, s96, s98, s100, s102, s104
// ALL-PAIRS-LIVE:s_mov_b32 s1, \live_reg
// ALL-PAIRS-LIVE:.endr
  s_endpgm
.size test_far, .-test_far

// Provide external zero-filled alignment space after a separate
// no-fallthrough function.
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
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_far
      .symbol: test_far.kd
      .sgpr_count: 14
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
