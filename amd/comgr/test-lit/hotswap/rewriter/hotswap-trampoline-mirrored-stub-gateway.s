// COM: Dense far sites can exhaust numbered SGPRs and leave only the pair
// COM: already reserved for their return edge. The source PC plus one shared
// COM: relocation-neutral delta selects a sparse branch stub in the pool.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: shared far-dispatch skipped 10 site(s) without four safe SGPRs
// LOG: hotswap: planned 1 mirrored-stub gateway group(s) for 10 pair-only source site(s)
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <source0>:
// DISASM-NEXT: s_call_i64 s[104:105],
// DISASM-NEXT: s_{{(nop|branch)}}
// DISASM-LABEL: <gateway_pad>:
// DISASM-NEXT: s_add_nc_u64 s[104:105], s[104:105],
// DISASM-NEXT: s_set_pc_i64 s[104:105]

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: An in-function NOP run belongs only to that function. Even though the
// COM: foreign run is branch-reachable from every source below, the affine
// COM: planner must not use it as a gateway for a different function.
// RUN: sed 's|^// FOREIGN-ONLY:|  |' %s > %t.foreign.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.foreign.s -o %t.foreign.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.foreign.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=FOREIGN %s
// FOREIGN-NOT: hotswap: planned {{.*}} mirrored-stub gateway group
// FOREIGN: hotswap: error: no safe short-branch gateway for far site

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.macro PATCH_SOURCE name, binding
  \binding \name
  .type \name,@function
\name:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
\name\()_after:
  s_mov_b32 s103, s103
  s_mov_b32 s0, vcc_lo
  s_endpgm
  .size \name, .-\name
.endm

.local targeter
.type targeter,@function
targeter:
  s_branch source0_after
  s_branch source1_after
  s_branch source2_after
  s_branch source3_after
  s_branch source4_after
  s_branch source5_after
  s_branch source6_after
  s_branch source7_after
  s_branch source8_after
  s_branch source9_after
  s_endpgm
  .size targeter, .-targeter

PATCH_SOURCE source0, .globl
.local gateway_pad
// FOREIGN-ONLY:.type gateway_pad,@function
gateway_pad:
  .rept 3
    s_nop 0
  .endr
// FOREIGN-ONLY:.size gateway_pad, .-gateway_pad
PATCH_SOURCE source1, .local
PATCH_SOURCE source2, .local
PATCH_SOURCE source3, .local
PATCH_SOURCE source4, .local
PATCH_SOURCE source5, .local
PATCH_SOURCE source6, .local
PATCH_SOURCE source7, .local
PATCH_SOURCE source8, .local
PATCH_SOURCE source9, .local

.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel source0
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 104
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: source0
      .symbol: source0.kd
      .sgpr_count: 106
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
