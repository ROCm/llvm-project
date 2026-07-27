// COM: Hundreds of pair-only far sites must not consume one independent
// COM: source-to-gateway island chain apiece. One affine entry maps source PCs
// COM: into a global sparse prefix; bounded regional gateways then map those
// COM: PCs into branch-reachable sparse body stubs.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: planned 1 hierarchical mirrored gateway group(s) with {{[2-9][0-9]*}} bounded region(s) for 300 pair-only source site(s)
// LOG: hotswap: planned 1 mirrored-stub gateway group(s) for 10 pair-only source site(s)
// LOG-NOT: no safe short-branch gateway
// LOG: RESULT: SUCCESS

// RUN: %llvm-readobj --notes --symbols --sections %t.out.elf > /dev/null
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <source0>:
// DISASM-NEXT: s_call_i64 s[104:105],
// DISASM-NEXT: s_{{(nop|branch)}}

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.macro PATCH_SOURCE name, binding
  .local \name\()_targeter
  .type \name\()_targeter,@function
\name\()_targeter:
  s_branch \name\()_after
  s_endpgm
  .size \name\()_targeter, .-\name\()_targeter

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

.macro AUTO_SOURCE
  PATCH_SOURCE auto_source\@, .local
.endm

.macro RESIDUAL_SOURCE
  .local residual_source\@_targeter
  .type residual_source\@_targeter,@function
residual_source\@_targeter:
  s_branch residual_source\@_after
  s_endpgm
  .size residual_source\@_targeter, .-residual_source\@_targeter

  .local residual_source\@
  .type residual_source\@,@function
residual_source\@:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
residual_source\@_after:
  // Keep the hierarchical population's s[104:105] pair live so these ten
  // sites form a separate, ordinary mirrored-stub group.
  s_mov_b32 s0, s104
  s_mov_b32 s1, s105
  s_mov_b32 s103, s103
  s_mov_b32 s2, vcc_lo
  s_endpgm
  .size residual_source\@, .-residual_source\@
.endm

PATCH_SOURCE source0, .globl
.rept 120
  s_mov_b32 s0, s1
.endr

.rept 149
  AUTO_SOURCE
  .rept 120
    s_mov_b32 s0, s1
  .endr
.endr

// This external, no-fallthrough padding supplies the common 12-byte affine
// entry. The source population spans more than one bounded regional prefix.
s_endpgm
.local gateway_pad
gateway_pad:
  .rept 3
    s_nop 0
  .endr

.rept 10
  RESIDUAL_SOURCE
  .rept 8
    s_mov_b32 s0, s1
  .endr
.endr

// Keep an independent 12-byte gateway available for the residual group. This
// makes the regular mirrored planner run after hierarchical prefixes have
// already been installed and catches either planner overwriting those sizes.
s_endpgm
.local residual_gateway_pad
residual_gateway_pad:
  .rept 3
    s_nop 0
  .endr

.rept 150
  AUTO_SOURCE
  .rept 120
    s_mov_b32 s0, s1
  .endr
.endr

// Keep the trampoline pool outside short-branch reach of every source.
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
