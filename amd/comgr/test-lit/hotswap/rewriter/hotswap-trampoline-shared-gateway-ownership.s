// COM: A shared dispatcher gateway and every source it serves must belong to
// COM: the same function. An otherwise reachable NOP run in a foreign function
// COM: is not a legal gateway.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=GOOD %s
// GOOD: hotswap: planned 1 shared far-dispatch gateway group(s) for 10 source site(s)
// GOOD: RESULT: SUCCESS

// RUN: sed 's|^// FOREIGN-ONLY:|  |' %s > %t.foreign.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.foreign.s -o %t.foreign.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.foreign.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=FOREIGN %s
// FOREIGN-NOT: hotswap: planned {{.*}} shared far-dispatch gateway group
// FOREIGN: hotswap: error: no safe short-branch gateway for far site

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.macro PATCH_SOURCE number
  s_branch source\number\()_after
source\number:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
source\number\()_after:
.endm

.globl sources
.type sources,@function
sources:
PATCH_SOURCE 0
PATCH_SOURCE 1
PATCH_SOURCE 2
PATCH_SOURCE 3
PATCH_SOURCE 4
PATCH_SOURCE 5
PATCH_SOURCE 6
PATCH_SOURCE 7
PATCH_SOURCE 8
PATCH_SOURCE 9
  s_endpgm
.size sources, .-sources

.local gateway_pad
// FOREIGN-ONLY:.type gateway_pad,@function
gateway_pad:
  .rept 5
    s_nop 0
  .endr
// FOREIGN-ONLY:.size gateway_pad, .-gateway_pad

.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel sources
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 96
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: sources
      .symbol: sources.kd
      .sgpr_count: 98
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
