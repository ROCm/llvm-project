// COM: A shared dispatcher may admit a distant source through another
// COM: member's second dword. Each newly admitted member must publish its
// COM: source+4 tail, never its source s_call at source+0, as the next relay.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: planned 1 shared far-dispatch gateway group(s) for 8 source site(s)
// LOG: RESULT: SUCCESS

// COM: source5 is admitted through source7's tail, then publishes its own tail
// COM: for source3. That second hop locks the newly admitted anchor at
// COM: source+4 instead of source+0.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <source5>:
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_branch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.macro PATCH_SOURCE number
  s_branch source\number\()_after
source\number:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
source\number\()_after:
  .rept 15000
    s_mov_b32 s0, s1
  .endr
.endm

.globl chain
.type chain,@function
chain:
PATCH_SOURCE 0
PATCH_SOURCE 1
PATCH_SOURCE 2
PATCH_SOURCE 3
PATCH_SOURCE 4
PATCH_SOURCE 5
PATCH_SOURCE 6
PATCH_SOURCE 7
PATCH_SOURCE 8
  s_branch source9_after
source9:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
source9_after:
  s_mov_b32 s0, s1
gateway_pad:
  .rept 5
    s_nop 0
  .endr
  .rept 15000
    s_mov_b32 s0, s1
  .endr
  s_endpgm
.size chain, .-chain

.rodata
.p2align 8
.amdhsa_kernel chain
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: chain
      .symbol: chain.kd
      .sgpr_count: 34
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
