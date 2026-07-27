// COM: A required DS2 split should use proven padding in its own function even
// COM: when a large unrelated text tail places the appended pool outside
// COM: s_branch reach. Keeping the replacement local avoids a registerless
// COM: pool return and its object-size-dependent relay chain.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: trampoline: ds_load_addtid_b32 -> ds_load_b32
// LOG-NOT: hotswap: ds_2addr: used compact continuation
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_ds2_far_local_nop>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: ds_load_b32 v0, v2 offset:4
// DISASM-NEXT: ds_load_b32 v1, v2 offset:12
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_branch
// DISASM-LABEL: <test_ds2_far_compact_continuation>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_endpgm
// DISASM-NOT: ds_load_2addr
// DISASM-LABEL: <test_ds2_far_external_padding>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: ds_load_b32 v0, v2 offset:4
// DISASM-NEXT: ds_load_b32 v1, v2 offset:12
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_branch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_far_local_nop
.p2align 8
.type test_ds2_far_local_nop,@function
test_ds2_far_local_nop:
  ds_load_2addr_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_endpgm
.size test_ds2_far_local_nop, .-test_ds2_far_local_nop
s_mov_b32 s0, s0

// A DS2 followed by a wait should prefer a safe scratch-register trampoline
// over consuming an exact 20-byte compact-continuation tail.
.globl test_ds2_far_compact_continuation
.p2align 8
.type test_ds2_far_compact_continuation,@function
test_ds2_far_compact_continuation:
  ds_load_2addr_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.size test_ds2_far_compact_continuation, .-test_ds2_far_compact_continuation
.fill 20, 1, 0
s_mov_b32 s0, s0
.type compact_tail_barrier,@function
compact_tail_barrier:
  s_endpgm
.size compact_tail_barrier, .-compact_tail_barrier
.fill 20, 1, 0
s_mov_b32 s0, s0
.rept 40000
  s_mov_b32 s0, s0
.endr

.globl test_ds2_far_external_padding
.p2align 8
.type test_ds2_far_external_padding,@function
test_ds2_far_external_padding:
  ds_load_2addr_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.size test_ds2_far_external_padding, .-test_ds2_far_external_padding

// Ordinary far-pool replacements use the same audited external-local policy,
// so their source and return edges do not create object-spanning relay chains.
.globl test_ds2_source_tail_planner
.p2align 8
.type test_ds2_source_tail_planner,@function
test_ds2_source_tail_planner:
  ds_load_addtid_b32 v3 offset:128
  s_wait_dscnt 0x0
  s_endpgm
.size test_ds2_source_tail_planner, .-test_ds2_source_tail_planner

// Proven unreachable code-end alignment padding outside the sized function
// can host local replacements; any residual dwords remain available to global
// routing. Production linked objects use this pattern extensively.
.rept 32
  s_code_end
.endr

// Keep a hypothetical appended trampoline pool outside signed s_branch reach.
.rept 40000
  s_mov_b32 s2, s3
.endr

.rodata
.p2align 8
.amdhsa_kernel test_ds2_far_local_nop
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdhsa_kernel test_ds2_far_external_padding
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdhsa_kernel test_ds2_source_tail_planner
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdhsa_kernel test_ds2_far_compact_continuation
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds2_far_local_nop
      .symbol: test_ds2_far_local_nop.kd
      .sgpr_count: 4
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds2_far_external_padding
      .symbol: test_ds2_far_external_padding.kd
      .sgpr_count: 4
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds2_source_tail_planner
      .symbol: test_ds2_source_tail_planner.kd
      .sgpr_count: 4
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds2_far_compact_continuation
      .symbol: test_ds2_far_compact_continuation.kd
      .sgpr_count: 4
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
