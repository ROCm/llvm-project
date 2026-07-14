// COM: A far trampoline placed before a boundary-reachable site grows during
// COM: source-window expansion and receives a pool branch island. Queue layout
// COM: must reserve both before deciding that the later site can use short
// COM: branches; otherwise final fixup sees a shifted trampoline and fails.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// DISASM-LABEL: <first_far>:
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM-LABEL: <boundary_site>:
// DISASM-NEXT: s_branch 32767
// DISASM: ds_load_b32 v4, v6 offset:256
// DISASM-NEXT: ds_load_b32 v5, v6 offset:768
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl first_far
.type first_far,@function
first_far:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_mov_b32 s0, s1
  s_mov_b32 s2, s3
  s_mov_b32 s4, s5
  s_endpgm
.size first_far, .-first_far

// Keep the first site far from the pool without donating NOP padding.
.rept 41640
  s_mov_b32 s0, s1
.endr

.globl boundary_site
.type boundary_site,@function
boundary_site:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size boundary_site, .-boundary_site

// A far classification has a legal short-hop set-PC gateway. Before the queue
// reservation fix this site was misclassified as short, so the gateway was
// never assigned and final branch fixup failed after the earlier pool growth.
.rept 4
  s_nop 0
.endr

  // Tuned so the later trampoline is at the short-branch boundary before the
  // earlier far trampoline's final growth and island are accounted for.
  .rept 31100
    s_mov_b32 s0, s1
  .endr

.rodata
.p2align 8
.amdhsa_kernel first_far
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel

.amdhsa_kernel boundary_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: first_far
      .symbol: first_far.kd
      .sgpr_count: 10
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: boundary_site
      .symbol: boundary_site.kd
      .sgpr_count: 10
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
