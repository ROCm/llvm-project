// COM: Production activation kernels select one of several local callees with
// COM: get-PC/carry materialization, merge the selected address, and reuse it
// COM: across many register calls. Resolve the finite reaching-target set so
// COM: an unrelated required far rewrite may safely use external gateway
// COM: padding. A selector bypass would leave the target unknown and must
// COM: continue to fail closed.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: resolved reusable PC-materialized call
// LOG-SAME: to 2 target(s)
// LOG-NOT: hotswap: unresolved call target
// LOG: hotswap: planned 1 shared far-dispatch gateway group(s) for 8 source site(s)
// LOG: RESULT: SUCCESS

// RUN: sed 's/^\.set unsafe_selector, 0$/.set unsafe_selector, 1/' \
// RUN:   %s > %t.bypass.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.bypass.s -o %t.bypass.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.bypass.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=BYPASS,FAIL %s
// BYPASS: hotswap: unresolved call target
// FAIL: hotswap: unresolved control-flow target disables NOP-sled emission,
// FAIL-SAME: trampoline coalescing, source relocation, and .text gateways
// FAIL: hotswap: error: no safe short-branch gateway for far site
// FAIL: RESULT: ERROR

// RUN: sed 's/^\.set outside_selector, 0$/.set outside_selector, 1/' \
// RUN:   %s > %t.outside.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.outside.s -o %t.outside.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.outside.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.outside.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=OUTSIDE %s
// OUTSIDE: hotswap: resolved reusable PC-materialized call
// OUTSIDE-SAME: to 3 target(s)
// OUTSIDE-NOT: hotswap: unresolved call target
// OUTSIDE: hotswap: planned 1 shared far-dispatch gateway group(s) for 8 source site(s)
// OUTSIDE: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.outside.out.elf \
// RUN:   | %FileCheck --check-prefix=OUTSIDE-DISASM %s
// OUTSIDE-DISASM-LABEL: <reusable_pc_targets>:
// OUTSIDE-DISASM: s_swap_pc_i64
// OUTSIDE-DISASM-NEXT: s_branch

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// DISASM-LABEL: <reusable_pc_targets>:
// DISASM: s_swap_pc_i64
// DISASM: s_call_i64 s[12:13]
// DISASM-NEXT: s_nop 0
// DISASM-LABEL: <gateway_barrier>:
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM: ds_load_b32 v0, v2 offset:256
// DISASM-NEXT: ds_load_b32 v1, v2 offset:768

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.set unsafe_selector, 0
.set outside_selector, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl reusable_pc_targets
.p2align 8
.type reusable_pc_targets,@function
reusable_pc_targets:
.if unsafe_selector
  // This edge reaches the call without executing either get-PC sequence.
  s_cmp_eq_u32 s0, 2
  s_cbranch_scc1 .Lselected
.endif
.if outside_selector
  s_cmp_eq_u32 s0, 3
  s_cbranch_scc1 .Lselect_outside
.endif
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Lselect_second
.Lselect_first:
  s_get_pc_i64 s[2:3]
  s_add_co_i32 s4, callee_first-(.Lselect_first+4)-4, 4
  s_add_co_u32 s2, s2, s4
  s_add_co_ci_u32 s3, s3, 0
  s_branch .Lselected
.Lselect_second:
  s_get_pc_i64 s[2:3]
  s_add_co_i32 s4, callee_second-(.Lselect_second+4)-4, 4
  s_add_co_u32 s2, s2, s4
  s_add_co_ci_u32 s3, s3, 0
.if outside_selector
  s_branch .Lselected
.Lselect_outside:
  s_get_pc_i64 s[2:3]
  s_add_co_i32 s4, outside_text_end-(.Lselect_outside+4)-4, 4
  s_add_co_u32 s2, s2, s4
  s_add_co_ci_u32 s3, s3, 0
.endif
.Lselected:
  s_swap_pc_i64 s[6:7], s[2:3]
  s_branch .Lpatch0
.Lpatch0:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_branch .Lpatch1
.Lpatch1:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_branch .Lpatch2
.Lpatch2:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_branch .Lpatch3
.Lpatch3:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_branch .Lpatch4
.Lpatch4:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_branch .Lpatch5
.Lpatch5:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_branch .Lpatch6
.Lpatch6:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_branch .Lpatch7
.Lpatch7:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
.Lpatch_done:
  s_wait_dscnt 0x0
  s_endpgm
.size reusable_pc_targets, .-reusable_pc_targets

.local callee_first
.type callee_first,@function
callee_first:
  s_mov_b32 s8, 1
  s_set_pc_i64 s[6:7]
.size callee_first, .-callee_first

.local callee_second
.type callee_second,@function
callee_second:
  s_mov_b32 s8, 2
  s_set_pc_i64 s[6:7]
.size callee_second, .-callee_second

.type gateway_barrier,@function
gateway_barrier:
  s_endpgm
.size gateway_barrier, .-gateway_barrier
.fill 20, 1, 0

.rept 40000
  s_mov_b32 s10, s11
.endr

outside_text_end:
.rodata
.p2align 8
.amdhsa_kernel reusable_pc_targets
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: reusable_pc_targets
      .symbol: reusable_pc_targets.kd
      .sgpr_count: 12
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
