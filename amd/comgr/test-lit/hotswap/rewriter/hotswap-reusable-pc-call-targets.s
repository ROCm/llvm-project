// COM: Production activation kernels select one of several local callees with
// COM: get-PC/carry materialization, merge the selected address, and reuse it
// COM: across many register calls. Resolve the finite reaching-target set so
// COM: an unrelated required far rewrite may safely use external gateway
// COM: padding. A selector bypass would leave the target unknown and must
// COM: continue to fail closed.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: resolved PC-materialized call
// LOG: hotswap: resolved reusable PC-materialized call
// LOG-SAME: to 1 target(s)
// LOG: hotswap: resolved reusable PC-materialized call
// LOG-SAME: to 2 target(s)
// LOG-NOT: hotswap: unresolved call target
// LOG: hotswap: planned 1 shared far-dispatch gateway group(s) for 8 source site(s)
// LOG: RESULT: SUCCESS

// RUN: sed 's/^\.set unsafe_selector, 0$/.set unsafe_selector, 1/' \
// RUN:   %s > %t.bypass.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.bypass.s -o %t.bypass.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.bypass.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=BYPASS,FAIL %s
// BYPASS: hotswap: unresolved call target
// OVERLAP: hotswap: unresolved call target
// CLOBBER: hotswap: resolved reusable PC-materialized call
// CLOBBER: hotswap: unresolved call target
// COM: The roll-up conservatively invalidates the bootstrap call itself when
// COM: its callee clobbers the reusable target pair.
// BOOTSTRAP-CLOBBER: hotswap: unresolved call target
// BOOTSTRAP-CLOBBER: hotswap: unresolved call target
// TAIL: hotswap: resolved reusable PC-materialized call
// TAIL: hotswap: unresolved call target
// INDIRECT: hotswap: resolved reusable PC-materialized call
// INDIRECT: hotswap: unresolved call target
// EXTERNAL: hotswap: unresolved call target
// UNDECODED: hotswap: resolved reusable PC-materialized call
// UNDECODED: hotswap: unresolved call target
// RECONVERGE: hotswap: resolved reusable PC-materialized call
// RECONVERGE: hotswap: unresolved call target
// FAIL: hotswap: unresolved control-flow target disables NOP-sled emission,
// FAIL-SAME: trampoline coalescing, source relocation, and .text gateways
// FAIL: hotswap: error: no safe short-branch gateway for far site
// FAIL: RESULT: ERROR

// RUN: sed 's/^\.set overlap_delta, 0$/.set overlap_delta, 1/' \
// RUN:   %s > %t.overlap.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.overlap.s -o %t.overlap.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.overlap.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=OVERLAP,FAIL %s

// RUN: sed 's/^\.set clobber_target, 0$/.set clobber_target, 1/' \
// RUN:   %s > %t.clobber.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.clobber.s -o %t.clobber.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.clobber.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=CLOBBER,FAIL %s

// RUN: sed 's/^\.set clobber_bootstrap, 0$/.set clobber_bootstrap, 1/' \
// RUN:   %s > %t.bootstrap-clobber.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.bootstrap-clobber.s -o %t.bootstrap-clobber.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.bootstrap-clobber.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=BOOTSTRAP-CLOBBER,FAIL %s

// RUN: sed 's/^\.set tail_escape, 0$/.set tail_escape, 1/' \
// RUN:   %s > %t.tail.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.tail.s -o %t.tail.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.tail.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=TAIL,FAIL %s

// RUN: sed 's/^\.set indirect_escape, 0$/.set indirect_escape, 1/' \
// RUN:   %s > %t.indirect.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.indirect.s -o %t.indirect.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.indirect.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=INDIRECT,FAIL %s

// RUN: sed 's/^\.set external_entry, 0$/.set external_entry, 1/' \
// RUN:   %s > %t.external.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.external.s -o %t.external.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.external.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=EXTERNAL,FAIL %s

// RUN: sed 's/^\.set outside_selector, 0$/.set outside_selector, 1/' \
// RUN:   %s > %t.outside.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.outside.s -o %t.outside.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.outside.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=OUTSIDE %s
// COM: The joint roll-up audit may conservatively decline the independent
// COM: bootstrap call once this variant leaves the object-wide entry set open.
// OUTSIDE: hotswap: resolved reusable PC-materialized call
// OUTSIDE-SAME: to 1 target(s)
// OUTSIDE: hotswap: resolved reusable PC-materialized call
// OUTSIDE-SAME: to 3 target(s)
// OUTSIDE: hotswap: unresolved call target
// OUTSIDE-SAME: (s_swap_pc_i64)
// OUTSIDE: hotswap: unresolved control-flow target disables NOP-sled emission,
// OUTSIDE-SAME: trampoline coalescing, source relocation, and .text gateways
// OUTSIDE: hotswap: error: no safe short-branch gateway for far site
// OUTSIDE: RESULT: ERROR

// An undecoded slot between the two reused calls must not let the later call
// resolve: its bytes could clobber the target pair or divert control flow.
// RUN: sed 's/^\.set undecoded_gap, 0$/.set undecoded_gap, 1/' \
// RUN:   %s > %t.undecoded.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.undecoded.s -o %t.undecoded.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.undecoded.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=UNDECODED,FAIL %s

// A back-edge that reloads the target pair through an unprovable definition
// makes the reconverged call Unknown; a stale finite result from the first
// visit must not survive.
// RUN: sed 's/^\.set reconverge_reload, 0$/.set reconverge_reload, 1/' \
// RUN:   %s > %t.reconverge.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.reconverge.s -o %t.reconverge.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.reconverge.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=RECONVERGE,FAIL %s

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// DISASM-LABEL: <reusable_pc_targets>:
// DISASM: s_swap_pc_i64 s[6:7]
// DISASM-NEXT: s_swap_pc_i64 s[6:7]
// DISASM-NEXT: s_branch
// COM: The validated roll-up router promotes each far source into an
// COM: SGPR-backed s_call_i64 hop rather than the older shared get-PC chain.
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_call_i64
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
.set overlap_delta, 0
.set clobber_target, 0
.set clobber_bootstrap, 0
.set tail_escape, 0
.set indirect_escape, 0
.set external_entry, 0
.set undecoded_gap, 0
.set reconverge_reload, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.if external_entry
.type external_materialization_entry,@function
external_materialization_entry:
  // This independent function enters after get-PC and delta creation.
  s_branch .Lfirst_add_low
  s_endpgm
.size external_materialization_entry, .-external_materialization_entry
.endif

.local callee_bootstrap
.type callee_bootstrap,@function
callee_bootstrap:
  s_mov_b32 s8, 3
.if clobber_bootstrap
  s_mov_b32 s10, 0
.endif
  s_set_pc_i64 s[12:13]
.size callee_bootstrap, .-callee_bootstrap

.globl reusable_pc_targets
.p2align 8
.type reusable_pc_targets,@function
reusable_pc_targets:
  // A straight-line first call is resolved by the one-shot matcher. Its
  // proven target remains reusable after the audited callee returns.
.Lbootstrap_getpc:
  s_get_pc_i64 s[10:11]
  s_add_nc_u64 s[10:11], s[10:11], callee_bootstrap-(.Lbootstrap_getpc+4)
  s_swap_pc_i64 s[12:13], s[10:11]
  s_swap_pc_i64 s[12:13], s[10:11]
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
.if overlap_delta
  // The temporary aliases the PC pair and destroys its low half.
  s_add_co_i32 s2, callee_first-(.Lselect_first+4)-4, 4
.Lfirst_add_low:
  s_add_co_u32 s2, s2, s2
.else
  s_add_co_i32 s4, callee_first-(.Lselect_first+4)-4, 4
.Lfirst_add_low:
  s_add_co_u32 s2, s2, s4
.endif
  s_add_co_ci_u32 s3, s3, 0
  s_branch .Lselected
.Lselect_second:
  s_get_pc_i64 s[2:3]
.if overlap_delta
  s_add_co_i32 s2, callee_second-(.Lselect_second+4)-4, 4
  s_add_co_u32 s2, s2, s2
.else
  s_add_co_i32 s4, callee_second-(.Lselect_second+4)-4, 4
  s_add_co_u32 s2, s2, s4
.endif
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
.if undecoded_gap
  // Keep the first call's return continuation decoded, then place an unknown
  // word before the second call. The solver must not carry the finite target
  // set across it.
  s_nop 0
  .long 0xffffffff
.endif
.if reconverge_reload
  // One path reaches the second reused call with the proven pair; a sibling
  // path reloads it through an unprovable definition. They reconverge on the
  // second call, which must therefore be Unknown -- not the finite result the
  // proven path alone recorded on its earlier visit.
  s_cmp_eq_u32 s0, 5
  s_cbranch_scc1 .Lreload_join
  s_branch .Lreused_second
.Lreload_join:
  s_mov_b32 s2, 0
  s_mov_b32 s3, 0
.Lreused_second:
.endif
  // Production kernels reuse this selected pair for multiple calls.
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
.if clobber_target
  s_mov_b32 s2, 0
.endif
.if tail_escape
  s_branch clobber_helper
.endif
.if indirect_escape
  s_set_pc_i64 s[10:11]
.endif
  s_set_pc_i64 s[6:7]
.size callee_first, .-callee_first

.local callee_second
.type callee_second,@function
callee_second:
  s_mov_b32 s8, 2
.if clobber_target
  s_mov_b32 s2, 0
.endif
.if tail_escape
  s_branch clobber_helper
.endif
.if indirect_escape
  s_set_pc_i64 s[10:11]
.endif
  s_set_pc_i64 s[6:7]
.size callee_second, .-callee_second

.if tail_escape
.local clobber_helper
.type clobber_helper,@function
clobber_helper:
  s_mov_b32 s2, 0
  s_set_pc_i64 s[6:7]
.size clobber_helper, .-clobber_helper
.endif

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
  .amdhsa_next_free_sgpr 14
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: reusable_pc_targets
      .symbol: reusable_pc_targets.kd
      .sgpr_count: 14
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
