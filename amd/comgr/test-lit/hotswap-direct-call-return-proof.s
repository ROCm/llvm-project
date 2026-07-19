// COM: A nonstandard SGPR pair is a return address only when an exact direct
// COM: call supplies it on every path. Partial clobbers, alternate branch or
// COM: materialized ingress, opaque control flow, and an unresolved ABI-link
// COM: call turn the set-PC back into an arbitrary transfer.

// RUN: %clang -x assembler-with-cpp -DCASE=0 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.good.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.good.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.good.out.elf 2>&1 | %FileCheck --check-prefix=GOOD-API %s
// GOOD-API: recognized proven direct-call return
// GOOD-API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.good.out.elf | %FileCheck --check-prefix=GOOD %s
// RUN: hotswap-rewrite %t.good.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.good.out2.elf
// RUN: cmp %t.good.out.elf %t.good.out2.elf

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.clobber.elf
// RUN: hotswap-rewrite %t.clobber.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.clobber.out.elf | %FileCheck --check-prefix=BAD-API %s
// RUN: %llvm-objdump -d %t.clobber.out.elf | %FileCheck --check-prefix=BAD %s

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.ingress.elf
// RUN: hotswap-rewrite %t.ingress.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.ingress.out.elf | %FileCheck --check-prefix=BAD-API %s
// RUN: %llvm-objdump -d %t.ingress.out.elf | %FileCheck --check-prefix=BAD %s

// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.rfe.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.rfe.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.rfe.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=REJECT-API %s
// RUN: %llvm-objdump -d %t.rfe.out.elf | %FileCheck --check-prefix=BAD %s

// RUN: %clang -x assembler-with-cpp -DCASE=4 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.unknown.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.unknown.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.unknown.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=REJECT-API %s
// RUN: %llvm-objdump -d %t.unknown.out.elf | %FileCheck --check-prefix=BAD %s

// RUN: %clang -x assembler-with-cpp -DCASE=5 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.trap.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.trap.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.trap.out.elf 2>&1 | \
// RUN:   %FileCheck --check-prefix=GOOD-API %s
// RUN: %llvm-objdump -d %t.trap.out.elf | %FileCheck --check-prefix=GOOD %s

// RUN: %clang -x assembler-with-cpp -DCASE=6 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.materialized.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.materialized.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.materialized.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=REJECT-API %s
// RUN: %llvm-objdump -d %t.materialized.out.elf | %FileCheck --check-prefix=BAD %s

// RUN: %clang -x assembler-with-cpp -DCASE=7 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.global.elf
// RUN: hotswap-rewrite %t.global.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.global.out.elf | %FileCheck --check-prefix=BAD-API %s
// RUN: %llvm-objdump -d %t.global.out.elf | %FileCheck --check-prefix=BAD %s

// RUN: %clang -x assembler-with-cpp -DCASE=8 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.weak.elf
// RUN: hotswap-rewrite %t.weak.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.weak.out.elf | %FileCheck --check-prefix=BAD-API %s
// RUN: %llvm-objdump -d %t.weak.out.elf | %FileCheck --check-prefix=BAD %s

// RUN: %clang -x assembler-with-cpp -DCASE=9 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.standard-link.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.standard-link.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.standard-link.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=REJECT-API %s
// RUN: %llvm-objdump -d %t.standard-link.out.elf \
// RUN:   | %FileCheck --check-prefix=BAD %s

// BAD-API: RESULT: SUCCESS
// REJECT-API-NOT: recognized proven direct-call return
// REJECT-API: RESULT: SUCCESS

// GOOD-LABEL: <direct_return_observer>:
// GOOD-NEXT: global_wb
// GOOD-NEXT: v_nop
// GOOD-NEXT: global_load_b32 v0
// GOOD-NEXT: s_wait_loadcnt 0x0
// GOOD-NEXT: s_nop 0
// GOOD-NEXT: global_load_b32 v1

// BAD-LABEL: <direct_return_observer>:
// BAD-NEXT: global_wb
// BAD-NEXT: v_nop
// BAD-NEXT: global_load_b32 v0
// BAD-NEXT: s_wait_loadcnt 0x0
// BAD-NEXT: s_nop 0
// BAD-NEXT: global_load_b32 v1

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl direct_return_caller
.p2align 8
.type direct_return_caller,@function
direct_return_caller:
  global_wb scope:SCOPE_CU
  v_nop
#if CASE == 2
  s_cmp_eq_u32 s2, 0
  s_cbranch_scc1 direct_return_helper
#endif
#if CASE == 3
  s_rfe_i64 s[4:5]
#endif
#if CASE == 4
  .long 0xffffffff
#endif
#if CASE == 5
  s_trap 0
#endif
#if CASE == 6
.Lmaterialized_getpc:
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], \
      direct_return_helper-(.Lmaterialized_getpc+4)
  s_set_pc_i64 s[4:5]
#endif
#if CASE == 9
  s_swap_pc_i64 s[30:31], s[4:5]
#endif
  s_call_i64 s[0:1], direct_return_helper
  s_endpgm
.Ldirect_return_caller_end:
.size direct_return_caller, .Ldirect_return_caller_end-direct_return_caller

.local direct_return_helper
.type direct_return_helper,@function
direct_return_helper:
#if CASE == 1
  s_mov_b32 s0, 0
#endif
  s_set_pc_i64 s[0:1]
.Ldirect_return_helper_end:
.size direct_return_helper, .Ldirect_return_helper_end-direct_return_helper

#if CASE == 7
.globl direct_return_helper_external
.type direct_return_helper_external,@function
.set direct_return_helper_external, direct_return_helper
.size direct_return_helper_external, \
    .Ldirect_return_helper_end-direct_return_helper
#elif CASE == 8
.weak direct_return_helper_external
.type direct_return_helper_external,@function
.set direct_return_helper_external, direct_return_helper
.size direct_return_helper_external, \
    .Ldirect_return_helper_end-direct_return_helper
#endif

.globl direct_return_observer
.p2align 8
.type direct_return_observer,@function
direct_return_observer:
  global_wb scope:SCOPE_CU
  v_nop
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0
  s_clause 0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0
  s_endpgm
.Ldirect_return_observer_end:
.size direct_return_observer, .Ldirect_return_observer_end-direct_return_observer

.rodata
.p2align 8
.amdhsa_kernel direct_return_caller
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 3
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel direct_return_observer
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: direct_return_caller
      .symbol: direct_return_caller.kd
      .sgpr_count: 3
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: direct_return_observer
      .symbol: direct_return_observer.kd
      .sgpr_count: 0
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
