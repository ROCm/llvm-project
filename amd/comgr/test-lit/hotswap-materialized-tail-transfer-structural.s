// COM: A compiler may schedule arbitrary straight-line instructions between a
// COM: materialized PC-relative address and its terminal s_set_pc_i64. Prove
// COM: this transfer structurally: one unchanged SGPR pair, an absolute delta,
// COM: a contiguous control-flow-free gap with no pair access or interior
// COM: entry, an owning-function tail transfer, and another function's entry
// COM: as the destination. Do not recognize one mnemonic schedule, one SGPR
// COM: pair, or a bounded instruction count.
// COM:
// COM: The far DS2 rewrite below depends on the sole donated gateway. Thus a
// COM: positive RESULT: SUCCESS also proves that the transfer was recognized;
// COM: an unknown indirect target globally disables the gateway and fails the
// COM: rewrite. Negative cases pin every part of the structural certificate.

// RUN: for CASE in 1 2 3; do \
// RUN:   %clang -x assembler-with-cpp -DCASE=$CASE -target amdgcn-amd-amdhsa \
// RUN:     -mcpu=gfx1250 -nostdlib %s -o %t.$CASE.elf && \
// RUN:   env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.$CASE.elf \
// RUN:     amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:     amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:     --output %t.$CASE.out.elf 2>&1 \
// RUN:     | %FileCheck --check-prefix=POSITIVE %s || exit 1; \
// RUN: done

// RUN: for CASE in 4 5 6 7 8 9 10 11; do \
// RUN:   %clang -x assembler-with-cpp -DCASE=$CASE -target amdgcn-amd-amdhsa \
// RUN:     -mcpu=gfx1250 -nostdlib %s -o %t.$CASE.elf && \
// RUN:   env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.$CASE.elf \
// RUN:     amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:     amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:     --expect-status ERROR 2>&1 \
// RUN:     | %FileCheck --check-prefix=NEGATIVE %s || exit 1; \
// RUN: done

// POSITIVE: hotswap: recognized materialized PC transfer
// POSITIVE: hotswap: assigned 1 SCC-neutral forward gateway(s)
// POSITIVE: RESULT: SUCCESS

// NEGATIVE-NOT: hotswap: recognized materialized PC transfer
// NEGATIVE: hotswap: incomplete control-flow targets disable NOP padding donation
// NEGATIVE: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl gateway_kernel
.type gateway_kernel,@function
gateway_kernel:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0
  s_endpgm
.size gateway_kernel, .-gateway_kernel

// This is the sole source-side gateway for the far patch above.
.rept 8
  s_nop 0
.endr

.local materialized_source
.type materialized_source,@function
materialized_source:
#if CASE == 7
  // Entering the otherwise safe gap through a direct branch invalidates the
  // sequence even though no control-flow instruction lies inside it.
  s_cbranch_scc1 .Ldirect_gap_entry
#endif

.Lgetpc:
#if CASE == 8
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], materialized_target-(.Lgetpc+4)
#elif CASE == 9
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], s[8:9]
#elif CASE == 1
  s_get_pc_i64 s[0:1]
  s_add_nc_u64 s[0:1], s[0:1], materialized_target-(.Lgetpc+4)
#elif CASE == 3
  s_get_pc_i64 s[6:7]
  s_add_nc_u64 s[6:7], s[6:7], materialized_target-(.Lgetpc+4)
#elif CASE == 11
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], \
      .Lmaterialized_target_interior-(.Lgetpc+4)
#else
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], materialized_target-(.Lgetpc+4)
#endif

#if CASE == 1
  // More than the former bounded look-back, all independent of s[0:1].
  .rept 10
    s_mov_b32 s8, s9
  .endr
#elif CASE == 2
  // A separated tail transfer may use any otherwise valid SGPR pair.
  s_nop 0
#elif CASE == 3
  // Safe scheduling is not defined by a fixed mnemonic order.
  s_mov_b32 s8, s9
  v_mov_b32_e32 v0, v1
  s_wait_dscnt 0
  s_cmp_eq_u32 s10, s11
  v_add_nc_u32_e32 v2, v3, v4
  s_nop 0
#elif CASE == 4
  // An explicit access to either half of the target pair is unsafe.
  s_mov_b32 s4, s8
#elif CASE == 5
  // The gap must be straight-line, even when the branch only skips a NOP.
  s_cbranch_scc1 .Lcontrol_gap_target
  s_nop 0
.Lcontrol_gap_target:
  s_nop 0
#elif CASE == 6
  // An emitted symbol is an addressable interior entry.
  .local materialized_gap_symbol
materialized_gap_symbol:
  s_nop 0
#elif CASE == 7
  s_nop 0
.Ldirect_gap_entry:
  s_nop 0
#elif CASE == 8
  // The terminal transfer names a different pair from get-PC and add.
  s_nop 0
#elif CASE == 9
  // A register delta is not a statically materialized destination.
  s_nop 0
#elif CASE == 10
  s_nop 0
#elif CASE == 11
  s_nop 0
#else
  .error "CASE must select a test body"
#endif

#if CASE == 1
  s_setpc_b64 s[0:1]
#elif CASE == 3
  s_setpc_b64 s[6:7]
#elif CASE == 8
  s_setpc_b64 s[6:7]
#else
  s_setpc_b64 s[4:5]
#endif

#if CASE == 10
  // A tail transfer must end its owning function.
  s_nop 0
  s_endpgm
#endif
.size materialized_source, .-materialized_source

.local materialized_target
.type materialized_target,@function
materialized_target:
#if CASE == 11
  s_nop 0
.Lmaterialized_target_interior:
#endif
  s_endpgm
.size materialized_target, .-materialized_target

// Put the appended trampoline beyond direct s_branch reach without creating
// another NOP donor.
.rept 40000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel gateway_kernel
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: gateway_kernel
      .symbol: gateway_kernel.kd
      .sgpr_count: 66
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
