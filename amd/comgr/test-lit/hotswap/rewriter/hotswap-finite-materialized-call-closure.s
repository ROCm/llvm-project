// COM: An exact PC-materialized singleton call enters a defined local helper.
// COM: The helper uses s[30:31] as scratch only after saving the incoming link
// COM: in callee-saved VGPR lanes, then restores it before returning. Prove the
// COM: call, canonical return, and call continuation together without falling
// COM: back to an object-wide unknown indirect-entry assumption.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: accepted exact materialized-call/canonical-return closure for 1 register call(s)
// LOG-NOT: hotswap: unresolved call target
// LOG-NOT: hotswap: unresolved control-flow target disables
// LOG: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl exact_closure_kernel
.type exact_closure_kernel,@function
exact_closure_kernel:
  s_get_pc_i64 s[0:1]
  // PC after get-PC is 4; exact_closure_helper begins at 16.
  s_add_nc_u64 s[0:1], s[0:1], 12
  s_swap_pc_i64 s[30:31], s[0:1]
  s_endpgm
.size exact_closure_kernel, .-exact_closure_kernel

.local exact_closure_helper
.type exact_closure_helper,@function
exact_closure_helper:
  v_writelane_b32 v40, s30, 0
  v_writelane_b32 v41, s31, 1
  s_mov_b32 s30, 0
  s_mov_b32 s31, 0
  v_readlane_b32 s30, v40, 0
  v_readlane_b32 s31, v41, 1
  s_set_pc_i64 s[30:31]
  s_endpgm
.size exact_closure_helper, .-exact_closure_helper

.rodata
.p2align 8
.amdhsa_kernel exact_closure_kernel
  .amdhsa_next_free_vgpr 42
  .amdhsa_next_free_sgpr 42
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: exact_closure_kernel
      .symbol: exact_closure_kernel.kd
      .sgpr_count: 42
      .vgpr_count: 42
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
