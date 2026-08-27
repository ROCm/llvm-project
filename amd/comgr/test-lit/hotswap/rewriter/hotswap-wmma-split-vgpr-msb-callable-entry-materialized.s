// A block that holds a mandatory WMMA is reached two ways inside one function:
// by fallthrough from an s_set_vgpr_msb prefix (a concrete mode) and by a
// same-function PC-materialized s_swap_pc_i64 call. A call target is a fresh
// ABI (mode-0) entry, so it must be seeded like the function start. Because
// collectDirectBranchTargets resolves the canonical materialized call there is
// no unresolved-target bail, but evaluateDirectControlFlowTarget cannot resolve
// a register-target call. The VGPR-MSB analysis must therefore seed the call
// target over the same control-flow surface it uses for interior entries; the
// mode-0 contribution then conflicts with the fallthrough mode and the join is
// unprovable, so the mandatory split fails closed. Without seeding the
// materialized call target, the analysis drops the mode-0 edge, converges on
// the fallthrough mode, and patches the WMMA under a mode that is not provable.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck %s
// RUN: test ! -e %t.out.elf
// The materialized call must resolve (no global unresolved-target bail), so the
// failure comes from the unprovable join at the shared block, not that bail.
// CHECK: hotswap: resolved PC-materialized call
// CHECK-NOT: hotswap: unresolved call target
// CHECK-LABEL: WMMA split: cannot determine VGPR-MSB mode
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// A canonical local return helper at .text+0. MC lowering turns the compiler
// return pseudo into plain s_set_pc_i64, whose destination is proven from the
// incoming call's link register. It is the first call's target below.
.type local_return_helper,@function
local_return_helper:
  s_nop 0
.Llocal_return_epilogue:
  s_set_pc_i64 s[0:1]
  s_branch .Llocal_return_epilogue
.Llocal_return_helper_end:
.size local_return_helper, .Llocal_return_helper_end-local_return_helper

.globl test_callable_entry
.p2align 8
.type test_callable_entry,@function
test_callable_entry:
  // First canonical PC-materialized call (to local_return_helper at .text+0).
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], -260
  s_swap_pc_i64 s[0:1], s[4:5]

  // Second canonical PC-materialized call whose target is the shared block
  // below (.Lshared), making it a mode-0 callable entry. The displacement is a
  // constant that lands on .Lshared given the instruction sizes here.
  s_get_pc_i64 s[2:3]
  s_add_nc_u64 s[2:3], s[2:3], 20
  s_swap_pc_i64 s[0:1], s[2:3]
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2

  // Fallthrough path establishes a concrete VGPR-MSB mode before the shared
  // block. Its join with the call's mode-0 entry is unprovable, so the
  // mandatory WMMA split at .Lshared must fail closed.
  s_set_vgpr_msb 0x60
.Lshared:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_callable_entry_end:
.size test_callable_entry, .Ltest_callable_entry_end-test_callable_entry

// Padding after s_endpgm, outside the function, so it stays safe.
.fill 64, 1, 0

// Push the appended trampoline pool beyond s_branch's signed 16-bit dword
// range so far-site handling matches the production materialized-call shape.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_callable_entry
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 6
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_callable_entry
      .symbol: test_callable_entry.kd
      .sgpr_count: 6
      .vgpr_count: 40
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
