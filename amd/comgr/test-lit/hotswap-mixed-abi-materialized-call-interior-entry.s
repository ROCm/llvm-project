// COM: An unrelated opaque ABI call must not suppress alternate-entry
// COM: validation for an exact PC-materialized call in the same object. A
// COM: direct call enters the exact call at its add, bypassing s_get_pc_i64.
// COM: The exact call therefore remains unresolved, disables .text gateways,
// COM: and makes the required far DS2 patch fail closed.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu --no-undefined -shared \
// RUN:   -plugin-opt=mcpu=gfx1250 %t.o -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: exact materialized-call/canonical-return closure rejected:
// LOG-SAME: alternate entry inside call materialization ending at 0x
// LOG: hotswap: unresolved call target
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.local mixed_abi_helper
.type mixed_abi_helper,@function
mixed_abi_helper:
  s_set_pc_i64 s[30:31]
  s_endpgm
.size mixed_abi_helper, .-mixed_abi_helper

.globl test_mixed_abi_materialized_call_interior_entry
.p2align 8
.type test_mixed_abi_materialized_call_interior_entry,@function
test_mixed_abi_materialized_call_interior_entry:
  // This call's target is intentionally opaque but has the exact ABI shape.
  s_swap_pc_i64 s[30:31], s[8:9]

  // Enter the exact materialization after its defining get-PC.
  s_call_i64 s[4:5], .Lexact_add

  // This required patch needs the external zero-filled gateway below because
  // the appended trampoline pool is outside the signed s_branch range.
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2

  s_get_pc_i64 s[0:1]
.Lexact_pc:
.Lexact_add:
  s_add_nc_u64 s[0:1], s[0:1], mixed_abi_helper-.Lexact_pc
  s_swap_pc_i64 s[30:31], s[0:1]
  s_endpgm
.size test_mixed_abi_materialized_call_interior_entry, .-test_mixed_abi_materialized_call_interior_entry

// This padding is usable as a gateway only when every original control-flow
// destination is known.
.fill 64, 1, 0

// Keep the appended trampoline pool beyond one signed s_branch span.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_mixed_abi_materialized_call_interior_entry
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_mixed_abi_materialized_call_interior_entry
      .symbol: test_mixed_abi_materialized_call_interior_entry.kd
      .sgpr_count: 32
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
