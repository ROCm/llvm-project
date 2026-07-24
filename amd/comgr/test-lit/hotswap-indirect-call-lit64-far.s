// COM: A compiler-emitted lit64 s_add_nc_u64 materializes one finite call
// COM: target. Absolute MC expressions must be accepted like immediates.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: resolved PC-materialized call at 0x14 to .text+0x20
// LOG-NOT: unresolved control-flow target disables
// LOG: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_indirect_call_lit64_far
.p2align 8
.type test_indirect_call_lit64_far,@function
test_indirect_call_lit64_far:
  // s_get_pc_i64 captures .text+4. The 64-bit literal reaches .text+32.
  s_get_pc_i64 s[2:3]
  s_add_nc_u64 s[2:3], s[2:3], lit64(28)
  s_mov_b64 s[4:5], s[0:1]
  s_swap_pc_i64 s[0:1], s[2:3]
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lindirect_target:
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  s_wait_dscnt 0x0
  s_endpgm
.size test_indirect_call_lit64_far, .-test_indirect_call_lit64_far

.fill 64, 1, 0
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_indirect_call_lit64_far
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 6
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_indirect_call_lit64_far
      .symbol: test_indirect_call_lit64_far.kd
      .sgpr_count: 6
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
