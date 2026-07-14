// COM: A recognized ADDTID instruction is an A0-required rewrite. If its
// COM: trampoline cannot be placed, fail closed without producing an output
// COM: object instead of reporting success with the incompatible instruction.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck %s
// RUN: test ! -e %t.out.elf
// CHECK: hotswap: error: safe far return: no aligned block of 2 safe SGPRs fits below s106
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_addtid_required_failure
.p2align 8
.type test_addtid_required_failure,@function
test_addtid_required_failure:
  s_cbranch_scc1 .Lresume
  ds_load_addtid_b32 v0 offset:128
.Lresume:
  s_endpgm
.Ltest_addtid_required_failure_end:
.size test_addtid_required_failure, .Ltest_addtid_required_failure_end-test_addtid_required_failure

// Non-NOP far filler leaves no local replacement sled or forward gateway.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_addtid_required_failure
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_addtid_required_failure
      .symbol: test_addtid_required_failure.kd
      .sgpr_count: 106
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
