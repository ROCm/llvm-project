// COM: A0 tensor where both strategies genuinely fail, so hotswap must reject
// COM: rather than emit an object that can still hang A0. The descriptor is a
// COM: bare operand (no construction region -> definition clear not applicable)
// COM: and its SGPR is live after the tensor while the kernel's SGPR budget is
// COM: saturated to s106, so the at-site fallback has no scratch register.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: error: tensor_load_to_lds: no scratch SGPR available
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_failclosed
.p2align 8
.type test_tensor_failclosed,@function
test_tensor_failclosed:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  s_endpgm
.Ltest_tensor_failclosed_end:
.size test_tensor_failclosed, .Ltest_tensor_failclosed_end-test_tensor_failclosed

.rodata
.p2align 8
.amdhsa_kernel test_tensor_failclosed
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_failclosed
      .symbol: test_tensor_failclosed.kd
      .sgpr_count: 106
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
