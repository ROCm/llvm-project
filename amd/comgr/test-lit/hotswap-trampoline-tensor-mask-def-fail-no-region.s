// COM: A0 tensor mask clearing fails closed when the descriptor arrives in an
// COM: SGPR with no local construction region (bare operand). The pass cannot
// COM: prove where the multicast mask is set, so it must reject rather than
// COM: emit an object that may still hang A0.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: error: tensor_load_to_lds at 0x{{[0-9A-F]+}}: no descriptor construction region found for s4
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_no_region
.p2align 8
.type test_tensor_mask_no_region,@function
test_tensor_mask_no_region:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.Ltest_tensor_mask_no_region_end:
.size test_tensor_mask_no_region, .Ltest_tensor_mask_no_region_end-test_tensor_mask_no_region

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_no_region
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_no_region
      .symbol: test_tensor_mask_no_region.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
