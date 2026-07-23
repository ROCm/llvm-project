// COM: A0 tensor mask clearing fails closed when a construction region has no
// COM: low16-preserving s_and mask-set (the descriptor is restarted and ORed
// COM: but never normalized with a maskable s_and). Without an identifiable
// COM: mask-set the pass cannot force workgroup_mask to zero, so it rejects.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: error: tensor_load_to_lds at 0x{{[0-9A-F]+}}: descriptor construction region at 0x{{[0-9A-F]+}} has no low16-preserving s_and mask-set
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_no_maskset
.p2align 8
.type test_tensor_mask_no_maskset,@function
test_tensor_mask_no_maskset:
  s_mov_b32 s4, 0
  s_or_b32 s4, s4, s5
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.Ltest_tensor_mask_no_maskset_end:
.size test_tensor_mask_no_maskset, .Ltest_tensor_mask_no_maskset_end-test_tensor_mask_no_maskset

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_no_maskset
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_no_maskset
      .symbol: test_tensor_mask_no_maskset.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
