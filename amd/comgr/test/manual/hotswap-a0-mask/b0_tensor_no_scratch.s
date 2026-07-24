// Live tensor_load_to_lds descriptor SGPRs require scratch save/restore around
// the A0 D# Group 1 mask clear. With every user-addressable SGPR consumed,
// hotswap must return ERROR rather than dropping the mandatory mask patch.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.text
.globl b0_tensor_no_scratch
.p2align 8
.type b0_tensor_no_scratch,@function
b0_tensor_no_scratch:
  s_branch .Ldone

.Lcold:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4

.Ldone:
  s_endpgm
.Lend:
.size b0_tensor_no_scratch, .Lend-b0_tensor_no_scratch

.rodata
.p2align 8
.amdhsa_kernel b0_tensor_no_scratch
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 1
    - 2
  amdhsa.kernels:
    - .gfx1250_revision: B0
      .name: b0_tensor_no_scratch
      .symbol: b0_tensor_no_scratch.kd
      .sgpr_count: 106
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .kernarg_segment_align: 8
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .max_flat_workgroup_size: 64
      .wavefront_size: 32
  amdhsa.target: amdgcn-amd-amdhsa--gfx1250
.end_amdgpu_metadata
