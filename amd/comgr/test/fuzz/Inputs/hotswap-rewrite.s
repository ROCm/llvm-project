.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.text
.globl hotswap_fuzz_seed
.p2align 8
.type hotswap_fuzz_seed,@function
hotswap_fuzz_seed:
  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
  v_nop
  s_branch .Ldone
  ds_load_2addr_stride64_b32 v[4:5], v2 offset0:1 offset1:3
  ds_load_2addr_stride64_b32 v[6:7], v3 offset0:2 offset1:4
.Ldone:
  s_endpgm
.Lend:
.size hotswap_fuzz_seed, .Lend-hotswap_fuzz_seed

.rodata
.p2align 8
.amdhsa_kernel hotswap_fuzz_seed
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 2
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .gfx1250_revision: B0
      .name: hotswap_fuzz_seed
      .symbol: hotswap_fuzz_seed.kd
      .sgpr_count: 2
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .kernarg_segment_align: 8
      .group_segment_fixed_size: 256
      .private_segment_fixed_size: 0
      .max_flat_workgroup_size: 64
      .wavefront_size: 32
.end_amdgpu_metadata
