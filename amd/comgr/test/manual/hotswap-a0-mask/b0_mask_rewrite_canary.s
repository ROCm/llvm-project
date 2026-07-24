// Dispatchable B0 canary for gfx1250 A0 hotswap mask validation.
//
// The entry path stores a sentinel to the output pointer passed as kernarg 0.
// The B0-only instructions live in a cold block so the same code object can be
// safely loaded and dispatched on A0 after rewriting, while still forcing
// hotswap to patch every mask case in the code object.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.text
.globl b0_mask_rewrite_canary
.p2align 8
.type b0_mask_rewrite_canary,@function
b0_mask_rewrite_canary:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_wait_kmcnt 0x0
  s_mov_b32 s14, 0xb0a00001
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, s14
  global_store_b32 v0, v1, s[2:3] scope:SCOPE_SYS
  s_wait_storecnt 0x0
  s_branch .Ldone

.Lcold:
  s_mov_b32 m0, -1

  // Tensor D# Group 1 is operand 1, so the A0 rewrite must clear s4[15:0].
  // s4 is read after the tensor op to force save/restore through scratch.
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s12, s4

  // SADDR cluster loads remain cluster loads and must be wrapped with M0
  // save, low-half clear, and restore.
  cluster_load_b64 v[0:1], v2, s[4:5]
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b32 v3, v4, s[6:7]
  s_wait_loadcnt 0x0

  // Off-form cluster loads should still demote in place to global_load.
  cluster_load_b32 v5, v[6:7], off
  s_wait_loadcnt 0x0

.Ldone:
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Lend:
.size b0_mask_rewrite_canary, .Lend-b0_mask_rewrite_canary

.rodata
.p2align 8
.amdhsa_kernel b0_mask_rewrite_canary
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 1
    - 2
  amdhsa.kernels:
    - .args:
        - .address_space: global
          .offset: 0
          .size: 8
          .value_kind: global_buffer
      .gfx1250_revision: B0
      .name: b0_mask_rewrite_canary
      .symbol: b0_mask_rewrite_canary.kd
      .sgpr_count: 24
      .vgpr_count: 8
      .kernarg_segment_size: 8
      .kernarg_segment_align: 8
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .max_flat_workgroup_size: 64
      .wavefront_size: 32
  amdhsa.target: amdgcn-amd-amdhsa--gfx1250
.end_amdgpu_metadata
