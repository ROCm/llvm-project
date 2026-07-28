; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=scaled_modrep_wgsize_virtualize_kernel 2>&1 \
; RUN:   | %FileCheck %s

; The in-kernel hidden_group_size_x read is halved by the scale factor.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	scaled_modrep_wgsize_virtualize_kernel
	.type	scaled_modrep_wgsize_virtualize_kernel,@function
; CHECK: define amdgpu_kernel void @scaled_modrep_wgsize_virtualize_kernel(
scaled_modrep_wgsize_virtualize_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
; CHECK: %source_hidden_wg_size_0{{.*}} = lshr i32 {{.+}}, 1
	s_load_b32 s5, s[0:1], 0x8
	v_bfe_u32 v2, v0, 10, 10
	s_wait_kmcnt 0x0
	v_cmp_lt_u32_e64 s4, v2, 16
	v_cndmask_b32_e64 v0, -1, v0, s4
	v_add_nc_u32_e32 v0, s5, v0
	v_mov_b32_e32 v1, v2
	global_store_b32 v1, v0, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scaled_modrep_wgsize_virtualize_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
      - .offset:       8
        .size:         2
        .value_kind:   hidden_group_size_x
      - .offset:       10
        .size:         2
        .value_kind:   hidden_group_size_y
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     12
    .max_flat_workgroup_size:  512
    .name:                     scaled_modrep_wgsize_virtualize_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         scaled_modrep_wgsize_virtualize_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
