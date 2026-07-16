; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=ds_permute_b32_wave32_rebase_kernel \
; RUN:   | %FileCheck %s

; Rebase wave32 ds_permute_b32 (forward/PUSH) source-lane addressing
; onto the target wave64 layout. Mirror of ds_bpermute_b32_wave32_rebase.s:
; the push selector is source-wave-local, so it is masked into [0, 128)
; and OR'd with the current source-wave byte base before the wave64 intrinsic.
; CHECK-LABEL: define amdgpu_kernel void @ds_permute_b32_wave32_rebase_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_permute_b32_wave32_rebase_kernel
	.p2align	8
	.type	ds_permute_b32_wave32_rebase_kernel,@function
ds_permute_b32_wave32_rebase_kernel:
	v_mov_b32_e32 v1, v0
	v_and_b32_e32 v2, 31, v0
	v_lshlrev_b32_e32 v2, 2, v2
; CHECK: %perm_local_addr{{[0-9]*}} = and i32 %{{[^,]+}}, 127
; CHECK: %perm_srcwave_lane_base{{[0-9]*}} = and i32 %{{[^,]+}}, -32
; CHECK: %perm_srcwave_byte_base{{[0-9]*}} = shl i32 %perm_srcwave_lane_base{{[0-9]*}}, 2
; CHECK: %perm_srcwave_addr{{[0-9]*}} = or i32 %perm_local_addr{{[0-9]*}}, %perm_srcwave_byte_base{{[0-9]*}}
; CHECK: %perm = call i32 @llvm.amdgcn.ds.permute(i32 %perm_srcwave_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: declare {{.*}}i32 @llvm.amdgcn.ds.permute(i32, i32)
	ds_permute_b32 v3, v2, v1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_permute_b32_wave32_rebase_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           ds_permute_b32_wave32_rebase_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         ds_permute_b32_wave32_rebase_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
