; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=ds_bpermute_b32_wave32_rebase_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Rebase wave32 ds_bpermute_b32 source-lane addressing onto the target wave64 layout.
; CHECK-LABEL: define amdgpu_kernel void @ds_bpermute_b32_wave32_rebase_kernel(
; CHECK: %bperm_local_addr{{[0-9]*}} = and i32 %{{[^,]+}}, 127
; CHECK: %bperm_srcwave_lane_base{{[0-9]*}} = and i32 %{{[^,]+}}, -32
; CHECK: %bperm_srcwave_byte_base{{[0-9]*}} = shl i32 %bperm_srcwave_lane_base{{[0-9]*}}, 2
; CHECK: %bperm_srcwave_addr{{[0-9]*}} = or i32 %bperm_local_addr{{[0-9]*}}, %bperm_srcwave_byte_base{{[0-9]*}}
; CHECK: %bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %bperm_srcwave_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_bpermute_b32_wave32_rebase_kernel
	.p2align	8
	.type	ds_bpermute_b32_wave32_rebase_kernel,@function
ds_bpermute_b32_wave32_rebase_kernel:
	v_mov_b32_e32 v1, v0
	v_and_b32_e32 v2, 31, v0
	v_lshlrev_b32_e32 v2, 2, v2
	ds_bpermute_b32 v3, v2, v1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_bpermute_b32_wave32_rebase_kernel
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
    .name:           ds_bpermute_b32_wave32_rebase_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         ds_bpermute_b32_wave32_rebase_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
