; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_tanh_f16_kernel 2>&1 | %FileCheck %s --check-prefix=IR
; RUN: raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_tanh_f16_kernel 2>&1 | %FileCheck %s --check-prefix=SAME
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_tanh_f16_hi_kernel 2>&1 | %FileCheck %s --check-prefix=HI
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_tanh_f16_mods_kernel 2>&1 | %FileCheck %s --check-prefix=MODS
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_tanh_f16_dpp_kernel 2>&1 | %FileCheck %s --check-prefix=DPP
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_tanh_f16_omod_refuse_kernel 2>&1 | %FileCheck %s --check-prefix=MOD

; v_tanh_f16 cross-target expands via ocml (native amdgcn.tanh.f16 same-target); hi/mods/dpp lanes handled, clamp/omod refused.
; IR-LABEL: define amdgpu_kernel void @v_tanh_f16_kernel(
; IR-NOT: call {{.*}}@__ocml_tanh_f16
; IR: tanh_f16_merge_lo
; IR: ret void
; IR-NOT: declare {{.*}}@__ocml_tanh_f16
; SAME-LABEL: define amdgpu_kernel void @v_tanh_f16_kernel(
; SAME: call half @llvm.amdgcn.tanh.f16(
; SAME-NOT: __ocml_tanh_f16
; HI-LABEL: define amdgpu_kernel void @v_tanh_f16_hi_kernel(
; HI: lshr i32 {{.*}}, 16
; HI: tanh_f16_merge_hi
; HI-NOT: call {{.*}}@__ocml_tanh_f16
; HI: ret void
; HI-NOT: declare {{.*}}@__ocml_tanh_f16
; MODS-LABEL: define amdgpu_kernel void @v_tanh_f16_mods_kernel(
; MODS: call half @llvm.fabs.f16(
; MODS: fneg half
; MODS-NOT: call {{.*}}@__ocml_tanh_f16
; MODS: tanh_f16_merge_lo
; MODS: ret void
; MODS-NOT: declare {{.*}}@__ocml_tanh_f16
; DPP-LABEL: define amdgpu_kernel void @v_tanh_f16_dpp_kernel(
; DPP-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
; DPP-DAG: %cwd_dpp_selector = shl i32 %cwd_dpp_src_abs, 2
; DPP-DAG: %cwd_dpp_bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector, i32 %{{[^,]+}})
; DPP-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
; DPP-NOT: call {{.*}}@__ocml_tanh_f16
; DPP: tanh_f16_merge_lo
; DPP: declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)
; DPP-NOT: declare {{.*}}@__ocml_tanh_f16
; MOD-DAG: kernel 'v_tanh_f16_omod_refuse_kernel'
; MOD-DAG: V_TANH_F16 with non-default clamp/omod is not yet lifted
; MOD-DAG: output modifier semantics must not be silently dropped

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_tanh_f16_kernel
	.p2align	8
	.type	v_tanh_f16_kernel,@function
v_tanh_f16_kernel:
	v_tanh_f16 v0.l, v0.l
	s_endpgm

	.globl	v_tanh_f16_hi_kernel
	.p2align	8
	.type	v_tanh_f16_hi_kernel,@function
v_tanh_f16_hi_kernel:
	v_tanh_f16_e64 v0.h, v0.h
	s_endpgm

	.globl	v_tanh_f16_mods_kernel
	.p2align	8
	.type	v_tanh_f16_mods_kernel,@function
v_tanh_f16_mods_kernel:
	v_tanh_f16_e64 v0.l, -|v0.l|
	s_endpgm

	.globl	v_tanh_f16_dpp_kernel
	.p2align	8
	.type	v_tanh_f16_dpp_kernel,@function
v_tanh_f16_dpp_kernel:
	v_tanh_f16 v0.l, v0.l row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_endpgm

	.globl	v_tanh_f16_omod_refuse_kernel
	.p2align	8
	.type	v_tanh_f16_omod_refuse_kernel,@function
v_tanh_f16_omod_refuse_kernel:
	v_tanh_f16_e64 v0.l, v0.l mul:2
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_tanh_f16_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.p2align	6, 0x0
	.amdhsa_kernel v_tanh_f16_hi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.p2align	6, 0x0
	.amdhsa_kernel v_tanh_f16_omod_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_tanh_f16_mods_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_tanh_f16_dpp_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_tanh_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_tanh_f16_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_tanh_f16_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_tanh_f16_hi_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_tanh_f16_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_tanh_f16_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_tanh_f16_mods_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_tanh_f16_mods_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_tanh_f16_dpp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_tanh_f16_dpp_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
