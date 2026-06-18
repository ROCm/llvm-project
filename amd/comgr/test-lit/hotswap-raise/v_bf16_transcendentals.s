; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_bf16_transcendentals_kernel 2>/dev/null | %FileCheck %s
;
; bf16 unary transcendentals lift through bf16->f32 fpext, an f32 intrinsic
; (or __ocml_tanh_f32), and an fptrunc merged back into the dst half-register.

; CHECK-LABEL: define amdgpu_kernel void @v_bf16_transcendentals_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_bf16_transcendentals_kernel
	.p2align	8
	.type	v_bf16_transcendentals_kernel,@function
v_bf16_transcendentals_kernel:
	; Every op shares the bf16->f32 fpext / f32->bf16 fptrunc wrapping.
	; CHECK-DAG: fpext bfloat {{.*}} to float
	; CHECK-DAG: fptrunc float {{.*}} to bfloat
	; CHECK-DAG: call float @llvm.amdgcn.cos.f32(float
	v_cos_bf16  v0.l, v0.l
	; CHECK-DAG: call float @llvm.amdgcn.exp2.f32(float
	v_exp_bf16  v1.l, v1.l
	; CHECK-DAG: call float @llvm.amdgcn.log.f32(float
	v_log_bf16  v2.l, v2.l
	; CHECK-DAG: call float @llvm.amdgcn.rcp.f32(float
	v_rcp_bf16  v3.l, v3.l
	; CHECK-DAG: call float @llvm.amdgcn.rsq.f32(float
	v_rsq_bf16  v4.l, v4.l
	; CHECK-DAG: call float @llvm.amdgcn.sin.f32(float
	v_sin_bf16  v5.l, v5.l
	; CHECK-DAG: call float @llvm.amdgcn.sqrt.f32(float
	v_sqrt_bf16 v6.l, v6.l
	; v_tanh_bf16 has no f32 intrinsic; it lifts through inlined __ocml_tanh_f32.
	; CHECK-DAG: __ocml_tanh_f32.exit:
	v_tanh_bf16 v7.l, v7.l
	; op_sel:[1,1]: read src high half, rcp via f32, merge into dst high half.
	; CHECK:      lshr i32 {{.*}}, 16
	; CHECK:      call float @llvm.amdgcn.rcp.f32(
	; CHECK:      %[[RES:[0-9]+]] = zext i16 {{.*}} to i32
	; CHECK:      %[[LO:[0-9]+]] = and i32 {{.*}}, 65535
	; CHECK:      %[[HI:[0-9]+]] = shl i32 %[[RES]], 16
	; CHECK-NEXT: %bf16_merge_hi = or i32 %[[LO]], %[[HI]]
	v_rcp_bf16_e64  v8.h, v8.h  op_sel:[1,1]
	; Subreg-name half-select (v9.h), no op_sel: its own read -> rcp -> merge.
	; CHECK:      lshr i32 {{.*}}, 16
	; CHECK:      call float @llvm.amdgcn.rcp.f32(
	; CHECK:      %[[RES9:[0-9]+]] = zext i16 {{.*}} to i32
	; CHECK:      %[[LO9:[0-9]+]] = and i32 {{.*}}, 65535
	; CHECK:      %[[HI9:[0-9]+]] = shl i32 %[[RES9]], 16
	; CHECK-NEXT: bf16_merge_hi{{[0-9]+}} = or i32 %[[LO9]], %[[HI9]]
	v_rcp_bf16  v9.h, v9.h
	s_endpgm

; CHECK: declare {{.*}}float @llvm.amdgcn.rcp.f32(float)
; CHECK-NOT: call {{.*}}@__ocml_tanh_f32
; CHECK-NOT: declare {{.*}}@__ocml_tanh_f32

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_bf16_transcendentals_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 10
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
    .name:           v_bf16_transcendentals_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_bf16_transcendentals_kernel.kd
    .vgpr_count:     10
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
