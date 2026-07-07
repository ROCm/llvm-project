; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_tanh_f32_kernel 2>&1 | %FileCheck %s --check-prefix=IR
; RUN: raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_tanh_f32_kernel 2>&1 | %FileCheck %s --check-prefix=SAME
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_tanh_f32_omod_refuse_kernel 2>&1 | %FileCheck %s --check-prefix=MOD

; v_tanh_f32 cross-target expands via __ocml_tanh_f32 (native amdgcn.tanh.f32 same-target); clamp/omod refused.
; IR-LABEL: define amdgpu_kernel void @v_tanh_f32_kernel(
; IR: __ocml_tanh_f32.exit:
; IR-NOT: call {{.*}}@__ocml_tanh_f32
; IR: ret void
; IR-NOT: declare {{.*}}@__ocml_tanh_f32
; SAME-LABEL: define amdgpu_kernel void @v_tanh_f32_kernel(
; SAME: call float @llvm.amdgcn.tanh.f32(
; SAME-NOT: __ocml_tanh_f32
; MOD-DAG: kernel 'v_tanh_f32_omod_refuse_kernel'
; MOD-DAG: V_TANH_F32 with non-default clamp/omod is not yet lifted
; MOD-DAG: output modifier semantics must not be silently dropped

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_tanh_f32_kernel
	.p2align	8
	.type	v_tanh_f32_kernel,@function
v_tanh_f32_kernel:
	v_tanh_f32 v0, v0
	s_endpgm

	.globl	v_tanh_f32_omod_refuse_kernel
	.p2align	8
	.type	v_tanh_f32_omod_refuse_kernel,@function
v_tanh_f32_omod_refuse_kernel:
	v_tanh_f32_e64 v0, v0 mul:2
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_tanh_f32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.p2align	6, 0x0
	.amdhsa_kernel v_tanh_f32_omod_refuse_kernel
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
    .name:           v_tanh_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_tanh_f32_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_tanh_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_tanh_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
