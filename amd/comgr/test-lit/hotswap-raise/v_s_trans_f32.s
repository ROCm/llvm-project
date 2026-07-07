; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_s_trans_f32_kernel 2>/dev/null | %FileCheck %s
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_s_log_f32_clamp_kernel 2>&1 | %FileCheck %s --check-prefix=REFUSE-LOG
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_s_rcp_f32_omod_kernel 2>&1 | %FileCheck %s --check-prefix=REFUSE-RCP
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_s_rsq_f32_clamp_kernel 2>&1 | %FileCheck %s --check-prefix=REFUSE-RSQ
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_s_sqrt_f32_omod_kernel 2>&1 | %FileCheck %s --check-prefix=REFUSE-SQRT

; scalar v_s_log/rcp/rsq/sqrt_f32 lift to amdgcn log/rcp/rsq/sqrt.f32; non-default clamp/omod refused.
; CHECK-LABEL: define amdgpu_kernel void @v_s_trans_f32_kernel(
; CHECK: call float @llvm.amdgcn.log.f32(float {{.*}})
; CHECK: call float @llvm.amdgcn.rcp.f32(float {{.*}})
; CHECK: call float @llvm.amdgcn.rsq.f32(float {{.*}})
; CHECK: call float @llvm.amdgcn.sqrt.f32(float {{.*}})
; CHECK-NOT: call {{.*}}@llvm.log2.f32
; CHECK-NOT: fdiv float 1.000000e+00
; CHECK: declare {{.*}}float @llvm.amdgcn.log.f32(float)
; CHECK: declare {{.*}}float @llvm.amdgcn.rcp.f32(float)
; CHECK: declare {{.*}}float @llvm.amdgcn.rsq.f32(float)
; CHECK: declare {{.*}}float @llvm.amdgcn.sqrt.f32(float)
; REFUSE-LOG: V_S_LOG_F32 with non-default clamp/omod is not yet lifted
; REFUSE-RCP: V_S_RCP_F32 with non-default clamp/omod is not yet lifted
; REFUSE-RSQ: V_S_RSQ_F32 with non-default clamp/omod is not yet lifted
; REFUSE-SQRT: V_S_SQRT_F32 with non-default clamp/omod is not yet lifted

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_s_trans_f32_kernel
	.p2align	8
	.type	v_s_trans_f32_kernel,@function
v_s_trans_f32_kernel:
	v_s_log_f32 s0, s6
	v_s_rcp_f32 s1, s6
	v_s_rsq_f32 s2, s6
	v_s_sqrt_f32 s3, s6
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_s_trans_f32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

	.text
	.globl	v_s_log_f32_clamp_kernel
	.p2align	8
	.type	v_s_log_f32_clamp_kernel,@function
v_s_log_f32_clamp_kernel:
	v_s_log_f32 s0, s6 clamp
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_s_log_f32_clamp_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

	.text
	.globl	v_s_rcp_f32_omod_kernel
	.p2align	8
	.type	v_s_rcp_f32_omod_kernel,@function
v_s_rcp_f32_omod_kernel:
	v_s_rcp_f32 s0, s6 mul:2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_s_rcp_f32_omod_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

	.text
	.globl	v_s_rsq_f32_clamp_kernel
	.p2align	8
	.type	v_s_rsq_f32_clamp_kernel,@function
v_s_rsq_f32_clamp_kernel:
	v_s_rsq_f32 s0, s6 clamp
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_s_rsq_f32_clamp_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

	.text
	.globl	v_s_sqrt_f32_omod_kernel
	.p2align	8
	.type	v_s_sqrt_f32_omod_kernel,@function
v_s_sqrt_f32_omod_kernel:
	v_s_sqrt_f32 s0, s6 mul:2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_s_sqrt_f32_omod_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
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
    .name:           v_s_trans_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_s_trans_f32_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_s_log_f32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_s_log_f32_clamp_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_s_rcp_f32_omod_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_s_rcp_f32_omod_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_s_rsq_f32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_s_rsq_f32_clamp_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_s_sqrt_f32_omod_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_s_sqrt_f32_omod_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
