; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fmac_f32_fused_kernel,v_fmac_f32_vopd_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_fmac_f32 fused-FMA lift.
; CHECK-LABEL: define amdgpu_kernel void @v_fmac_f32_fused_kernel(
; CHECK: %fmac = call float @llvm.fma.f32(float %{{.+}}, float %{{.+}}, float %{{.+}})
; CHECK-NOT: call {{.*}}@llvm.fmuladd.f32
; CHECK-NOT: fmul {{.*}}float
; CHECK-NOT: fadd {{.*}}float
; CHECK-LABEL: define amdgpu_kernel void @v_fmac_f32_vopd_kernel(
; CHECK: %vopd_fmac = call float @llvm.fma.f32(float %{{.+}}, float %{{.+}}, float %{{.+}})
; CHECK-NOT: call {{.*}}@llvm.fmuladd.f32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_fmac_f32_fused_kernel
	.p2align	8
	.type	v_fmac_f32_fused_kernel,@function
v_fmac_f32_fused_kernel:
	v_fmac_f32_e64 v3, v1, v2
	s_endpgm

	.globl	v_fmac_f32_vopd_kernel
	.p2align	8
	.type	v_fmac_f32_vopd_kernel,@function
v_fmac_f32_vopd_kernel:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[0:1] offset:0
	global_load_b32 v2, v0, s[0:1] offset:4
	global_load_b32 v3, v0, s[0:1] offset:8
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v4, v0 :: v_dual_fmac_f32 v3, v1, v2
	global_store_b32 v0, v3, s[2:3]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_fmac_f32_fused_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.amdhsa_kernel v_fmac_f32_vopd_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
    .name:           v_fmac_f32_fused_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fmac_f32_fused_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_fmac_f32_vopd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_fmac_f32_vopd_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
