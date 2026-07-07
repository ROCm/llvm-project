; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_mixlo_f16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_fma_mixlo_f16 low-half packed FMA-mix lift.
; CHECK-LABEL: define amdgpu_kernel void @v_fma_mixlo_f16_kernel(
; CHECK-DAG: %mixlo_cvt = fpext half %{{.*}} to float
; CHECK-DAG: lshr i32 %{{.*}}, 16
; CHECK-DAG: %mixlo_cvt{{[0-9]+}} = fpext half %{{.*}} to float
; CHECK: %fma_mixlo_f16 = call float @llvm.fma.f32(float %mixlo_cvt, float %mixlo_cvt{{[0-9]+}}, float %{{.*}})
; CHECK: %fma_mixlo_f16_round = fptrunc float %fma_mixlo_f16 to half
; CHECK: bitcast half %fma_mixlo_f16_round to i16
; CHECK: zext i16 %{{.*}} to i32
; CHECK: %fma_mixlo_f16_old_hi = and i32 %{{.*}}, -65536
; CHECK: %fma_mixlo_f16_pack = or i32 %fma_mixlo_f16_old_hi, %{{.*}}
; CHECK-NOT: unsupported instruction

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_fma_mixlo_f16_kernel
	.p2align	8
	.type	v_fma_mixlo_f16_kernel,@function
v_fma_mixlo_f16_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_add_nc_u32_e64 v3, s0, 4
	v_add_nc_u32_e64 v5, s0, 8
	v_mov_b32_e32 v1, s0
	v_mov_b32_e32 v2, s0
	v_fma_mixlo_f16 v2, v1, v3, v5 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_mixlo_f16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 2
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
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_fma_mixlo_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         v_fma_mixlo_f16_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
