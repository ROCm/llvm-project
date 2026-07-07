; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cmp_class_f32_kernel 2>/dev/null | %FileCheck %s

; v_cmp_class_f32 class.f32+ballot lift.
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_class_f32_kernel(
; CHECK: %vclass{{[0-9]*}} = call i1 @llvm.amdgcn.class.f32(float %{{[^,]+}}, i32 512)
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %vclass{{[0-9]*}})
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32
; CHECK-NOT: fcmp {{.*}} float %{{[^,]+}}, {{.*}}i32
; CHECK-NOT: sext i1 %vclass{{[0-9]*}} to i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cmp_class_f32_kernel
	.p2align	8
	.type	v_cmp_class_f32_kernel,@function
v_cmp_class_f32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_class_f32_e64 s4, v1, 0x200
	s_mov_b32 s2, s4
	
	v_mov_b32_e32 v1, s2
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cmp_class_f32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 5
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
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_cmp_class_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         v_cmp_class_f32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
