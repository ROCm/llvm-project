; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_add_max_i32_kernel,v_add_max_i32_clamp_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_add_max_i32 sadd.sat+smax lift.
; CHECK-LABEL: define amdgpu_kernel void @v_add_max_i32_kernel(
; CHECK: %v_add_max_i32_sum{{[0-9]*}} = call i32 @llvm.sadd.sat.i32(i32 %{{[^,]+}}, i32 -1)
; CHECK: %v_add_max_i32{{[0-9]*}} = call i32 @llvm.smax.i32(i32 %v_add_max_i32_sum{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK-NOT: call {{.*}}@llvm.amdgcn.add.max.i32
; CHECK-LABEL: define amdgpu_kernel void @v_add_max_i32_clamp_kernel(
; CHECK: %v_add_max_i32_sum{{[0-9]*}} = call i32 @llvm.sadd.sat.i32(i32 {{[^,]+}}, i32 {{[^,]+}})
; CHECK: %v_add_max_i32{{[0-9]*}} = call i32 @llvm.smax.i32(i32 %v_add_max_i32_sum{{[0-9]*}}, i32 3)
; CHECK-NOT: call {{.*}}@llvm.amdgcn.add.max.i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_add_max_i32_kernel
	.p2align	8
	.type	v_add_max_i32_kernel,@function
v_add_max_i32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e64 v[0:1], s[0:1]
	global_load_b96 v[0:2], v[0:1], off
	s_wait_loadcnt 0x0
	v_add_max_i32 v0, s2, -1, v0
	global_store_b32 v3, v0, s[0:1] scale_offset
	s_endpgm
	.globl	v_add_max_i32_clamp_kernel
	.p2align	8
	.type	v_add_max_i32_clamp_kernel,@function
v_add_max_i32_clamp_kernel:
	v_mov_b32_e32 v0, 1
	v_mov_b32_e32 v1, 2
	v_mov_b32_e32 v2, 3
	v_add_max_i32 v0, v0, v1, v2 clamp
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_add_max_i32_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_add_max_i32_clamp_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           v_add_max_i32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         v_add_max_i32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_add_max_i32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_add_max_i32_clamp_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
