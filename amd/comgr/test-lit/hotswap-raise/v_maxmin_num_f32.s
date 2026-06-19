; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_maxmin_num_f32_kernel 2>/dev/null | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --write-hsaco=%t.out --kernel=v_maxmin_num_f32_kernel 2>&1 | %FileCheck %s --check-prefix=PIPE
;
; Lift test for v_maxmin_num_f32, the .NUM dual of v_minmax_num_f32:
;   dst = minimumnum(maximumnum(src0, src1), src2)
; This is the NaN-pruning family and must not use IEEE maximum/minimum.

; IR-LABEL: define amdgpu_kernel void @v_maxmin_num_f32_kernel(
; IR: [[INNER:%[^ ]+]] = call float @llvm.maximumnum.f32(float %{{[^,]+}}, float %{{[^)]+}})
; IR: [[OUT:%[^ ]+]] = call float @llvm.minimumnum.f32(float [[INNER]], float %{{[^)]+}})
; IR-NOT: call {{.*}}@llvm.maximum.f32
; IR-NOT: call {{.*}}@llvm.minimum.f32

; PIPE: raise_cli: wrote
; PIPE-SAME: v_maxmin_num_f32_kernel

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_maxmin_num_f32_kernel
	.p2align	8
	.type	v_maxmin_num_f32_kernel,@function
v_maxmin_num_f32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, 0
	;;#ASMSTART
	v_maxmin_num_f32 v0, v0, v1, v2
	;;#ASMEND
	global_store_b32 v3, v0, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_maxmin_num_f32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
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
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_maxmin_num_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_maxmin_num_f32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
