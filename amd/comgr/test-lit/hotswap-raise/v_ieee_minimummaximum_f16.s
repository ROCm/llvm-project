; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_ieee_minimummaximum_f16_kernel 2>/dev/null | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --write-hsaco=%t.out --kernel=v_ieee_minimummaximum_f16_kernel 2>&1 | %FileCheck %s --check-prefix=PIPE
;
; Canary for the f16 minimum/maximum family. IEEE forms must use
; NaN-propagating half intrinsics, .NUM forms must use maximumnum/minimumnum,
; and high-half source/destination op_sel must be explicit in IR.

; IR-LABEL: define amdgpu_kernel void @v_ieee_minimummaximum_f16_kernel(
; IR-NOT: @llvm.maxnum.f16
; IR-NOT: @llvm.minnum.f16
; IR: call half @llvm.maximum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: call half @llvm.minimum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: [[MAX3_INNER:%[^ ]+]] = call half @llvm.maximum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: call half @llvm.maximum.f16(half [[MAX3_INNER]], half %{{[^)]+}})
; IR: [[MIN3_INNER:%[^ ]+]] = call half @llvm.minimum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: call half @llvm.minimum.f16(half [[MIN3_INNER]], half %{{[^)]+}})
; IR: [[MAXMIN_INNER:%[^ ]+]] = call half @llvm.maximum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: call half @llvm.minimum.f16(half [[MAXMIN_INNER]], half %{{[^)]+}})
; IR: [[MINMAX_INNER:%[^ ]+]] = call half @llvm.minimum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: call half @llvm.maximum.f16(half [[MINMAX_INNER]], half %{{[^)]+}})
; IR-NOT: @llvm.maxnum.f16
; IR-NOT: @llvm.minnum.f16
; IR: [[NUM_MINMAX_INNER:%[^ ]+]] = call half @llvm.minimumnum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: call half @llvm.maximumnum.f16(half [[NUM_MINMAX_INNER]], half %{{[^)]+}})
; IR: [[NUM_MAXMIN_INNER:%[^ ]+]] = call half @llvm.maximumnum.f16(half %{{[^,]+}}, half %{{[^)]+}})
; IR: call half @llvm.minimumnum.f16(half [[NUM_MAXMIN_INNER]], half %{{[^)]+}})
; IR: f16_src_hi
; IR: f16_merge_hi

; PIPE: raise_cli: wrote
; PIPE-SAME: v_ieee_minimummaximum_f16_kernel

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_ieee_minimummaximum_f16_kernel
	.p2align	8
	.type	v_ieee_minimummaximum_f16_kernel,@function
v_ieee_minimummaximum_f16_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, 0
	;;#ASMSTART
	v_maximum_f16 v4, v0, v1
	v_minimum_f16 v5, v0, v1
	v_maximum3_f16 v6, v0, v1, v2
	v_minimum3_f16 v7, v0, v1, v2
	v_maximumminimum_f16 v8, v0, v1, v2
	v_minimummaximum_f16 v9, v0, v1, v2
	v_minmax_num_f16 v10, v0, v1, v2
	v_maxmin_num_f16 v11, v0, v1, v2
	v_maximumminimum_f16 v12, v0, v1, v2 op_sel:[1,1,1,1]
	;;#ASMEND
	global_store_b32 v3, v4, s[0:1]
	global_store_b32 v3, v5, s[0:1] offset:4
	global_store_b32 v3, v6, s[0:1] offset:8
	global_store_b32 v3, v7, s[0:1] offset:12
	global_store_b32 v3, v8, s[0:1] offset:16
	global_store_b32 v3, v9, s[0:1] offset:20
	global_store_b32 v3, v10, s[0:1] offset:24
	global_store_b32 v3, v11, s[0:1] offset:28
	global_store_b32 v3, v12, s[0:1] offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_ieee_minimummaximum_f16_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 13
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_16_64 3
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
    .name:           v_ieee_minimummaximum_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_ieee_minimummaximum_f16_kernel.kd
    .vgpr_count:     13
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
