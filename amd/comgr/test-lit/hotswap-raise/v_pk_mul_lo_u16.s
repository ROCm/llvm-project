; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_pk_mul_lo_u16_kernel 2>/dev/null | %FileCheck %s

; v_pk_mul_lo_u16 packed u16 multiply lift.
; CHECK-LABEL: define amdgpu_kernel void @v_pk_mul_lo_u16_kernel(
; CHECK: %pk_mul_lo_u16{{[0-9]*}} = mul <2 x i16> %{{[^,]+}}, %{{[^)]+}}
; CHECK: %pk_i16_pack{{[0-9]*}} = bitcast <2 x i16> %pk_mul_lo_u16{{[0-9]*}} to i32
; CHECK-NOT: mul nuw <2 x i16>
; CHECK-NOT: mul nsw <2 x i16>
; CHECK-NOT: call {{.*}} @llvm.amdgcn.pk
; CHECK-NOT: fadd <2 x i16>
; CHECK-NOT: add <2 x i16>
; CHECK-NOT: zext <2 x i16> %pk_mul_lo_u16{{[0-9]*}} to <2 x i32>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	v_pk_mul_lo_u16_kernel
	.p2align	8
	.type	v_pk_mul_lo_u16_kernel,@function
v_pk_mul_lo_u16_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, 0x00030002
	v_mov_b32_e32 v2, 0x00050007
	v_pk_mul_lo_u16 v3, v1, v2
	global_store_dword v0, v3, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_mul_lo_u16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: v_pk_mul_lo_u16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_pk_mul_lo_u16_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
