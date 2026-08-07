; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=mfma_fp8_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS

; The scaled F8F6F4 family picks each operand's element format from cbsz /
; blgp at run time, so its fp8/bf8 bytes cannot be re-encoded for gfx942's
; FNUZ hardware. Refuse instead of passing OCP bytes through unconverted.
; RUN: %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=mfma_scale_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=SCALED_CROSS

; gfx90a has MFMA but no fp8/bf8 hardware, so there is nothing to re-encode
; the operands for; refuse instead of emitting an unselectable intrinsic.
; RUN: %not raise_cli %t.hsaco --target-isa=gfx90a --emit-ir=mfma_fp8_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=NO_FP8 \
; RUN:       --implicit-check-not=llvm.amdgcn.mfma

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	mfma_fp8_kernel
	.p2align	8
	.type	mfma_fp8_kernel,@function
mfma_fp8_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
; The MFMA operands must be REBUILT from the converted dwords. The raiser runs
; no DCE, so a conversion CHECK that is not tied to the call's operands still
; matches when the def-use edge is dropped and OCP bytes reach FNUZ hardware.
; CROSS-LABEL: define amdgpu_kernel void @mfma_fp8_kernel(
; CROSS: %[[AB0:[^ ]+]] = trunc <4 x i32> %e4m3_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[AD0:[^ ]+]] = bitcast <4 x i8> %[[AB0]] to i32
; CROSS: %[[AB1:[^ ]+]] = trunc <4 x i32> %e4m3_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[AD1:[^ ]+]] = bitcast <4 x i8> %[[AB1]] to i32
; CROSS: %[[AZ0:[^ ]+]] = zext i32 %[[AD0]] to i64
; CROSS: %[[AZ1:[^ ]+]] = zext i32 %[[AD1]] to i64
; CROSS: %[[ASH:[^ ]+]] = shl i64 %[[AZ1]], 32
; CROSS: %[[A:[^ ]+]] = or i64 %[[AZ0]], %[[ASH]]
; CROSS: %[[BB0:[^ ]+]] = trunc <4 x i32> %e4m3_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[BD0:[^ ]+]] = bitcast <4 x i8> %[[BB0]] to i32
; CROSS: %[[BB1:[^ ]+]] = trunc <4 x i32> %e4m3_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[BD1:[^ ]+]] = bitcast <4 x i8> %[[BB1]] to i32
; CROSS: %[[BZ0:[^ ]+]] = zext i32 %[[BD0]] to i64
; CROSS: %[[BZ1:[^ ]+]] = zext i32 %[[BD1]] to i64
; CROSS: %[[BSH:[^ ]+]] = shl i64 %[[BZ1]], 32
; CROSS: %[[B:[^ ]+]] = or i64 %[[BZ0]], %[[BSH]]
; CROSS: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %[[A]], i64 %[[B]], <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; NO_FP8: kernel 'mfma_fp8_kernel' failed to raise:
; NO_FP8-SAME: v_mfma_f32_16x16x32_fp8_fp8
; NO_FP8-SAME: target ISA has no fp8/bf8 hardware
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[0:1], v[2:3], v[4:7]
	s_nop 8
; Mixed format: A is bf8 (E5M2), B is fp8 (E4M3). Each side must pick its own
; converter, so swapping the two `Fp8Sides` booleans has to fail here.
; CROSS: %[[MAB0:[^ ]+]] = trunc <4 x i32> %e5m2_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[MAD0:[^ ]+]] = bitcast <4 x i8> %[[MAB0]] to i32
; CROSS: %[[MAB1:[^ ]+]] = trunc <4 x i32> %e5m2_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[MAD1:[^ ]+]] = bitcast <4 x i8> %[[MAB1]] to i32
; CROSS: %[[MAZ0:[^ ]+]] = zext i32 %[[MAD0]] to i64
; CROSS: %[[MAZ1:[^ ]+]] = zext i32 %[[MAD1]] to i64
; CROSS: %[[MASH:[^ ]+]] = shl i64 %[[MAZ1]], 32
; CROSS: %[[MA:[^ ]+]] = or i64 %[[MAZ0]], %[[MASH]]
; CROSS: %[[MBB0:[^ ]+]] = trunc <4 x i32> %e4m3_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[MBD0:[^ ]+]] = bitcast <4 x i8> %[[MBB0]] to i32
; CROSS: %[[MBB1:[^ ]+]] = trunc <4 x i32> %e4m3_fnuz{{[0-9]*}} to <4 x i8>
; CROSS: %[[MBD1:[^ ]+]] = bitcast <4 x i8> %[[MBB1]] to i32
; CROSS: %[[MBZ0:[^ ]+]] = zext i32 %[[MBD0]] to i64
; CROSS: %[[MBZ1:[^ ]+]] = zext i32 %[[MBD1]] to i64
; CROSS: %[[MBSH:[^ ]+]] = shl i64 %[[MBZ1]], 32
; CROSS: %[[MB:[^ ]+]] = or i64 %[[MBZ0]], %[[MBSH]]
; CROSS: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %[[MA]], i64 %[[MB]], <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
	v_mfma_f32_16x16x32_bf8_fp8 v[12:15], v[0:1], v[2:3], v[12:15]
	s_nop 8
	v_mov_b32_e32 v8, 0
	global_store_dwordx4 v8, v[4:7], s[0:1]
	global_store_dwordx4 v8, v[12:15], s[0:1] offset:16
	s_endpgm
	.globl	mfma_scale_kernel
	.p2align	8
	.type	mfma_scale_kernel,@function
mfma_scale_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
; SCALED_CROSS: kernel 'mfma_scale_kernel' failed to raise:
; SCALED_CROSS-SAME: scaled F8F6F4 MFMA crosses an fp8/bf8 OCP<->FNUZ boundary
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] v20, v21
	s_nop 8
	v_mov_b32_e32 v22, 0
	global_store_dwordx4 v22, v[0:3], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mfma_scale_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 23
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 24
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel mfma_fp8_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mfma_fp8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         mfma_fp8_kernel.kd
    .vgpr_count:     16
    .wavefront_size: 64
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mfma_scale_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         mfma_scale_kernel.kd
    .vgpr_count:     23
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
