; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vopd_fmamk_vsrc1_vgpr_msb_kernel \
; RUN:   | %FileCheck %s

; Regression guard for issue #153: V_DUAL_FMAMK_F32's vsrc1 must read its
; s_set_vgpr_msb VGPR-MSB bank from operand slot 2 (bits [5:4]), NOT slot 1.
; V_DUAL_FMAMK has (src0 @0, literalK @1, vsrc1 @2) so vsrc1 is index 2.
;
; The kernel writes 1.0 into the low-bank copy of v3 (bank 0) and 2.0 into the
; high-bank copy (bank +1 == v3+256), then issues v_dual_fmamk_f32 with vsrc1=v3
; under s_set_vgpr_msb 0x10 (slot 2 == 1, slot 1 == 0). The literal K is 3.0.
; The raiser materialises the read-back constant as a bitcast from its i32 bit
; pattern (2.0 == 0x40000000 == 1073741824; 1.0 == 0x3f800000 == 1065353216):
;   Correct (slot 2): fma addend reads the high copy -> 2.0 (1073741824)
;   Buggy   (slot 1): fma addend reads the low copy  -> 1.0 (1065353216)
; The dead low-copy store is DCE'd post-fix, so 1065353216 must not appear at all.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_fmamk_vsrc1_vgpr_msb_kernel
	.p2align	8
	.type	vopd_fmamk_vsrc1_vgpr_msb_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @vopd_fmamk_vsrc1_vgpr_msb_kernel(
vopd_fmamk_vsrc1_vgpr_msb_kernel:
; %bb.0:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v8, v0
	; low-bank (bank 0) copy of v3 := 1.0  (the buggy slot-1 target)
	s_set_vgpr_msb 0
	v_dual_mov_b32 v2, v0 :: v_dual_mov_b32 v3, 1.0
	; high-bank (v3+256) copy := 2.0  (the correct slot-2 target)
	s_set_vgpr_msb 0x40
	v_dual_mov_b32 v2, v0 :: v_dual_mov_b32 v3, 2.0
	; fmamk: vsrc1 = v3, slot 2 bank = 1, slot 1 bank = 0, K = 3.0
	s_set_vgpr_msb 0x10
; CHECK: %[[ADDEND:[0-9]+]] = bitcast i32 1073741824 to float
; CHECK: call float @llvm.fma.f32(float %{{[^,]+}}, float 3.000000e+00, float %[[ADDEND]])
; CHECK-NOT: bitcast i32 1065353216 to float
	v_dual_mov_b32 v2, v8 :: v_dual_fmamk_f32 v3, v8, 0x40400000, v3
	s_set_vgpr_msb 0
	global_store_dword v0, v3, s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_fmamk_vsrc1_vgpr_msb_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 396
		.amdhsa_next_free_sgpr 2
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_vgpr_workitem_id 0
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.kernels:
  - .name:           vopd_fmamk_vsrc1_vgpr_msb_kernel
    .symbol:         vopd_fmamk_vsrc1_vgpr_msb_kernel.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 2
    .vgpr_count: 396
    .max_flat_workgroup_size: 64
    .args:
      - .name: out
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .is_const: false
        .is_restrict: false
        .is_volatile: false
        .type_name: uint32_t*
...
	.end_amdgpu_metadata
