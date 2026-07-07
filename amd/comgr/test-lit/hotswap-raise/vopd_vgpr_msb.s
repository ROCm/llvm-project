; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vopd_vgpr_msb_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; VOPD component operands use s_set_vgpr_msb high-VGPR addressing (gfx125x VGPR MSB).
; CHECK-LABEL: define amdgpu_kernel void @vopd_vgpr_msb_kernel(
; CHECK-NOT: shl i32 undef, 4
; CHECK: shl i32 %{{[^,]+}}, 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_vgpr_msb_kernel
	.p2align	8
	.type	vopd_vgpr_msb_kernel,@function
vopd_vgpr_msb_kernel:
; %bb.0:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 7
	s_set_vgpr_msb 0x40
	v_dual_mov_b32 v139, 0 :: v_dual_bitop2_b32 v138, 3, v0 bitop3:0x40
	s_set_vgpr_msb 0x4
	v_lshlrev_b32_e32 v1, 4, v138
	s_set_vgpr_msb 0
	global_store_dword v0, v1, s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_vgpr_msb_kernel
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
  - .name:           vopd_vgpr_msb_kernel
    .symbol:         vopd_vgpr_msb_kernel.kd
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
