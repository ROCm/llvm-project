; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vopd_bitop2_bitop3_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_dual_bitop2 with a bitop3 truth-table lifts to the encoded boolean op (xor here).
; CHECK-LABEL: define amdgpu_kernel void @vopd_bitop2_bitop3_kernel(
; CHECK: xor i32 %tid, -1
; CHECK: xor i32 1, -1
; CHECK-NOT: %vopd_and = and i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_bitop2_bitop3_kernel
	.p2align	8
	.type	vopd_bitop2_bitop3_kernel,@function
vopd_bitop2_bitop3_kernel:
; %bb.0:
	v_mov_b32_e32 v3, 1
	v_dual_lshlrev_b32 v1, 1, v0 :: v_dual_bitop2_b32 v2, v0, v3 bitop3:0x14
	global_store_dword v0, v2, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_bitop2_bitop3_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 4
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
    .name:           vopd_bitop2_bitop3_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         vopd_bitop2_bitop3_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
