; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vop_integer_bitwise | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vop_integer_bitwise
	.p2align	8
	.type	vop_integer_bitwise,@function
; CHECK-LABEL: define amdgpu_kernel void @vop_integer_bitwise(
vop_integer_bitwise:
; Exercise both native VOP1 and VOP3-encoded forms.
; CHECK: xor i32 {{.+}}, -1
	v_not_b32_e64 v4, v0
; CHECK: call i32 @llvm.bitreverse.i32
	v_bfrev_b32_e32 v5, v0
; CHECK: call i32 @llvm.ctlz.i32
	v_clz_i32_u32_e32 v6, v0
; CHECK: call i32 @llvm.cttz.i32
	v_ctz_i32_b32_e64 v7, v0
; CHECK: call i32 @llvm.amdgcn.sffbh.i32
	v_cls_i32_e32 v8, v0

; CHECK: call i32 @llvm.ctpop.i32
	v_bcnt_u32_b32 v9, v0, v1
; CHECK: bfm
	v_bfm_b32 v10, v0, v1
; CHECK: and i32
	v_and_b32_e64 v30, v0, v1
; CHECK: or i32
	v_or_b32_e64 v31, v0, v1
; CHECK: xor i32
	v_xor_b32_e64 v32, v0, v1
; CHECK: xnor
	v_xnor_b32_e64 v33, v0, v1
; CHECK: lshl
	v_lshlrev_b32_e64 v34, v0, v1
; CHECK: lshr
	v_lshrrev_b32_e64 v35, v0, v1
; CHECK: ashr
	v_ashrrev_i32_e64 v36, v0, v1
; CHECK: lshl.add
	v_lshl_add_u32 v11, v0, v1, v2
; CHECK: add.lshl
	v_add_lshl_u32 v12, v0, v1, v2
; CHECK: lshl.or
	v_lshl_or_b32 v13, v0, v1, v2
; CHECK: and.or
	v_and_or_b32 v14, v0, v1, v2
; CHECK: or3
	v_or3_b32 v15, v0, v1, v2
; CHECK: xor3
	v_xor3_b32 v16, v0, v1, v2
; CHECK: xad
	v_xad_u32 v17, v0, v1, v2
; CHECK: call i32 @llvm.fshr.i32
	v_alignbit_b32 v18, v0, v1, v2
; CHECK: bfe
	v_bfe_u32 v19, v0, v1, v2
; CHECK: bfe.sign.extend
	v_bfe_i32 v20, v0, v1, v2
; CHECK: bfi
	v_bfi_b32 v21, v0, v1, v2
; CHECK: call i32 @llvm.amdgcn.perm
	v_perm_b32 v22, v0, v1, v2
; CHECK: bitop3
	v_bitop3_b32 v23, v0, v1, v2 bitop3:0x96

; CHECK: lshr64
	v_lshrrev_b64 v[24:25], v0, v[2:3]
; CHECK: ashr64
	v_ashrrev_i64 v[26:27], v0, v[2:3]
; CHECK: lshl64
	v_lshlrev_b64_e64 v[38:39], v0, v[2:3]
; CHECK: lshl.add64
	v_lshl_add_u64 v[28:29], v[0:1], v2, v[2:3]
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop_integer_bitwise
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 1
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop_integer_bitwise
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vop_integer_bitwise.kd
    .vgpr_count:     40
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
