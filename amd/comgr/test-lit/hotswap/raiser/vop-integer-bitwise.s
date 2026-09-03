; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=vop_integer_bitwise \
; RUN:   | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	vop_integer_bitwise
	.p2align	8
	.type	vop_integer_bitwise,@function
; CHECK-LABEL: define amdgpu_kernel void @vop_integer_bitwise(
vop_integer_bitwise:
; CHECK: xor i32 {{.+}}, -1
	v_not_b32_e32 v4, v0
; CHECK: call i32 @llvm.bitreverse.i32
	v_bfrev_b32_e32 v5, v0
; CHECK: call i32 @llvm.ctlz.i32
; CHECK: select i1 {{.+}}, i32 -1
	v_ffbh_u32_e32 v6, v0
; CHECK: call i32 @llvm.cttz.i32
; CHECK: select i1 {{.+}}, i32 -1
	v_ffbl_b32_e32 v7, v0
; CHECK: [[FFBH_SIGN:%.+]] = ashr i32 {{.+}}, 31
; CHECK: [[FFBH_NORMALIZED:%.+]] = xor i32 {{.+}}, [[FFBH_SIGN]]
; CHECK: [[FFBH_COUNT:%.+]] = call i32 @llvm.ctlz.i32(i32 [[FFBH_NORMALIZED]], i1 false)
; CHECK: select i1 {{.+}}, i32 -1, i32 [[FFBH_COUNT]]
	v_ffbh_i32_e32 v8, v0

; CHECK: call i32 @llvm.ctpop.i32
; CHECK: add i32
	v_bcnt_u32_b32 v9, v0, v1
; CHECK: [[BFM_WIDTH:%.+]] = and i32 {{.+}}, 31
; CHECK: [[BFM_OFFSET:%.+]] = and i32 {{.+}}, 31
; CHECK: shl i32 {{.+}}, [[BFM_OFFSET]]
	v_bfm_b32 v10, v0, v1

; CHECK: [[LSHL_ADD_AMOUNT:%.+]] = and i32 {{.+}}, 31
; CHECK: shl i32 {{.+}}, [[LSHL_ADD_AMOUNT]]
; CHECK: add i32
	v_lshl_add_u32 v11, v0, v1, v2
; CHECK: [[ADD_LSHL_AMOUNT:%.+]] = and i32 {{.+}}, 31
; CHECK: shl i32 {{.+}}, [[ADD_LSHL_AMOUNT]]
	v_add_lshl_u32 v12, v0, v1, v2
; CHECK: [[LSHL_OR_AMOUNT:%.+]] = and i32 {{.+}}, 31
; CHECK: shl i32 {{.+}}, [[LSHL_OR_AMOUNT]]
; CHECK: or i32
	v_lshl_or_b32 v13, v0, v1, v2
; CHECK: and i32
; CHECK: or i32
	v_and_or_b32 v14, v0, v1, v2
; CHECK: or i32
; CHECK: or i32
	v_or3_b32 v15, v0, v1, v2
; CHECK: xor i32
; CHECK: add i32
	v_xad_u32 v16, v0, v1, v2
; CHECK: call i32 @llvm.fshr.i32
	v_alignbit_b32 v17, v0, v1, v2

; CHECK: [[UBFE_OFFSET:%.+]] = and i32 {{.+}}, 31
; CHECK: [[UBFE_WIDTH:%.+]] = and i32 {{.+}}, 31
; CHECK: lshr i32 {{.+}}, [[UBFE_OFFSET]]
; CHECK: select i1 {{.+}}, i32 {{.+}}, i32 0
	v_bfe_u32 v18, v0, v1, v2
; CHECK: [[IBFE_OFFSET:%.+]] = and i32 {{.+}}, 31
; CHECK: [[IBFE_WIDTH:%.+]] = and i32 {{.+}}, 31
; CHECK: ashr i32 {{.+}}, [[IBFE_OFFSET]]
; CHECK: select i1 {{.+}}, i32 {{.+}}, i32 0
	v_bfe_i32 v19, v0, v1, v2
; CHECK: xor i32 {{.+}}, -1
; CHECK: and i32
; CHECK: or i32
	v_bfi_b32 v20, v0, v1, v2
; CHECK: [[PERM_INPUT:%.+]] = or i64
; CHECK: [[PERM_SELECTOR:%.+]] = and i32 {{.+}}, 255
; CHECK: [[PERM_BYTE:%.+]] = select i1 {{.+}}, i8 -1, i8 {{.+}}
; CHECK: [[PERM_RESULT:%.+]] = or i32 {{.+}}, {{.+}}
	v_perm_b32 v21, v0, v1, v2

; CHECK: [[LSHR64_AMOUNT32:%.+]] = and i32 {{.+}}, 63
; CHECK: [[LSHR64_AMOUNT:%.+]] = zext i32 [[LSHR64_AMOUNT32]] to i64
; CHECK: lshr i64 {{.+}}, [[LSHR64_AMOUNT]]
	v_lshrrev_b64 v[22:23], v0, v[2:3]
; CHECK: [[ASHR64_AMOUNT32:%.+]] = and i32 {{.+}}, 63
; CHECK: [[ASHR64_AMOUNT:%.+]] = zext i32 [[ASHR64_AMOUNT32]] to i64
; CHECK: ashr i64 {{.+}}, [[ASHR64_AMOUNT]]
	v_ashrrev_i64 v[24:25], v0, v[2:3]
; CHECK: [[LSHL_ADD64_AMOUNT32:%.+]] = and i32 {{.+}}, 7
; CHECK: [[LSHL_ADD64_SUPPORTED:%.+]] = icmp ule i32 [[LSHL_ADD64_AMOUNT32]], 4
; CHECK: [[LSHL_ADD64_SELECTED:%.+]] = select i1 [[LSHL_ADD64_SUPPORTED]], i32 [[LSHL_ADD64_AMOUNT32]], i32 0
; CHECK: [[LSHL_ADD64_AMOUNT:%.+]] = zext i32 [[LSHL_ADD64_SELECTED]] to i64
; CHECK: shl i64 {{.+}}, [[LSHL_ADD64_AMOUNT]]
; CHECK: add i64
	v_lshl_add_u64 v[26:27], v[0:1], v2, v[2:3]
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop_integer_bitwise
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 28
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 28
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
    .vgpr_count:     28
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
