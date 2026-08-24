; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=sop2_integer \
; RUN:   | %FileCheck %s --check-prefix=DECODE
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=sop2_integer \
; RUN:   | %FileCheck %s --check-prefix=IR

; Verify the integer arithmetic and selection family, including distinct
; signed and unsigned add/sub operations.
; DECODE: S_ADD_U32{{.+}}s_add_u32
; DECODE: S_ADD_I32{{.+}}s_add_i32
; DECODE: S_ADDC_U32
; DECODE: S_SUB_U32{{.+}}s_sub_u32
; DECODE: S_SUB_I32{{.+}}s_sub_i32
; DECODE: S_SUBB_U32
; DECODE: S_ABSDIFF_I32
; DECODE: S_MUL_I32
; DECODE: S_MUL_HI_U32
; DECODE: S_MUL_HI_I32
; DECODE: S_CSELECT_B32
; DECODE: S_CSELECT_B64
; DECODE: S_MIN_I32
; DECODE: S_MIN_U32
; DECODE: S_MAX_I32
; DECODE: S_MAX_U32
; DECODE: S_LSHL1_ADD_U32
; DECODE: S_LSHL2_ADD_U32
; DECODE: S_LSHL3_ADD_U32
; DECODE: S_LSHL4_ADD_U32

; Pin the important semantic distinctions in the raised module: carry/borrow,
; signed versus unsigned multiply-high and comparisons, selection on SCC, and
; widened shift-add carry.
; IR-LABEL: define amdgpu_kernel void @sop2_integer(
; IR: call { i32, i1 } @llvm.uadd.with.overflow.i32
; IR: call { i32, i1 } @llvm.sadd.with.overflow.i32
; IR-COUNT-2: call { i32, i1 } @llvm.uadd.with.overflow.i32
; IR: call { i32, i1 } @llvm.usub.with.overflow.i32
; IR: call { i32, i1 } @llvm.ssub.with.overflow.i32
; IR-COUNT-2: call { i32, i1 } @llvm.usub.with.overflow.i32
; IR: zext i32 {{.*}} to i64
; IR: sext i32 {{.*}} to i64
; IR: mul i64
; IR: icmp slt i32
; IR: icmp ult i32
; IR: select i1
; IR: lshl4_add_wide = add i64
; IR: lshl4_add_carry = icmp ugt i64
; IR: ret void

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	sop2_integer
	.p2align	8
	.type	sop2_integer,@function
sop2_integer:
	s_add_u32 s2, s0, s1
	s_add_i32 s2, s0, s1
	s_addc_u32 s2, s0, s1
	s_sub_u32 s2, s0, s1
	s_sub_i32 s2, s0, s1
	s_subb_u32 s2, s0, s1
	s_absdiff_i32 s2, s0, s1
	s_mul_i32 s2, s0, s1
	s_mul_hi_u32 s2, s0, s1
	s_mul_hi_i32 s2, s0, s1
	s_cselect_b32 s2, s0, s1
	s_cselect_b64 s[2:3], s[0:1], s[4:5]
	s_min_i32 s2, s0, s1
	s_min_u32 s2, s0, s1
	s_max_i32 s2, s0, s1
	s_max_u32 s2, s0, s1
	s_lshl1_add_u32 s2, s0, s1
	s_lshl2_add_u32 s2, s0, s1
	s_lshl3_add_u32 s2, s0, s1
	s_lshl4_add_u32 s2, s0, s1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sop2_integer
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 4
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
    .name:           sop2_integer
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         sop2_integer.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
