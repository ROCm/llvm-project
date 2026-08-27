; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=sop2_integer \
; RUN:   | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	sop2_integer
	.p2align	8
	.type	sop2_integer,@function
; IR-LABEL: define amdgpu_kernel void @sop2_integer(
sop2_integer:
	; IR: [[UADD:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[UADD]], 0
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[UADD]], 1
	s_add_u32 s2, s0, s1
	; IR: [[SADD:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[SADD]], 0
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[SADD]], 1
	s_add_i32 s2, s0, s1
	; IR: [[CARRY_IN:%.*]] = zext i1 {{%.*}} to i32
	; IR-NEXT: [[ADDC_FIRST:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32
	; IR-NEXT: [[ADDC_SUM:%.*]] = extractvalue { i32, i1 } [[ADDC_FIRST]], 0
	; IR-NEXT: [[ADDC_SECOND:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[ADDC_SUM]], i32 [[CARRY_IN]])
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[ADDC_SECOND]], 0
	; IR-NEXT: [[ADDC_CARRY0:%.*]] = extractvalue { i32, i1 } [[ADDC_FIRST]], 1
	; IR-NEXT: [[ADDC_CARRY1:%.*]] = extractvalue { i32, i1 } [[ADDC_SECOND]], 1
	; IR-NEXT: {{%.*}} = or i1 [[ADDC_CARRY0]], [[ADDC_CARRY1]]
	s_addc_u32 s2, s0, s1
	; IR: [[USUB:%.*]] = call { i32, i1 } @llvm.usub.with.overflow.i32
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[USUB]], 0
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[USUB]], 1
	s_sub_u32 s2, s0, s1
	; IR: [[SSUB:%.*]] = call { i32, i1 } @llvm.ssub.with.overflow.i32
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[SSUB]], 0
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[SSUB]], 1
	s_sub_i32 s2, s0, s1
	; IR: [[BORROW_IN:%.*]] = zext i1 {{%.*}} to i32
	; IR-NEXT: [[SUBB_FIRST:%.*]] = call { i32, i1 } @llvm.usub.with.overflow.i32
	; IR-NEXT: [[SUBB_DIFF:%.*]] = extractvalue { i32, i1 } [[SUBB_FIRST]], 0
	; IR-NEXT: [[SUBB_SECOND:%.*]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[SUBB_DIFF]], i32 [[BORROW_IN]])
	; IR-NEXT: {{%.*}} = extractvalue { i32, i1 } [[SUBB_SECOND]], 0
	; IR-NEXT: [[SUBB_BORROW0:%.*]] = extractvalue { i32, i1 } [[SUBB_FIRST]], 1
	; IR-NEXT: [[SUBB_BORROW1:%.*]] = extractvalue { i32, i1 } [[SUBB_SECOND]], 1
	; IR-NEXT: {{%.*}} = or i1 [[SUBB_BORROW0]], [[SUBB_BORROW1]]
	s_subb_u32 s2, s0, s1
	; IR: [[DIFF:%.*]] = sub i32 {{.*}}
	; IR-NEXT: [[IS_NEGATIVE:%.*]] = icmp slt i32 [[DIFF]], 0
	; IR-NEXT: [[NEGATED:%.*]] = sub i32 0, [[DIFF]]
	; IR-NEXT: [[ABSDIFF:%.*]] = select i1 [[IS_NEGATIVE]], i32 [[NEGATED]], i32 [[DIFF]]
	; IR: {{%.*}} = icmp ne i32 [[ABSDIFF]], 0
	s_absdiff_i32 s2, s0, s1
	; IR: {{%.*}} = mul i32 {{.*}}
	s_mul_i32 s2, s0, s1
	; IR: [[MULHU_A:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[MULHU_B:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[MULHU_WIDE:%.*]] = mul i64 [[MULHU_A]], [[MULHU_B]]
	; IR-NEXT: [[MULHU_SHIFTED:%.*]] = lshr i64 [[MULHU_WIDE]], 32
	; IR-NEXT: {{%.*}} = trunc i64 [[MULHU_SHIFTED]] to i32
	s_mul_hi_u32 s2, s0, s1
	; IR: [[MULHI_A:%.*]] = sext i32 {{.*}} to i64
	; IR-NEXT: [[MULHI_B:%.*]] = sext i32 {{.*}} to i64
	; IR-NEXT: [[MULHI_WIDE:%.*]] = mul i64 [[MULHI_A]], [[MULHI_B]]
	; IR-NEXT: [[MULHI_SHIFTED:%.*]] = lshr i64 [[MULHI_WIDE]], 32
	; IR-NEXT: {{%.*}} = trunc i64 [[MULHI_SHIFTED]] to i32
	s_mul_hi_i32 s2, s0, s1
	; IR: {{%.*}} = select i1 {{%.*}}, i32 {{.*}}, i32 {{.*}}
	s_cselect_b32 s2, s0, s1
	; IR: [[CSELECT_LO0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[CSELECT_HI0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[CSELECT_SHL0:%.*]] = shl i64 [[CSELECT_HI0]], 32
	; IR-NEXT: [[CSELECT_SRC0:%.*]] = or i64 [[CSELECT_LO0]], [[CSELECT_SHL0]]
	; IR-NEXT: [[CSELECT_LO1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[CSELECT_HI1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[CSELECT_SHL1:%.*]] = shl i64 [[CSELECT_HI1]], 32
	; IR-NEXT: [[CSELECT_SRC1:%.*]] = or i64 [[CSELECT_LO1]], [[CSELECT_SHL1]]
	; IR-NEXT: [[CSELECT:%.*]] = select i1 {{%.*}}, i64 [[CSELECT_SRC0]], i64 [[CSELECT_SRC1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[CSELECT]] to i32
	; IR-NEXT: [[CSELECT_SHIFTED:%.*]] = lshr i64 [[CSELECT]], 32
	; IR-NEXT: {{%.*}} = trunc i64 [[CSELECT_SHIFTED]] to i32
	s_cselect_b64 s[2:3], s[0:1], s[4:5]
	; IR: [[SMIN_COND:%.*]] = icmp slt i32 {{.*}}
	; IR-NEXT: {{%.*}} = select i1 [[SMIN_COND]], i32 {{.*}}, i32 {{.*}}
	; IR-NOT: icmp ne i32
	s_min_i32 s2, s0, s1
	; IR: [[UMIN_COND:%.*]] = icmp ult i32 {{.*}}
	; IR-NEXT: {{%.*}} = select i1 [[UMIN_COND]], i32 {{.*}}, i32 {{.*}}
	; IR-NOT: icmp ne i32
	s_min_u32 s2, s0, s1
	; IR: [[SMAX_COND:%.*]] = icmp sge i32 {{.*}}
	; IR-NEXT: {{%.*}} = select i1 [[SMAX_COND]], i32 {{.*}}, i32 {{.*}}
	; IR-NOT: icmp ne i32
	s_max_i32 s2, s0, s1
	; IR: [[UMAX_COND:%.*]] = icmp uge i32 {{.*}}
	; IR-NEXT: {{%.*}} = select i1 [[UMAX_COND]], i32 {{.*}}, i32 {{.*}}
	; IR-NOT: icmp ne i32
	s_max_u32 s2, s0, s1
	; IR: [[LSHL1_S0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL1_S1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL1_SHIFTED:%.*]] = shl i64 [[LSHL1_S0]], 1
	; IR-NEXT: [[LSHL1_WIDE:%.*]] = add i64 [[LSHL1_SHIFTED]], [[LSHL1_S1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[LSHL1_WIDE]] to i32
	; IR: {{%.*}} = icmp ugt i64 [[LSHL1_WIDE]], 4294967295
	s_lshl1_add_u32 s2, s0, s1
	; IR: [[LSHL2_S0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL2_S1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL2_SHIFTED:%.*]] = shl i64 [[LSHL2_S0]], 2
	; IR-NEXT: [[LSHL2_WIDE:%.*]] = add i64 [[LSHL2_SHIFTED]], [[LSHL2_S1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[LSHL2_WIDE]] to i32
	; IR: {{%.*}} = icmp ugt i64 [[LSHL2_WIDE]], 4294967295
	s_lshl2_add_u32 s2, s0, s1
	; IR: [[LSHL3_S0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL3_S1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL3_SHIFTED:%.*]] = shl i64 [[LSHL3_S0]], 3
	; IR-NEXT: [[LSHL3_WIDE:%.*]] = add i64 [[LSHL3_SHIFTED]], [[LSHL3_S1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[LSHL3_WIDE]] to i32
	; IR: {{%.*}} = icmp ugt i64 [[LSHL3_WIDE]], 4294967295
	s_lshl3_add_u32 s2, s0, s1
	; IR: [[LSHL4_S0:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL4_S1:%.*]] = zext i32 {{.*}} to i64
	; IR-NEXT: [[LSHL4_SHIFTED:%.*]] = shl i64 [[LSHL4_S0]], 4
	; IR-NEXT: [[LSHL4_WIDE:%.*]] = add i64 [[LSHL4_SHIFTED]], [[LSHL4_S1]]
	; IR-NEXT: {{%.*}} = trunc i64 [[LSHL4_WIDE]] to i32
	; IR: {{%.*}} = icmp ugt i64 [[LSHL4_WIDE]], 4294967295
	s_lshl4_add_u32 s2, s0, s1
	; IR: ret void
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
