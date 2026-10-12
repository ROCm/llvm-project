; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=packed_int16 | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_clamp 2>&1 | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	packed_int16
	.p2align	8
	.type	packed_int16,@function
; IR-LABEL: define amdgpu_kernel void @packed_int16(
packed_int16:
	; Each source register splits into the two lanes the operation acts on, and
	; the result is packed back into one register.
	; IR: [[ADD_LO:%.+]] = trunc i32 {{.+}} to i16
	; IR-NEXT: [[ADD_SHIFTED:%.+]] = lshr i32 {{.+}}, 16
	; IR-NEXT: [[ADD_HI:%.+]] = trunc i32 [[ADD_SHIFTED]] to i16
	; IR-NEXT: [[ADD_LANE0:%.+]] = insertelement <2 x i16> poison, i16 [[ADD_LO]], i64 0
	; IR-NEXT: [[ADD_SRC0:%.+]] = insertelement <2 x i16> [[ADD_LANE0]], i16 [[ADD_HI]], i64 1
	; IR: [[ADD:%.+]] = add <2 x i16> [[ADD_SRC0]], {{.+}}
	; IR-NEXT: {{.+}} = bitcast <2 x i16> [[ADD]] to i32
	v_pk_add_u16 v0, v1, v2
	; IR: {{.+}} = call <2 x i16> @llvm.sadd.sat.v2i16(
	v_pk_add_i16 v3, v4, v5 clamp
	; IR: {{.+}} = sub <2 x i16> {{.+}}, {{.+}}
	v_pk_sub_u16 v6, v7, v8
	; IR: {{.+}} = call <2 x i16> @llvm.usub.sat.v2i16(
	v_pk_sub_u16 v6, v7, v8 clamp
	; IR: {{.+}} = mul <2 x i16> {{.+}}, {{.+}}
	v_pk_mul_lo_u16 v9, v10, v11
	; The shift count is the first source and is read four bits wide.
	; IR: [[SHL_AMOUNT:%.+]] = and <2 x i16> {{.+}}, splat (i16 15)
	; IR-NEXT: {{.+}} = shl <2 x i16> {{.+}}, [[SHL_AMOUNT]]
	v_pk_lshlrev_b16 v12, v13, v14
	; IR: [[LSHR_AMOUNT:%.+]] = and <2 x i16> {{.+}}, splat (i16 15)
	; IR-NEXT: {{.+}} = lshr <2 x i16> {{.+}}, [[LSHR_AMOUNT]]
	v_pk_lshrrev_b16 v12, v13, v14
	; IR: [[ASHR_AMOUNT:%.+]] = and <2 x i16> {{.+}}, splat (i16 15)
	; IR-NEXT: {{.+}} = ashr <2 x i16> {{.+}}, [[ASHR_AMOUNT]]
	v_pk_ashrrev_i16 v15, v16, v17
	; IR: {{.+}} = call <2 x i16> @llvm.smin.v2i16(
	v_pk_min_i16 v18, v19, v20
	; op_sel feeds the high half of src0 to the low result lane, and op_sel_hi
	; feeds the high half of src1 to the high result lane.
	; IR: [[SEL_HI:%.+]] = trunc i32 [[SEL_SHIFTED:%.+]] to i16
	; IR-NEXT: {{.+}} = insertelement <2 x i16> poison, i16 [[SEL_HI]], i64 0
	; IR: {{.+}} = call <2 x i16> @llvm.umax.v2i16(
	v_pk_max_u16 v21, v22, v23 op_sel:[1,0] op_sel_hi:[0,1]
	; The lanes widen to 32 bits so a set clamp bit saturates the value the
	; hardware computes rather than a wrapped one.
	; IR: [[MAD_MUL:%.+]] = mul <2 x i32> {{.+}}, {{.+}}
	; IR: [[MAD_SUM:%.+]] = add <2 x i32> [[MAD_MUL]], {{.+}}
	; IR-NEXT: [[MAD_SAT:%.+]] = call <2 x i32> @llvm.umin.v2i32(<2 x i32> [[MAD_SUM]], <2 x i32> splat (i32 65535))
	; IR-NEXT: {{.+}} = trunc <2 x i32> [[MAD_SAT]] to <2 x i16>
	v_pk_mad_u16 v24, v25, v26, v27 clamp
	; IR: [[MIN3_INNER:%.+]] = call <2 x i16> @llvm.smin.v2i16(
	; IR-NEXT: {{.+}} = call <2 x i16> @llvm.smin.v2i16(<2 x i16> [[MIN3_INNER]],
	v_pk_min3_i16 v28, v29, v30, v31
	; IR: ret void
	s_endpgm

	.globl	refuse_clamp
	.p2align	8
	.type	refuse_clamp,@function
; REFUSE: unsupported-instruction-form: v_pk_mul_lo_u16 [VOP3P]
; REFUSE-SAME: packed integer operation does not define clamp
refuse_clamp:
	v_pk_mul_lo_u16 v0, v1, v2 clamp
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel packed_int16
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 32
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_clamp
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
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
    .name:           packed_int16
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         packed_int16.kd
    .vgpr_count:     32
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_clamp
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         refuse_clamp.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
