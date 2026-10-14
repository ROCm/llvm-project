; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=comparison_operands | %FileCheck %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text
.globl comparison_operands
.p2align 8
.type comparison_operands,@function
; CHECK-LABEL: define amdgpu_kernel void @comparison_operands(
comparison_operands:
; CHECK: [[HIGH:%.+]] = lshr i32 {{.+}}, 16
; CHECK: [[HALF0:%.+]] = trunc i32 [[HIGH]] to i16
; CHECK: [[HALF1:%.+]] = trunc i32 {{.+}} to i16
; CHECK: icmp slt i16 [[HALF0]], [[HALF1]]
	v_cmp_lt_i16_e64 s4, v0.h, v1.l
; CHECK: icmp eq i16 -1, {{.+}}
	v_cmp_eq_u16_e32 vcc_lo, 0xffff, v0.l
; CHECK: [[HIGH0:%.+]] = lshr i32 {{.+}}, 16
; CHECK: [[LOW0:%.+]] = trunc i32 [[HIGH0]] to i16
; CHECK: [[HIGH1:%.+]] = lshr i32 {{.+}}, 16
; CHECK: [[LOW1:%.+]] = trunc i32 [[HIGH1]] to i16
; CHECK: icmp ne i16 [[LOW0]], [[LOW1]]
	v_cmp_ne_u16_e32 vcc_lo, v0.h, v1.h
; CHECK: [[DOUBLE:%.+]] = bitcast i64 {{.+}} to double
; CHECK: [[ABS:%.+]] = call double @llvm.fabs.f64(double [[DOUBLE]])
; CHECK: [[NEG:%.+]] = fneg double [[ABS]]
; CHECK: fcmp une double [[NEG]], -1.000000e+00
	v_cmp_neq_f64_e64 vcc_lo, -|v[0:1]|, -1.0
; CHECK: [[FLOAT:%.+]] = bitcast i32 {{.+}} to float
; CHECK: [[ABS32:%.+]] = call float @llvm.fabs.f32(float [[FLOAT]])
; CHECK: fcmp ule float [[ABS32]], -2.000000e+00
	v_cmp_ngt_f32_e64 s4, |v0|, -2.0
; CHECK: [[ABS1:%.+]] = call float @llvm.fabs.f32(float {{.+}})
; CHECK: [[NEG1:%.+]] = fneg float [[ABS1]]
; CHECK: fcmp oeq float {{.+}}, [[NEG1]]
	v_cmp_eq_f32_e64 s4, v0, -|v1|
; CHECK: fcmp une double f0x1234567800000000, {{.+}}
	v_cmp_neq_f64_e64 s4, 0x1234567800000000, v[0:1]
; CHECK: icmp ne i64 305419896, {{.+}}
	v_cmp_ne_u64_e64 s4, 0x12345678, v[0:1]
; CHECK: icmp slt i64 -2147483648, {{.+}}
	v_cmp_lt_i64_e64 s4, 0x80000000, v[0:1]
; CHECK: icmp eq i64 -2147483648, {{.+}}
	v_cmp_eq_i64_e64 s4, 0x80000000, v[0:1]
; CHECK: icmp eq i64 2147483648, {{.+}}
	v_cmp_eq_u64_e64 s4, 0x80000000, v[0:1]
; CHECK: icmp eq i64 2147483648, {{.+}}
	v_cmp_eq_i64_e32 vcc_lo, lit64(0x80000000), v[0:1]
; CHECK: fcmp une double 1.000000e+00, {{.+}}
	v_cmp_neq_f64_e64 s4, lit(1.0), v[0:1]
	s_mov_b32 s6, 1
	s_mov_b32 s7, 0x80000000
; CHECK: icmp ult i64 1, {{.+}}
	v_cmp_lt_u64_e64 s4, 1, s[6:7]
; CHECK: call i1 @llvm.amdgcn.class.f32(float {{.+}}, i32 3)
	v_cmp_class_f32_e64 s4, |v0|, 3
; CHECK: call i1 @llvm.amdgcn.class.f64(double {{.+}}, i32 516)
	v_cmp_class_f64_e64 s4, -v[0:1], 0x204
; CHECK: [[CMP:%.+]] = fcmp ord float {{.+}}, {{.+}}
; CHECK: [[SAVED:%.+]] = select i1 {{.+}}, i1 [[CMP]], i1 false
	v_cmp_o_f32_e32 vcc_lo, v0, v1
; CHECK: select i1 [[SAVED]], i32 2, i32 1
	v_cndmask_b32_e64 v2, 1, 2, vcc_lo
; CHECK: [[CMPX:%.+]] = fcmp ule float {{.+}}, {{.+}}
; CHECK: [[BIT:%.+]] = select i1 {{.+}}, i1 [[CMPX]], i1 false
; CHECK: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[BIT]])
; CHECK-NEXT: [[BASE:%.+]] = and i32 {{%.+}}, -32
; CHECK-NEXT: [[SHIFT:%.+]] = zext i32 [[BASE]] to i64
; CHECK-NEXT: [[SLICE:%.+]] = lshr i64 [[BALLOT]], [[SHIFT]]
; CHECK-NEXT: [[MASK:%.+]] = trunc i64 [[SLICE]] to i32
; CHECK: [[EXEC:%.+]] = and i32 {{.+}}, [[MASK]]
	v_cmpx_ngt_f32_e64 v0, v1
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[SAVED]])
	s_mov_b32 s5, vcc_lo
; CHECK: lshr i32 [[EXEC]], {{.+}}
	v_mov_b32 v2, 17
	s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel comparison_operands
	.amdhsa_next_free_vgpr 5
	.amdhsa_next_free_sgpr 8
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .name: comparison_operands
    .symbol: comparison_operands.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 5
    .max_flat_workgroup_size: 1024
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
