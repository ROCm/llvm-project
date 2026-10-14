; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=comparison_operands | %FileCheck %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.amdhsa_code_object_version 6
.text
.globl comparison_operands
.p2align 8
.type comparison_operands,@function
; CHECK-LABEL: define amdgpu_kernel void @comparison_operands(
comparison_operands:
; CHECK: [[CMP:%.+]] = fcmp une double {{.+}}, {{.+}}
; CHECK: [[PRED:%.+]] = select i1 {{.+}}, i1 [[CMP]], i1 false
; CHECK: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED]])
; CHECK: [[EXEC:%.+]] = and i64 {{.+}}, [[BALLOT]]
	v_cmpx_neq_f64_e32 vcc, v[0:1], v[2:3]
; CHECK: select i1 [[PRED]], i32 2, i32 1
	v_cndmask_b32_e64 v4, 1, 2, vcc
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED]])
	s_mov_b64 s[4:5], vcc
; CHECK: [[CLASS:%.+]] = call i1 @llvm.amdgcn.class.f64(double {{.+}}, i32 3)
; CHECK: [[CLASSBIT:%.+]] = select i1 {{.+}}, i1 [[CLASS]], i1 false
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[CLASSBIT]])
; CHECK: and i64 [[EXEC]], {{.+}}
	v_cmpx_class_f64_e64 s[4:5], v[0:1], 3
; CHECK: select i1 [[CLASSBIT]], i32 2, i32 1
	v_cndmask_b32_e64 v4, 1, 2, s[4:5]
; CHECK: [[CMP64:%.+]] = icmp sgt i64 {{.+}}, -1
; CHECK: [[PRED64:%.+]] = select i1 {{.+}}, i1 [[CMP64]], i1 false
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED64]])
	v_cmpx_gt_i64_e64 vcc, v[0:1], -1
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED64]])
	s_mov_b64 s[4:5], vcc
; CHECK: call i1 @llvm.amdgcn.class.f32(float {{.+}}, i32 {{.+}})
	v_cmp_class_f32_e32 vcc, v0, v1
	s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel comparison_operands
	.amdhsa_next_free_vgpr 5
	.amdhsa_next_free_sgpr 8
	.amdhsa_accum_offset 8
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
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 5
    .max_flat_workgroup_size: 1024
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
