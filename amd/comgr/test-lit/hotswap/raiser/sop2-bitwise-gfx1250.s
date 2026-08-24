; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=sop2_bitwise_gfx1250 | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sop2_bitwise_gfx1250
	.p2align	8
	.type	sop2_bitwise_gfx1250,@function
; IR-LABEL: define amdgpu_kernel void @sop2_bitwise_gfx1250(
sop2_bitwise_gfx1250:
	; IR: [[TID:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
	; IR-NEXT: [[WAVE_ID:%.*]] = lshr i32 [[TID]], 5
	; IR-NEXT: [[WAVE_ID_MASKED:%.*]] = and i32 [[WAVE_ID]], 31
	; IR: icmp ne i32 [[WAVE_ID_MASKED]], 0
	s_bfe_u32 s6, ttmp8, 0x50019
	; IR: [[EXEC_MASK:%.*]] = and i1 {{.*}}, {{.*}}
	; IR: [[EXEC_BALLOT:%.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[EXEC_MASK]])
	; IR-NEXT: {{%.*}} = trunc i64 [[EXEC_BALLOT]] to i32
	; IR: call i64 @llvm.amdgcn.ballot.i64(i1 [[EXEC_MASK]])
	s_and_b32 exec_lo, exec_lo, -2
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sop2_bitwise_gfx1250
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 7
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
    .name:           sop2_bitwise_gfx1250
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         sop2_bitwise_gfx1250.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
