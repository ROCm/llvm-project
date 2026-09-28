; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --isa=gfx900 --target-isa=gfx900 \
; RUN:   --emit-ir=readlane_lds_direct | %FileCheck %s
; RUN: %transpile_cli %t.hsaco --isa=gfx900 --target-isa=gfx942 \
; RUN:   --emit-ir=readlane_lds_direct | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl readlane_lds_direct
	.p2align 8
	.type readlane_lds_direct,@function
readlane_lds_direct:
; CHECK-LABEL: define amdgpu_kernel void @readlane_lds_direct(
	s_mov_b32 m0, 16
; CHECK: [[PTR:%.+]] = inttoptr i32 16 to ptr addrspace(3)
; CHECK: [[SRC:%.+]] = load i32, ptr addrspace(3) [[PTR]]
; CHECK: [[LANE:%.+]] = and i32 {{.+}}, 63
; CHECK: call i32 @llvm.amdgcn.readlane.i32(i32 [[SRC]], i32 [[LANE]])
	v_readlane_b32 s0, src_lds_direct, s0
	s_mov_b32 m0, 20
; CHECK: [[PTR2:%.+]] = inttoptr i32 20 to ptr addrspace(3)
; CHECK: [[SRC2:%.+]] = load i32, ptr addrspace(3) [[PTR2]]
; CHECK: call i32 @llvm.amdgcn.readlane.i32(i32 [[SRC2]], i32 7)
	v_readlane_b32 s1, src_lds_direct, 7
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel readlane_lds_direct
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: readlane_lds_direct
    .symbol: readlane_lds_direct.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 2
    .vgpr_count: 1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
