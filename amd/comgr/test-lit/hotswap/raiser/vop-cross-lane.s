; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --isa=gfx1250 \
; RUN:   --dump-decoded=cross_lane | %FileCheck %s --check-prefix=DECODE
; RUN: %hotswap_transpile_cli %t.hsaco --isa=gfx1250 \
; RUN:   --emit-ir=cross_lane \
; RUN:   | %FileCheck %s --check-prefix=SAME
; RUN: %hotswap_transpile_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:   --emit-ir=cross_lane | %FileCheck %s --check-prefix=WIDEN

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cross_lane
	.p2align	8
	.type	cross_lane,@function
cross_lane:
; SAME-LABEL: define amdgpu_kernel void @cross_lane(
	v_mov_b32_e32 v0, s0
; DECODE: V_READFIRSTLANE_B32  v_readfirstlane_b32 s4, v0
; SAME: call i32 @llvm.amdgcn.readfirstlane.i32
; WIDEN: call i32 @llvm.cttz.i32
; WIDEN: call i32 @llvm.amdgcn.ds.bpermute
; WIDEN: call i32 @llvm.amdgcn.strict.wwm.i32
	v_readfirstlane_b32 s4, v0
; DECODE: V_READLANE_B32  v_readlane_b32 s5, v0, 7
; SAME: call i32 @llvm.amdgcn.readlane.i32
; WIDEN: [[READ_ADDR:%.+]] = shl i32 {{.+}}, 2
; WIDEN: call i32 @llvm.amdgcn.ds.bpermute(i32 [[READ_ADDR]], i32 {{.+}})
	v_readlane_b32 s5, v0, 7
	s_mov_b32 s6, 123
	v_mov_b32_e32 v1, 0
	s_mov_b32 exec_lo, 0
; DECODE: V_WRITELANE_B32  v_writelane_b32 v1, s6, 9
; SAME: call i32 @llvm.amdgcn.writelane.i32
; WIDEN: [[SOURCE_LANE:%.+]] = and i32 {{.+}}, 31
; WIDEN: [[IS_SELECTED:%.+]] = icmp eq i32 [[SOURCE_LANE]], 9
; WIDEN: select i1 [[IS_SELECTED]], i32 123, i32 {{.+}}
	v_writelane_b32 v1, s6, 9
; SAME: ret void
; WIDEN: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cross_lane
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 2
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
    .name:           cross_lane
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         cross_lane.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
