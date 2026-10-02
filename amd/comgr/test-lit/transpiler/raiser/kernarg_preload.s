; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=preloaded_kernarg | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=private_segment_size 2>&1 | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

; Recreate the preloads from their recorded kernarg offsets. Offset 2 dwords
; starts at byte 8, so the test catches a zero-offset implementation.
; The pointer occupies s[0:1]; the four preloads occupy s2-s5.
	.globl	preloaded_kernarg
	.p2align	8
	.type	preloaded_kernarg,@function
; IR-LABEL: define amdgpu_kernel void @preloaded_kernarg(
; IR: [[SEG:%.+]] = call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
; IR: [[GEP0:%.+]] = getelementptr inbounds i8, ptr addrspace(4) [[SEG]], i64 8
; IR-NEXT: [[DW0:%.+]] = load i32, ptr addrspace(4) [[GEP0]], align 4
; IR: [[GEP1:%.+]] = getelementptr inbounds i8, ptr addrspace(4) [[SEG]], i64 12
; IR-NEXT: [[DW1:%.+]] = load i32, ptr addrspace(4) [[GEP1]], align 4
; IR: [[GEP2:%.+]] = getelementptr inbounds i8, ptr addrspace(4) [[SEG]], i64 16
; IR-NEXT: [[DW2:%.+]] = load i32, ptr addrspace(4) [[GEP2]], align 4
; IR: [[GEP3:%.+]] = getelementptr inbounds i8, ptr addrspace(4) [[SEG]], i64 20
; IR-NEXT: [[DW3:%.+]] = load i32, ptr addrspace(4) [[GEP3]], align 4
preloaded_kernarg:
; Use the first and last preloads so the offset-to-SGPR mapping is observable.
; IR: [[SUM:%.+]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[DW0]], i32 [[DW3]])
; IR-NEXT: [[ADD:%.+]] = extractvalue { i32, i1 } [[SUM]], 0
	s_add_co_i32 s6, s2, s5
; IR: phi i32 [ [[ADD]], %spe_do ]
	v_mov_b32 v2, s6
	v_mov_b32 v3, 0
	global_store_b32 v3, v2, s[2:3]
	s_endpgm

; Private-segment size has no equivalent target value and must still refuse.
	.globl	private_segment_size
	.p2align	8
	.type	private_segment_size,@function
; REFUSE: unsupported-entry-sgpr-source
; REFUSE-SAME: in kernel 'private_segment_size'
; REFUSE-SAME: s0 holds an entry source the raise cannot reproduce
private_segment_size:
	v_mov_b32 v0, s0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel preloaded_kernarg
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_kernarg_preload_length 4
		.amdhsa_user_sgpr_kernarg_preload_offset 2
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel private_segment_size
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_private_segment_size 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           preloaded_kernarg
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         preloaded_kernarg.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           private_segment_size
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         private_segment_size.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
