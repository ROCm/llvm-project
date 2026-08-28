; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=sop2_bitwise_gfx1250 | %FileCheck %s --check-prefix=IR
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=sop2_bitwise_gfx1250 | %FileCheck %s --check-prefix=NATIVE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sop2_bitwise_gfx1250
	.p2align	8
	.type	sop2_bitwise_gfx1250,@function
; IR-LABEL: define amdgpu_kernel void @sop2_bitwise_gfx1250(
; NATIVE-LABEL: define amdgpu_kernel void @sop2_bitwise_gfx1250(
sop2_bitwise_gfx1250:
	; NATIVE: %[[TARGET_WAVE:.*]] = call i32 @llvm.amdgcn.wave.id()
	; NATIVE: and i32 %[[TARGET_WAVE]], 31
	; IR-NOT: @llvm.amdgcn.wave.id
	; IR-DAG: %[[DISPATCH_PTR:.*]] = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
	; IR-DAG: %[[TID_X:.*]] = call i32 @llvm.amdgcn.workitem.id.x()
	; IR-DAG: %[[TID_Y:.*]] = call i32 @llvm.amdgcn.workitem.id.y()
	; IR-DAG: %[[TID_Z:.*]] = call i32 @llvm.amdgcn.workitem.id.z()
	; IR: %[[FLAT_YZ:.*]] = add i32 %[[TID_Y]], {{.*}}
	; IR: %[[FLAT_ID:.*]] = add i32 %[[TID_X]], {{.*}}
	; IR: %[[SOURCE_WAVE_ID:.*]] = udiv i32 %[[FLAT_ID]], 64
	; IR-NEXT: %[[WAVE_ID_MASKED:.*]] = and i32 %[[SOURCE_WAVE_ID]], 31
	; IR: icmp ne i32 %[[WAVE_ID_MASKED]], 0
	s_bfe_u32 s6, ttmp8, 0x50019
	s_mov_b32 ttmp8, 0
	; IR: %[[BFE_AFTER_CLOBBER:.*]] = select i1 {{.*}}, i32 0, i32 {{.*}}
	; IR: icmp ne i32 %[[BFE_AFTER_CLOBBER]], 0
	s_bfe_u32 s6, ttmp8, 0x50019
	s_mov_b32 exec_lo, s6
	; IR: %and_wave_mask = and i1 {{.*}}, {{.*}}
	; IR: call i64 @llvm.amdgcn.ballot.i64(i1 %and_wave_mask)
	; IR: %[[SCC_BALLOT:.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 %and_wave_mask)
	; IR: %[[SCC_AT_WAVE:.*]] = lshr i64 %[[SCC_BALLOT]], {{.*}}
	; IR-NEXT: %[[SCC_MASK:.*]] = trunc i64 %[[SCC_AT_WAVE]] to i32
	; IR-NEXT: %[[SCC:.*]] = icmp ne i32 %[[SCC_MASK]], 0
	s_and_b32 s2, exec_lo, -1
	; IR: select i1 %[[SCC]], i32 1, i32 0
	s_cselect_b32 s3, 1, 0
	s_mov_b32 s0, 0
	s_mov_b32 s1, 1
	; IR: %[[B64_SCALAR:.*]] = and i64 {{.*}}, -1
	; IR: %[[B64_SCC:.*]] = icmp ne i64 %[[B64_SCALAR]], 0
	s_and_b64 s[2:3], s[0:1], -1
	; IR: select i1 %[[B64_SCC]], i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR-NOT: @llvm.amdgcn.wave.id
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sop2_bitwise_gfx1250
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 2
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
    .reqd_workgroup_size: [8, 8, 2]
    .sgpr_count:     7
    .symbol:         sop2_bitwise_gfx1250.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
