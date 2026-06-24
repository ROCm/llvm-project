; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=saveexec_vcc_hi_kernel 2>&1 | %FileCheck %s
;
; Wave32 source -> wave64 target: vcc_hi is a scratch scalar, not the wave mask.
; A saveexec naming vcc_hi as its destination must write the saved EXEC into the
; scratch slot; one naming vcc_hi as its source must read the mask back from it.

; CHECK-LABEL: define amdgpu_kernel void @saveexec_vcc_hi_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	saveexec_vcc_hi_kernel
	.p2align	8
	.type	saveexec_vcc_hi_kernel,@function
saveexec_vcc_hi_kernel:
; Destination: saved old EXEC is narrowed into the vcc_hi scratch slot.
; CHECK: %[[SAVED:.+]] = trunc i64 %{{.+}} to i32
	s_and_saveexec_b32 vcc_hi, s4
; The cndmask reads vcc_hi scratch, consuming the saved value.
; CHECK: zext i32 %[[SAVED]] to i64
; CHECK: = select i1 %{{.+}}, i32
	v_cndmask_b32 v6, v2, v3, vcc_hi
; Source: second saveexec reads vcc_hi scratch as the AND mask.
; CHECK: %wn_src_to_exec_zext{{[0-9]*}} = zext i32 %[[SAVED]] to i64
; CHECK: and i64 %new_exec, %wn_src_to_exec_mask{{[0-9]*}}
	s_and_saveexec_b32 s6, vcc_hi
	ds_store_b32 v7, v6
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel saveexec_vcc_hi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           saveexec_vcc_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         saveexec_vcc_hi_kernel.kd
    .vgpr_count:     9
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
