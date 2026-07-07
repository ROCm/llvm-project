; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=sendmsg_interrupt_kernel 2>/dev/null | %FileCheck %s --check-prefix=INTERRUPT
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=sendmsg_gs_alloc_req_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=sendmsg_dealloc_passthrough_kernel 2>/dev/null | %FileCheck %s --check-prefix=PASSTHROUGH
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=sendmsg_dealloc_drop_kernel 2>/dev/null | %FileCheck %s --check-prefix=DROP

; s_sendmsg interrupt lift, dealloc passthrough/drop, and gs_alloc_req [SOPP] refusal.
; INTERRUPT-LABEL: define amdgpu_kernel void @sendmsg_interrupt_kernel(
; INTERRUPT: call void @llvm.amdgcn.s.sendmsg(i32 1, i32 66)

; STDERR: raise_cli: kernel 'sendmsg_gs_alloc_req_kernel' failed to raise:
; STDERR-SAME: s_sendmsg
; STDERR-SAME: [SOPP]

; PASSTHROUGH-LABEL: define amdgpu_kernel void @sendmsg_dealloc_passthrough_kernel(
; PASSTHROUGH: call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)

; DROP-LABEL: define amdgpu_kernel void @sendmsg_dealloc_drop_kernel(
; DROP-NOT: call {{.*}}@llvm.amdgcn.s.sendmsg
; DROP: ret void

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sendmsg_interrupt_kernel
	.p2align	8
	.type	sendmsg_interrupt_kernel,@function
sendmsg_interrupt_kernel:
	s_mov_b32 m0, 0x42
	s_sendmsg sendmsg(MSG_INTERRUPT)
	s_endpgm
	.globl	sendmsg_gs_alloc_req_kernel
	.p2align	8
	.type	sendmsg_gs_alloc_req_kernel,@function
sendmsg_gs_alloc_req_kernel:
	s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
	s_endpgm
	.globl	sendmsg_dealloc_passthrough_kernel
	.p2align	8
	.type	sendmsg_dealloc_passthrough_kernel,@function
sendmsg_dealloc_passthrough_kernel:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.globl	sendmsg_dealloc_drop_kernel
	.p2align	8
	.type	sendmsg_dealloc_drop_kernel,@function
sendmsg_dealloc_drop_kernel:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sendmsg_interrupt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel sendmsg_gs_alloc_req_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel sendmsg_dealloc_passthrough_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel sendmsg_dealloc_drop_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sendmsg_interrupt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         sendmsg_interrupt_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sendmsg_gs_alloc_req_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         sendmsg_gs_alloc_req_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sendmsg_dealloc_passthrough_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         sendmsg_dealloc_passthrough_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sendmsg_dealloc_drop_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         sendmsg_dealloc_drop_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
