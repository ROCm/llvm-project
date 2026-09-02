; REQUIRES: comgr-has-hotswap-transpile

; The SOPP opcodes that set a mode or send a message. The mode ones are gfx10
; additions and the VGPR bank select is a gfx1250 one, so the fixture is a
; gfx1250 kernel; sopp_messages_wave64.s covers what a pre-gfx11 source makes
; of the same messages.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=sethalt_kernel,vgpr_msb_kernel,mode_kernel,messages_kernel \
; RUN:   | %FileCheck %s

; RUN: not %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=round_up_kernel,denorm_flush_kernel,dealloc_halt_kernel,sysmsg_kernel,sysmsg_halt_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	sethalt_kernel
	.p2align	8
	.type	sethalt_kernel,@function
sethalt_kernel:
; The immediate reaches the intrinsic unchanged, so two different halts stay
; two different halts.
; CHECK-LABEL: define amdgpu_kernel void @sethalt_kernel(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.amdgcn.s.sethalt(i32 1)
	s_sethalt 1
; CHECK-NEXT: call void @llvm.amdgcn.s.sethalt(i32 3)
	s_sethalt 3
; CHECK-NEXT: ret void
	s_endpgm

	.globl	vgpr_msb_kernel
	.p2align	8
	.type	vgpr_msb_kernel,@function
vgpr_msb_kernel:
; The bank select renumbers the VGPRs the instructions after it name, which the
; raiser applies where it resolves a VGPR operand. It emits nothing of its own,
; and it leaves the scalar work around it alone.
; CHECK-LABEL: define amdgpu_kernel void @vgpr_msb_kernel(
	s_mov_b32 s2, 3
; CHECK-NOT: llvm.amdgcn
	s_set_vgpr_msb 0x1
; CHECK: [[REV:%.+]] = call i32 @llvm.bitreverse.i32(i32 3)
	s_brev_b32 s3, s2
; The high byte records the bank that was selected before, which the hardware
; ignores and the raiser drops: this one goes back to the low bank.
; CHECK-NOT: llvm.amdgcn
	s_set_vgpr_msb 0x100
; CHECK: call i32 @llvm.bitreverse.i32(i32 [[REV]])
	s_brev_b32 s4, s3
; CHECK: ret void
	s_endpgm

	.globl	mode_kernel
	.p2align	8
	.type	mode_kernel,@function
mode_kernel:
; Naming the mode the raised kernel already computes in asks for nothing, so
; both of these lift to nothing.
; CHECK-LABEL: define amdgpu_kernel void @mode_kernel(
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
; CHECK-NEXT: }
	s_round_mode 0x0
	s_denorm_mode 15
	s_endpgm

	.globl	messages_kernel
	.p2align	8
	.type	messages_kernel,@function
messages_kernel:
; Both interrupts go out carrying M0, which is what the message reads its
; payload from.
; CHECK-LABEL: define amdgpu_kernel void @messages_kernel(
; CHECK-NEXT: entry:
	s_mov_b32 m0, 42
; CHECK-NEXT: call void @llvm.amdgcn.s.sendmsg(i32 1, i32 42)
	s_sendmsg sendmsg(MSG_INTERRUPT)
; CHECK-NEXT: call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 42)
	s_sendmsghalt sendmsg(MSG_INTERRUPT)
; The deallocation hint is dropped rather than passed on: the target backend
; decides where the raised kernel is done with its VGPRs.
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; CHECK-NEXT: ret void
	s_endpgm

	.globl	round_up_kernel
	.p2align	8
	.type	round_up_kernel,@function
round_up_kernel:
; REFUSE: unsupported-instruction-form: s_round_mode [SOPP]
; REFUSE-SAME: selects rounding mode 1 rather than round-to-nearest-even
	s_round_mode 0x1
	s_endpgm

	.globl	denorm_flush_kernel
	.p2align	8
	.type	denorm_flush_kernel,@function
denorm_flush_kernel:
; REFUSE: unsupported-instruction-form: s_denorm_mode [SOPP]
; REFUSE-SAME: selects denormal mode 0 rather than keeping denormals
	s_denorm_mode 0
	s_endpgm

	.globl	dealloc_halt_kernel
	.p2align	8
	.type	dealloc_halt_kernel,@function
dealloc_halt_kernel:
; Dropping the deallocation hint is only free when nothing else rides on it,
; and the halting spelling carries a halt as well.
; REFUSE: unsupported-instruction-form: s_sendmsghalt [SOPP]
; REFUSE-SAME: halts the wave alongside a VGPR deallocation
	s_sendmsghalt sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm

	.globl	sysmsg_kernel
	.p2align	8
	.type	sysmsg_kernel,@function
sysmsg_kernel:
; REFUSE: unsupported-instruction-form: s_sendmsg [SOPP]
; REFUSE-SAME: sends message 0x1f, and the interrupt is the only message that
; REFUSE-SAME: means the same thing on every target
	s_sendmsg 0x1f
	s_endpgm

	.globl	sysmsg_halt_kernel
	.p2align	8
	.type	sysmsg_halt_kernel,@function
sysmsg_halt_kernel:
; REFUSE: unsupported-instruction-form: s_sendmsghalt [SOPP]
; REFUSE-SAME: sends message 0x1f
	s_sendmsghalt 0x1f
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sethalt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel vgpr_msb_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel mode_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel messages_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel round_up_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel denorm_flush_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel dealloc_halt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sysmsg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sysmsg_halt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
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
    .name:           sethalt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sethalt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vgpr_msb_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         vgpr_msb_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           mode_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         mode_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           messages_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         messages_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           round_up_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         round_up_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           denorm_flush_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         denorm_flush_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dealloc_halt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         dealloc_halt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sysmsg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sysmsg_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sysmsg_halt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sysmsg_halt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
