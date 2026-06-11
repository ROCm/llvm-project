; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=setreg_mode_vgpr_msb_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression guard for gfx1250 MODE-setreg VGPR_MSB capture. For why this
; matters, see handleSOPK() in handle-sopk.cpp: the `HwregId == HwregIdMode`
; branch of the S_SETREG_IMM32_B32 handling, and the VgprMsb* constants in
; amdgpu-mode-hwreg.h.
;
; This test drives the handoff end-to-end: a prologue MODE setreg with imm
; 0x1001 sets DST_VGPR_MSB=01; the following writelane to v0 must lift as a
; Vgpr256 phi; a later SRC0_VGPR_MSB=01 read of v0 must source that phi (not
; Vgpr0); and the final store keeps the Vgpr256 def-use chain alive past
; dead-code elimination (DCE).

; CHECK-LABEL: define amdgpu_kernel void @setreg_mode_vgpr_msb_kernel(
; CHECK: [[V256:%Vgpr256[._0-9]*]] = phi i32 [ %cwd_writelane_rewritten
; CHECK: [[V1:%Vgpr1[._0-9]*]] = phi i32 [ [[V256]]
; CHECK: store i32 [[V1]], ptr addrspace(1)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setreg_mode_vgpr_msb_kernel
	.p2align	8
	.type	setreg_mode_vgpr_msb_kernel,@function
setreg_mode_vgpr_msb_kernel:
; %bb.0:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	; Canonical gfx1250 kernel-prologue MODE setreg.  Bit 0 of the imm
	; programs REPLAY_MODE (per `hwreg(HW_REG_WAVE_MODE, 25, 1)`); bits
	; 12..19 of the imm program VGPR_MSB.  Here imm[12:19] = 0x01
	; corresponds to MODE-format DST_VGPR_MSB = 0b01, i.e. dst register
	; encodings get +256 until another MODE/SET_VGPR_MSB writes the
	; field.
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 0x1001
	; Writelane encodes v0 but, with DST_VGPR_MSB=01 in effect, the
	; assembler resolves it to physical v256 (note `/*v256*/` in
	; disassembly).
	v_writelane_b32 v0, s0, 0
	; Drop DST_VGPR_MSB back to 0 and raise SRC0_VGPR_MSB to 0b01 so
	; the following v_mov reads physical v256 (the writelane target)
	; through its v0-named encoding.  S_SET_VGPR_MSB low byte is the
	; new state (current=0x01 -> src0=01, src1=0, src2=0, dst=0); high
	; byte (0x40 -> previous dst=01) is compiler bookkeeping.
	s_set_vgpr_msb 0x4001
	v_mov_b32_e32 v1, v0
	s_set_vgpr_msb 0
	; Force the writelane->read chain to be live across the global
	; store so DCE cannot strip the Vgpr256 phi.
	s_wait_xcnt 0x0
	s_wait_kmcnt 0x0
	global_store_b32 v1, v1, s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setreg_mode_vgpr_msb_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 257
		.amdhsa_next_free_sgpr 2
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_vgpr_workitem_id 0
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.kernels:
  - .name:           setreg_mode_vgpr_msb_kernel
    .symbol:         setreg_mode_vgpr_msb_kernel.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 2
    .vgpr_count: 257
    .max_flat_workgroup_size: 64
    .args:
      - .name: out
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .is_const: false
        .is_restrict: false
        .is_volatile: false
        .type_name: uint32_t*
...
	.end_amdgpu_metadata
