; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:        --emit-ir=setpc_terminal_fallthrough_kernel 2>&1 | %FileCheck %s
;
; Regression for a kernel-boundary-violation in the s_set_pc_i64 analysis.
;
; When the FINAL decoded instruction of a kernel is an unconditional
; s_set_pc_i64 (a DirectA tail-jump / Pattern-B return), its fallthrough
; offset is `di.offset + di.size` == the kernel's one-past-the-end
; boundary (KernelEndOffset). Phase 1 of setpc-analysis.cpp used to
; register that fallthrough as an `extraBlockStart` unconditionally; the
; raiser then rejected the whole kernel with
;   "s_set_pc_i64 analysis discovered a target outside the selected
;    kernel extent"
; even though an unconditional s_set_pc has no fallthrough control-flow
; edge at all (nothing ever executes the boundary offset).
;
; This reproduces the shape seen in the wan2.2 Triton kernel
; triton_poi_fused__fused_rms_norm_cat_...view_8: a tail region of
; getpc+add+add+s_set_pc chains whose very last instruction is a
; terminal s_set_pc landing exactly at the kernel end. The fix gates the
; fallthrough block-leader registration on an instruction actually
; beginning at that offset, so the boundary offset is no longer treated
; as a decode target.
;
; The explicit `.size` below ends the kernel symbol exactly at the
; terminal s_set_pc, so its fallthrough == KernelEndOffset (the bug
; trigger). The chain resolves to an in-kernel block (.Ltarget), so the
; site is DirectA and lowers to a `br label %bb_0x<target>`.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setpc_terminal_fallthrough_kernel
	.p2align	8
	.type	setpc_terminal_fallthrough_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @setpc_terminal_fallthrough_kernel(
setpc_terminal_fallthrough_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
.Ltarget:
	v_mov_b32 v1, 0xDEAD0001
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	; Terminal DirectA set_pc: chain resolves to .Ltarget (a non-entry
	; in-kernel block). It is the LAST instruction; its fallthrough lands
	; on the kernel-end boundary.
	s_get_pc_i64 s[10:11]
.Lpost:
	s_add_co_u32 s10, s10, (.Ltarget - .Lpost)
	s_add_co_ci_u32 s11, s11, ((.Ltarget - .Lpost) >> 32)
; CHECK: [[TGT:bb_0x[0-9a-f]+]]:
	s_set_pc_i64 s[10:11]
; CHECK: br label %[[TGT]]
; CHECK-NOT: unreachable
.Lkernel_end:
	.size	setpc_terminal_fallthrough_kernel, .Lkernel_end - setpc_terminal_fallthrough_kernel
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setpc_terminal_fallthrough_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           setpc_terminal_fallthrough_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         setpc_terminal_fallthrough_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
