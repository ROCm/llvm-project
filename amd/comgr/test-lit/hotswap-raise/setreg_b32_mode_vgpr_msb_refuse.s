; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=setreg_b32_mode_vgpr_msb_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Regression guard: s_setreg_b32 targeting HW_REG_MODE with a field that
; overlaps VGPR_MSB bits [12:19] must be refused on gfx1250+. The raiser
; cannot statically determine the SGPR value, so it cannot update
; Ctx.VgprMsBs -- silently preserving the write would leave subsequent VGPR
; operand decoding using stale bank-selector state.

; STDERR: transpiler: s_setreg_b32 writes MODE with field overlapping VGPR_MSB bits
; STDERR-SAME: [12:19]
; STDERR: raise_cli: kernel 'setreg_b32_mode_vgpr_msb_kernel' failed to raise

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setreg_b32_mode_vgpr_msb_kernel
	.p2align	8
	.type	setreg_b32_mode_vgpr_msb_kernel,@function
setreg_b32_mode_vgpr_msb_kernel:
; %bb.0:
	; s_setreg_b32 targeting HW_REG_WAVE_MODE with field [12:19] (offset=12,
	; size=8). This writes s0 into the VGPR_MSB byte of MODE, but the raiser
	; cannot statically know the value of s0 -- refuse.
	s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setreg_b32_mode_vgpr_msb_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 0
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
  - .name:           setreg_b32_mode_vgpr_msb_kernel
    .symbol:         setreg_b32_mode_vgpr_msb_kernel.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 2
    .vgpr_count: 1
    .max_flat_workgroup_size: 64
    .args: []
...
	.end_amdgpu_metadata
