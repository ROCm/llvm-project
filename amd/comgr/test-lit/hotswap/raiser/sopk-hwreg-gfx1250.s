; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=sopk_hwreg | %FileCheck %s --check-prefix=IR
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_dynamic_vgpr_msb 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-DYNAMIC

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sopk_hwreg
	.p2align	8
	.type	sopk_hwreg,@function
; IR-LABEL: define amdgpu_kernel void @sopk_hwreg(
sopk_hwreg:
	; The diagnostic register read is modeled as zero.
	s_getreg_b32 s0, hwreg(HW_REG_IB_STS2, 6, 4)
	; IR: call void @llvm.amdgcn.s.setreg(i32 2305, i32 0)
	s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 4, 2), s0
	; IR: call void @llvm.amdgcn.s.setreg(i32 1601, i32 4097)
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 0x1001
	; The diagnostic register write is dropped.
	s_setreg_imm32_b32 hwreg(HW_REG_IB_STS2), 7
	; IR-NOT: call void @llvm.amdgcn.s.setreg
	; IR: ret void
	s_endpgm

	.globl	refuse_dynamic_vgpr_msb
	.p2align	8
	.type	refuse_dynamic_vgpr_msb,@function
refuse_dynamic_vgpr_msb:
	; REFUSE-DYNAMIC: unsupported-instruction-form
	; REFUSE-DYNAMIC-SAME: s_setreg_b32
	; REFUSE-DYNAMIC-SAME: dynamic MODE write overlaps VGPR_MSB bits [12:19]
	s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sopk_hwreg
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel refuse_dynamic_vgpr_msb
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
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
    .name:           sopk_hwreg
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sopk_hwreg.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_dynamic_vgpr_msb
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_dynamic_vgpr_msb.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
