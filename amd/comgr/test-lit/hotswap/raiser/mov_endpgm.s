; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; mov_endpgm_kernel lifts: the raiser decodes its .text, seeds the entry
; registers, dispatches the scalar move and program end, and emits a valid
; kernel function. s0 is never read, so the moved value is dead and only the
; entry seeding plus the terminator remain.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=mov_endpgm_kernel | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @mov_endpgm_kernel(
; CHECK: ret void

; The decoder maps the two instructions onto their canonical ops.
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=mov_endpgm_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE

; The dispatch only routes the landed SOP1 / SOPP families; a VALU opcode has
; no handler yet, so vmov_kernel is refused with a structured diagnostic rather
; than mislowered or crashed.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=vmov_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: v_mov_b32

; An unrecognised source ISA is refused before decoding.
; RUN: not %hotswap_transpile_cli %t.hsaco --isa=gfxbogus --emit-ir=mov_endpgm_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BADISA
; BADISA: does not name an AMDGPU GPU

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	mov_endpgm_kernel
	.p2align	8
	.type	mov_endpgm_kernel,@function
mov_endpgm_kernel:
; DECODE: S_MOV_B32{{.+}}s_mov_b32 s0, 0
	s_mov_b32 s0, 0
; DECODE: S_ENDPGM{{.+}}s_endpgm
	s_endpgm

	.globl	vmov_kernel
	.p2align	8
	.type	vmov_kernel,@function
vmov_kernel:
	v_mov_b32_e32 v0, 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mov_endpgm_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel vmov_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
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
    .name:           mov_endpgm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         mov_endpgm_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vmov_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vmov_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
