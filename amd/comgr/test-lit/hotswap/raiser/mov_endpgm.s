; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; s0 is never read, so the moved value is dead and the lifted body is just the
; terminator.
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

; Raising the whole code object runs both kernels through one call, so the
; refusal names the kernel it came out of and the ISA pair it ran under rather
; than leaving the caller to work out which request failed.
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx950 --emit-ir 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BATCH
; BATCH: unsupported-instruction-form: v_mov_b32 {{.+}} in kernel 'vmov_kernel' (gfx942 -> gfx950)

; The target ISA is a parameter of the raise, so a gfx942 kernel raises onto
; another GPU. Nothing this kernel lifts to depends on which one, so what the
; run pins is that naming a second GPU stands its side of the raise up.
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx950 --emit-ir=mov_endpgm_kernel \
; RUN:   | %FileCheck %s --check-prefix=RETARGET
; RETARGET-LABEL: define amdgpu_kernel void @mov_endpgm_kernel(
; RETARGET: ret void

; An unrecognised ISA is refused before decoding, at whichever end named it.
; RUN: not %hotswap_transpile_cli %t.hsaco --isa=gfxbogus --emit-ir=mov_endpgm_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BADISA
; BADISA: source ISA 'gfxbogus' does not name an AMDGPU GPU

; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfxbogus --emit-ir=mov_endpgm_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BADTARGET
; BADTARGET: target ISA 'gfxbogus' does not name an AMDGPU GPU

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
