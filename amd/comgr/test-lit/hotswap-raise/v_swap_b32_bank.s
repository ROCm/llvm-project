; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_swap_b32_bank_kernel 2>/dev/null | %FileCheck %s
;
; Exercises the V_SWAP_B32 handler (handle-valu.cpp).  v_swap_b32 exchanges
; vdst and src0; the writeback to src0's slot (the src0_out def, tied to src0)
; must land in src0's own VGPR_MSB bank, not the destination bank that
; computeVGPRAdjust assigns to every def slot -- exactly the
; v_permlane16_swap_b32 src0_out bug (see permlane16_swap_bank.s).  The two
; banks only diverge when vdst and src0 differ: s_set_vgpr_msb 0x40 puts vdst
; in bank1 and src0 in bank0, so the src0 writeback must target src0's bank-0
; register (Vgpr2), not the bank-1 alias Vgpr258.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	v_swap_b32_bank_kernel
	.p2align	8
	.type	v_swap_b32_bank_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_swap_b32_bank_kernel(
v_swap_b32_bank_kernel:
	global_load_b32 v1, v0, s[0:1]
	global_load_b32 v2, v0, s[2:3]
	s_set_vgpr_msb 0x40
	v_swap_b32 v1, v2
	; src0 (v2) is loaded into its real bank-0 register Vgpr2:
	; CHECK: %[[V2LOAD:Vgpr2\.[0-9]+]] = phi i32 [ %{{[0-9]+}}, %spe_do{{[0-9]+}} ]
	; The swap writes src0's slot back into src0's OWN bank-0 register, so a
	; second Vgpr2 phi consumes the previous Vgpr2 value.  Under the
	; destination-bank bug this writeback targets the dead Vgpr258 alias and
	; this phi never appears.
	; CHECK: %[[V2SWAP:Vgpr2\.[0-9]+]] = phi i32 {{.*}}[ %[[V2LOAD]], %spe_skip{{[0-9]+}} ]
	s_set_vgpr_msb 0
	global_store_b32 v0, v1, s[0:1]
	global_store_b32 v0, v2, s[2:3]
	; The swapped value in src0's bank is what gets stored back:
	; CHECK: store i32 %[[V2SWAP]],
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_swap_b32_bank_kernel
		.amdhsa_next_free_vgpr 3
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
    .name:           v_swap_b32_bank_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_swap_b32_bank_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
