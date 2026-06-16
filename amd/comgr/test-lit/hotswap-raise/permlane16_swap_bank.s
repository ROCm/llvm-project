; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=permlane16_swap_bank_kernel 2>/dev/null | %FileCheck %s
;
; Exercises emitPermLaneSwapEmulation (handle-valu-cross-lane.cpp), which lowers
; v_permlane16_swap_b32.  Its two outputs are tied: vdst to vdst_in and src0_out
; to src0.  src0_out must be written back to src0's own VGPR_MSB bank, not the
; destination bank (see the comment on Src0OutReg); the two only diverge when
; vdst and src0 are in different banks.  s_set_vgpr_msb 0x40 puts vdst in bank1
; and src0 in bank0, so src0_out must land in src0's bank-0 register (Vgpr2).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	permlane16_swap_bank_kernel
	.p2align	8
	.type	permlane16_swap_bank_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @permlane16_swap_bank_kernel(
permlane16_swap_bank_kernel:
	global_load_b32 v1, v0, s[0:1]
	global_load_b32 v2, v0, s[2:3]
	s_set_vgpr_msb 0x40
	v_permlane16_swap_b32 v1, v2
	; Partner-lane (XOR 16) byte address shared by both swap halves:
	; CHECK: %pls16_partner{{[0-9]*}} = xor i32 %{{[^,]+}}, 16
	; CHECK: %pls16_addr{{[0-9]*}} = shl i32 %pls16_partner{{[0-9]*}}, 2
	; src0 is read, and src0_out selected, from src0's real bank-0 register:
	; CHECK: %pls16_bperm_src0{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls16_addr{{[0-9]*}}, i32 %Vgpr2{{[._0-9]*}})
	; CHECK: %pls16_new_src0_out{{[0-9]*}} = select i1 %{{[^,]+}}, i32 %pls16_bperm_vdst{{[0-9]*}}, i32 %Vgpr2{{[._0-9]*}}
	; src0_out is written back to Vgpr2: this phi exists only because src0_out
	; targets src0's own bank; a destination-bank regression would drop it.
	; CHECK: %Vgpr2.{{[0-9]+}} = phi i32 [ %pls16_new_src0_out{{[0-9]*}},
	s_set_vgpr_msb 0
	global_store_b32 v0, v1, s[0:1]
	global_store_b32 v0, v2, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel permlane16_swap_bank_kernel
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
    .name:           permlane16_swap_bank_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         permlane16_swap_bank_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
