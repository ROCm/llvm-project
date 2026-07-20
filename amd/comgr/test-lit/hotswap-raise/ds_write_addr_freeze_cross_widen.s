; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=ds_write_addr_freeze_kernel \
; RUN:   | %FileCheck %s --check-prefixes=WN,BOTH
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:   --emit-ir=ds_write_addr_freeze_kernel \
; RUN:   | %FileCheck %s --check-prefixes=MR,BOTH
;
; A cross-widening (wave32 -> wave64) LDS-store address is frozen before the
; addrspace(3) pointer is materialised, so an inactive-lane undef index cannot
; reach the ds_write as poison (rocm-systems#157 sibling of the global-store
; fault; see RaiseContext::freezeMemAddr).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_write_addr_freeze_kernel
	.p2align	8
	.type	ds_write_addr_freeze_kernel,@function
; WN-LABEL: define amdgpu_kernel void @ds_write_addr_freeze_kernel(
; WN: call i1 @llvm.amdgcn.init.whole.wave()
; MR-LABEL: define amdgpu_kernel void @ds_write_addr_freeze_kernel(
; BOTH: %mem_addr_frozen = freeze i64
; BOTH: [[P:%.+]] = inttoptr i64 %mem_addr_frozen to ptr addrspace(3)
; BOTH: store i32 42, ptr addrspace(3) [[P]]
ds_write_addr_freeze_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 32
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	; divergent per-lane LDS byte index: inactive lanes never write v1
	v_cndmask_b32_e32 v1, v2, v3, vcc_lo
	v_mov_b32_e32 v4, 0x2a
	ds_store_b32 v1, v4
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_write_addr_freeze_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 1
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
  - .args:          []
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 128
    .name:           ds_write_addr_freeze_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         ds_write_addr_freeze_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
