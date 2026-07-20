; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=global_store_addr_freeze_kernel \
; RUN:   | %FileCheck %s --check-prefixes=WN,BOTH
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:   --emit-ir=global_store_addr_freeze_kernel \
; RUN:   | %FileCheck %s --check-prefixes=MR,BOTH
;
; A cross-widening (wave32 -> wave64) global-store address is frozen before the
; pointer is materialised, so an inactive-lane undef address cannot reach the
; store as poison (rocm-systems#157; see RaiseContext::freezeMemAddr). The
; v_cndmask below leaves v1 undef on inactive lanes, the undef-inactive-arm phi
; the freeze neutralises.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_store_addr_freeze_kernel
	.p2align	8
	.type	global_store_addr_freeze_kernel,@function
; WN-LABEL: define amdgpu_kernel void @global_store_addr_freeze_kernel(
; WN: call i1 @llvm.amdgcn.init.whole.wave()
; MR-LABEL: define amdgpu_kernel void @global_store_addr_freeze_kernel(
; The store address is frozen before inttoptr, and the store stays gated.
; BOTH: %mem_addr_frozen = freeze i64
; BOTH: [[P:%.+]] = inttoptr i64 %mem_addr_frozen to ptr addrspace(1)
; BOTH: store i32 42, ptr addrspace(1) [[P]]
global_store_addr_freeze_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 16
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	; divergent per-lane address low half: inactive lanes never write v1
	v_cndmask_b32_e32 v1, v2, v3, vcc_lo
	v_mov_b32_e32 v4, 0x2a
	s_wait_kmcnt 0x0
	global_store_b32 v1, v4, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_store_addr_freeze_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 4
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
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 128
    .name:           global_store_addr_freeze_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         global_store_addr_freeze_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
