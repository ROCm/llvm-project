; Cross-lane classifier must treat the WWM/WQM markers as VGPR-safe
; propagators: an update.dpp result flowing through strict.wwm (the
; permlane16 broadcast wrapped by wrapAsWWMValue under MODREP) is rewritten
; to ds_bpermute rather than refused.

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --disable-wave-native --enable-writelane-rewrite \
; RUN:     --emit-ir=dpp_reduction_wwm_chain_kernel \
; RUN:   | %FileCheck %s --check-prefix=MODREP

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --enable-wave-native --enable-writelane-rewrite \
; RUN:     --emit-ir=dpp_reduction_wwm_chain_kernel \
; RUN:   | %FileCheck %s --check-prefix=WAVENATIVE

; MODREP-LABEL: define amdgpu_kernel void @dpp_reduction_wwm_chain_kernel(
; WAVENATIVE-LABEL: define amdgpu_kernel void @dpp_reduction_wwm_chain_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	dpp_reduction_wwm_chain_kernel
	.p2align	8
	.type	dpp_reduction_wwm_chain_kernel,@function
dpp_reduction_wwm_chain_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x14
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v2, s0, s4, v0
	global_load_b32 v0, v2, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	v_mov_b32_dpp v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
; update.dpp is rewritten to a whole-wave ds_bpermute gather (both projections).
; MODREP-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
; MODREP-DAG: %cwd_dpp_bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector, i32 %{{[^)]+}})
; WAVENATIVE-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
; WAVENATIVE: %cwd_dpp_bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector, i32 %{{[^)]+}})
	v_permlane16_b32 v0, v0, 0x76543210, 0x76543210 op_sel:[1,0]
; MODREP wraps the permlane16 broadcast in strict.wwm; WaveNative leaves it bare.
; MODREP-DAG: %permlane16_emu_wwm = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %permlane16_emu)
; WAVENATIVE-NOT: @llvm.amdgcn.strict.wwm
	global_store_b32 v2, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel dpp_reduction_wwm_chain_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
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
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           dpp_reduction_wwm_chain_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         dpp_reduction_wwm_chain_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata

