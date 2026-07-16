; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --disable-wave-native --enable-writelane-rewrite \
; RUN:   --emit-ir=readlane_exec_inactive_wwm_kernel \
; RUN:   | %FileCheck %s --check-prefix=MODREP

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --enable-wave-native --enable-writelane-rewrite \
; RUN:   --emit-ir=readlane_exec_inactive_wwm_kernel \
; RUN:   | %FileCheck %s --check-prefix=WAVENATIVE

; `rewriteReadlaneCall` in amd/comgr/src/hotswap/rewrite-cross-lane-divergent.cpp
; lowers `v_readlane_b32` to a `ds_bpermute` gather. Unless the projection
; guarantees hardware EXEC = -1 kernel-wide, the gather is wrapped in
; `strict.wwm` so a read of an EXEC-inactive source lane still returns that
; lane's real VGPR instead of 0.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl readlane_exec_inactive_wwm_kernel
	.p2align 8
	.type readlane_exec_inactive_wwm_kernel,@function
; MODREP-LABEL: define amdgpu_kernel void @readlane_exec_inactive_wwm_kernel(
; WAVENATIVE-LABEL: define amdgpu_kernel void @readlane_exec_inactive_wwm_kernel(
readlane_exec_inactive_wwm_kernel: ; @readlane_exec_inactive_wwm_kernel
; %bb.0:
	s_load_b32 s2, s[0:1], 0x14
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x0
	s_add_co_i32 s3, s3, 1
	s_and_b32 s4, ttmp6, 15
	s_mul_i32 s3, ttmp9, s3
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s4, s4, s3
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s3, ttmp9, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v1, s3, s2, v0
	v_readlane_b32 s2, v1, 0
; MODREP: [[BPERM:%.+]] = call i32 @llvm.amdgcn.ds.bpermute
; MODREP: call i32 @llvm.amdgcn.strict.wwm.i32(i32 [[BPERM]])
; WAVENATIVE: call i32 @llvm.amdgcn.ds.bpermute
; WAVENATIVE-NOT: @llvm.amdgcn.strict.wwm
	v_mov_b32_e32 v0, s2
	global_store_b32 v1, v0, s[0:1] scale_offset
	s_endpgm
	.section .rodata,"a",@progbits
	.p2align 6, 0x0
	.amdhsa_kernel readlane_exec_inactive_wwm_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
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
      - .address_space:   global
        .offset:          0
        .size:            8
        .value_kind:      global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           readlane_exec_inactive_wwm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         readlane_exec_inactive_wwm_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
