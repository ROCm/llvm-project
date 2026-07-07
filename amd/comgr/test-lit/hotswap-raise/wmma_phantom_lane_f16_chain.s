; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_f16_chain_kernel 2>&1 \
; RUN:   | %FileCheck %s

; Phantom-lane wave-size regime chains f16 WMMA (mfma.f32.16x16x16f16) via wwm/bpermute.
; CHECK: phantom-lane regime
; CHECK-SAME: max_flat_workgroup_size=32
; CHECK-SAME: target wavefront width=64
; CHECK: define amdgpu_kernel void @wmma_f16_chain_kernel
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> {{.*}}, <4 x half> {{.*}}, <4 x float> {{.*}}
; CHECK: call <4 x float> @llvm.amdgcn.strict.wwm.v4f32(<4 x float>
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 {{.*}}, i32 {{.*}}
; CHECK: call i32 @llvm.amdgcn.strict.wwm.i32(i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_f16_chain_kernel
	.p2align	8
	.type	wmma_f16_chain_kernel,@function
wmma_f16_chain_kernel:                  ; @wmma_f16_chain_kernel
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 5, v0
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_load_b256 s[12:19], s[4:5], 0x0
	s_load_b256 s[20:27], s[6:7], 0x0
	s_load_b256 s[36:43], s[8:9], 0x0
	s_load_b256 s[44:51], s[10:11], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[2:3], s[12:13]
	v_mov_b64_e32 v[10:11], s[20:21]
	v_mov_b64_e32 v[4:5], s[14:15]
	v_mov_b64_e32 v[6:7], s[16:17]
	v_mov_b64_e32 v[8:9], s[18:19]
	v_mov_b64_e32 v[12:13], s[22:23]
	v_mov_b64_e32 v[14:15], s[24:25]
	v_mov_b64_e32 v[16:17], s[26:27]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_f16 v[18:25], v[2:9], v[10:17], 0
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[2:3], s[36:37]
	v_mov_b64_e32 v[10:11], s[44:45]
	v_mov_b64_e32 v[4:5], s[38:39]
	v_mov_b64_e32 v[6:7], s[40:41]
	v_mov_b64_e32 v[8:9], s[42:43]
	v_mov_b64_e32 v[12:13], s[46:47]
	v_mov_b64_e32 v[14:15], s[48:49]
	v_mov_b64_e32 v[16:17], s[50:51]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_f16 v[18:25], v[2:9], v[10:17], v[18:25]
	s_clause 0x1
	global_store_b128 v0, v[22:25], s[0:1] offset:16
	global_store_b128 v0, v[18:21], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_f16_chain_kernel
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 26
		.amdhsa_next_free_sgpr 52
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 32
    .name:           wmma_f16_chain_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_f16_chain_kernel.kd
    .vgpr_count:     26
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
