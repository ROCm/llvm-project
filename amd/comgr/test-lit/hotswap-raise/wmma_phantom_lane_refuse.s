; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_phantom_lane_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s

; Phantom-lane WMMA regime refuses and falls back to modulo-replication.
; CHECK: phantom-lane regime
; CHECK-SAME: max_flat_workgroup_size=32
; CHECK-SAME: target wavefront width=64
; CHECK-SAME: falling back to ModuloReplicationProjection

; CHECK: define amdgpu_kernel void @wmma_phantom_lane_refuse_kernel
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float {{.*}}, float {{.*}}, <4 x float> {{.*}}, i32 0, i32 0, i32 0)
; CHECK: call <4 x float> @llvm.amdgcn.strict.wwm.v4f32(<4 x float>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_phantom_lane_refuse_kernel
	.p2align	8
	.type	wmma_phantom_lane_refuse_kernel,@function
wmma_phantom_lane_refuse_kernel:        ; @wmma_phantom_lane_refuse_kernel
; %bb.0:
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[12:13], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_load_b64 s[14:15], s[8:9], 0x0
	s_load_b64 s[16:17], s[10:11], 0x0
	s_load_b256 s[0:7], s[12:13], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[8:9], s[14:15]
	v_mov_b64_e32 v[10:11], s[16:17]
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[10:11], v[0:7]
	v_mov_b32_e32 v8, 0
	s_clause 0x1
	global_store_b128 v8, v[4:7], s[12:13] offset:16
	global_store_b128 v8, v[0:3], s[12:13]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_phantom_lane_refuse_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 18
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name:           wmma_phantom_lane_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .symbol:         wmma_phantom_lane_refuse_kernel.kd
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
