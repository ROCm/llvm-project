; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=dpp_row_mask_source_wave_clip_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Lower v_mov_b32_dpp row_shr with row_mask via bpermute plus row/bank clip gating to the source wave.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	dpp_row_mask_source_wave_clip_kernel
	.p2align	8
	.type	dpp_row_mask_source_wave_clip_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @dpp_row_mask_source_wave_clip_kernel(
dpp_row_mask_source_wave_clip_kernel:
	v_mov_b32_dpp v1, v1 row_shr:1 row_mask:0x1 bank_mask:0xf bound_ctrl:1
	; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
	; CHECK: %cwd_dpp_within_row = and i32 %{{.+}}, 15
	; CHECK: %cwd_dpp_row_base = and i32 %{{.+}}, -16
	; CHECK: %cwd_dpp_bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector,
	; CHECK: %cwd_dpp_source_lane = and i32 %{{.+}}, 31
	; CHECK: %[[SROW:.+]] = lshr i32 %cwd_dpp_source_lane, 4
	; CHECK: %cwd_dpp_source_row = and i32 %[[SROW]], 3
	; CHECK: %[[SBANK:.+]] = lshr i32 %cwd_dpp_source_lane, 2
	; CHECK: %cwd_dpp_source_bank = and i32 %[[SBANK]], 3
	; CHECK: %cwd_dpp_row_active = icmp ne i32 %{{.+}}, 0
	; CHECK: %cwd_dpp_bank_active = icmp ne i32 %{{.+}}, 0
	; CHECK: %cwd_dpp_lane_active = and i1 %cwd_dpp_row_active, %cwd_dpp_bank_active
	; CHECK: %cwd_dpp_gated = select i1 %cwd_dpp_lane_active, i32 %cwd_dpp_inrange,
	; CHECK: declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)
	ds_store_b32 v0, v1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel dpp_row_mask_source_wave_clip_kernel
		.amdhsa_next_free_vgpr 2
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
    .name:           dpp_row_mask_source_wave_clip_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         dpp_row_mask_source_wave_clip_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
