; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=dpp_row_mask_source_wave_clip_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Wave32 source raised to a wave64 target.  Exercises the DPP
; row_mask/bank_mask source-wave clip in rewriteUpdateDppI32Call
; (rewrite-cross-lane-divergent.cpp): under widening one target wave hosts
; two source waves (lanes 0..31 and 32..63), so the write-gate row/bank must
; be derived from `LaneId & (SourceWaveSize - 1)` rather than the raw target
; LaneId; the source-fetch within-row/row-base stay target-physical by ISA
; definition.  A non-0xF row_mask drives the gating path.  Per-instruction
; CHECKs are inline in the kernel body below.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	dpp_row_mask_source_wave_clip_kernel
	.p2align	8
	.type	dpp_row_mask_source_wave_clip_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @dpp_row_mask_source_wave_clip_kernel(
dpp_row_mask_source_wave_clip_kernel:
	v_mov_b32_dpp v1, v1 row_shr:1 row_mask:0x1 bank_mask:0xf bound_ctrl:1
	; The faithful-lift update.dpp call must not survive the rewrite:
	; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
	;
	; Source-fetch side stays target-physical (raw LaneId):
	; CHECK: %cwd_dpp_within_row = and i32 %{{.+}}, 15
	; CHECK: %cwd_dpp_row_base = and i32 %{{.+}}, -16
	; CHECK: %cwd_dpp_bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector,
	;
	; Destination-gate side clips LaneId to the source-wave (wave32 => 31),
	; then derives the source-wave row/bank from the clipped lane:
	; CHECK: %cwd_dpp_source_lane = and i32 %{{.+}}, 31
	; CHECK: %[[SROW:.+]] = lshr i32 %cwd_dpp_source_lane, 4
	; CHECK: %cwd_dpp_source_row = and i32 %[[SROW]], 3
	; CHECK: %[[SBANK:.+]] = lshr i32 %cwd_dpp_source_lane, 2
	; CHECK: %cwd_dpp_source_bank = and i32 %[[SBANK]], 3
	;
	; The row/bank-active predicates gate DppVal vs OldVal:
	; CHECK: %cwd_dpp_row_active = icmp ne i32 %{{.+}}, 0
	; CHECK: %cwd_dpp_bank_active = icmp ne i32 %{{.+}}, 0
	; CHECK: %cwd_dpp_lane_active = and i1 %cwd_dpp_row_active, %cwd_dpp_bank_active
	; CHECK: %cwd_dpp_gated = select i1 %cwd_dpp_lane_active, i32 %cwd_dpp_inrange,
	;
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
