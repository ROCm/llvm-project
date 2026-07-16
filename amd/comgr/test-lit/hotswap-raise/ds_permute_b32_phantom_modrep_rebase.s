; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=ds_permute_b32_phantom_modrep_rebase_kernel \
; RUN:   | %FileCheck %s
;
; Confirm this kernel takes the phantom-lane MODREP fallback (not WaveNative).
; The regime message is a stderr diagnostic, so merge stderr into the checked stream.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=ds_permute_b32_phantom_modrep_rebase_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PROJ
;
; PROJ: phantom-lane regime
; PROJ-SAME: falling back to ModuloReplicationProjection
;
; A gfx1250 ds_permute_b32 (forward/PUSH) selector is a source-wave-local byte
; offset ((addr / 4) % 32). Lowering wave32 -> wave64 (gfx942), the selector is
; clamped to the source-wave byte range and rebased into the current source-wave
; half before the wave64 llvm.amdgcn.ds.permute; without the rebase, upper-half
; lanes scatter to lanes 0..31 (or out of bounds for a phantom selector). The
; rebase is gated on the cross-widening direction, so it also applies under
; phantom-lane MODREP (max_flat_workgroup_size 32) -- gating on the source-wave
; count instead would skip it here (MODREP reports 1) and regress to the #195
; OOB class. Sibling ds_permute_b32_wave32_rebase.s pins the same rebase under
; WaveNative.
; CHECK-LABEL: define amdgpu_kernel void @ds_permute_b32_phantom_modrep_rebase_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_permute_b32_phantom_modrep_rebase_kernel
	.p2align	8
	.type	ds_permute_b32_phantom_modrep_rebase_kernel,@function
ds_permute_b32_phantom_modrep_rebase_kernel:
	v_mov_b32_e32 v1, v0
	v_and_b32_e32 v2, 31, v0
	v_lshlrev_b32_e32 v2, 2, v2
; Clamp the selector to the source-wave byte range, derive the current source-wave
; byte base, and OR them into the address the wave64 permute scatters with.
; CHECK: [[LOCAL:%[0-9a-zA-Z_.]+]] = and i32 %{{[^,]+}}, 127
; CHECK: [[LANEBASE:%[0-9a-zA-Z_.]+]] = and i32 %{{[^,]+}}, -32
; CHECK: [[BYTEBASE:%[0-9a-zA-Z_.]+]] = shl i32 [[LANEBASE]], 2
; CHECK: [[ADDR:%[0-9a-zA-Z_.]+]] = or i32 [[LOCAL]], [[BYTEBASE]]
; CHECK: call i32 @llvm.amdgcn.ds.permute(i32 [[ADDR]], i32 %{{[^)]+}})
; The rebased address (not the raw clamped local offset) must reach the intrinsic.
; CHECK-NOT: call i32 @llvm.amdgcn.ds.permute(i32 [[LOCAL]],
	ds_permute_b32 v3, v2, v1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_permute_b32_phantom_modrep_rebase_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 32
    .name:           ds_permute_b32_phantom_modrep_rebase_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         ds_permute_b32_phantom_modrep_rebase_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
