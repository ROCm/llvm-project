; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c4_mbcnt_unrelated_exec_kernel 2>&1 \
; RUN:   | %FileCheck %s

; Accept mbcnt/ds_bpermute whose exec use is unrelated (no Class 4 cross-wave-lane-predicated-exec refusal); src-wave rebase.
; CHECK-NOT: cross-wave-lane-predicated-exec
; CHECK-LABEL: define amdgpu_kernel void @c4_mbcnt_unrelated_exec_kernel(
; CHECK: %bperm_srcwave_addr{{[0-9]*}} = or i32 %bperm_local_addr{{[0-9]*}}, %bperm_srcwave_byte_base{{[0-9]*}}
; CHECK: %bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %bperm_srcwave_addr{{[0-9]*}}, i32 %{{[^)]+}})

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c4_mbcnt_unrelated_exec_kernel
	.p2align	8
	.type	c4_mbcnt_unrelated_exec_kernel,@function
c4_mbcnt_unrelated_exec_kernel:
	s_mov_b32 s0, 64
	v_cmpx_gt_i32_e64 s0, v0
	s_cbranch_execz .Ldone
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_xor_b32_e32 v2, 1, v1
	v_cmp_gt_i32_e32 vcc_lo, 32, v2
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_lshlrev_b32_e32 v1, 2, v1
	v_mov_b32_e32 v2, v0
	ds_bpermute_b32 v3, v1, v2
	s_wait_dscnt 0x0
.Ldone:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c4_mbcnt_unrelated_exec_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
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
    .max_flat_workgroup_size: 1024
    .name:           c4_mbcnt_unrelated_exec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         c4_mbcnt_unrelated_exec_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
