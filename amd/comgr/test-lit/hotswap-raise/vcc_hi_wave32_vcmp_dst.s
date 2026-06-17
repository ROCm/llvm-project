; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vcmp_scratch_dst_kernel | %FileCheck %s
;
; A wave32 V_CMP naming vcc_hi / exec_hi as its destination must deposit the
; ballot into that scratch slot, not the real VCC / EXEC, so a downstream
; consumer reads a real wave mask. The consumer side is
; vcc_hi_wave32_cndmask_cond.s. See ParsedReg::VCC_HI_SCRATCH.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vcmp_scratch_dst_kernel
	.p2align	8
	.type	vcmp_scratch_dst_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @vcmp_scratch_dst_kernel(
vcmp_scratch_dst_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
; The vcc_hi compare is balloted into the vcc_hi scratch:
; CHECK: %vcmp = icmp
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1 %vcmp)
	v_cmp_gt_i64_e64 vcc_hi, s[4:5], s[6:7]
; and the cndmask consuming vcc_hi selects on the per-lane bit extracted from it:
; CHECK: %cndmask = select i1 %wn_mask_lane_i1{{[0-9]*}}, i32 {{.*}}, i32
	v_cndmask_b32 v5, v0, v1, vcc_hi
; The same routing applies to the exec_hi scratch slot:
; CHECK: icmp
; CHECK: call i64 @llvm.amdgcn.ballot.i64(
; CHECK: select i1 %wn_mask_lane_i1{{[0-9]*}}, i32 {{.*}}, i32
	v_cmp_gt_i64_e64 exec_hi, s[4:5], s[6:7]
	v_cndmask_b32 v6, v0, v1, exec_hi
	ds_store_b32 v7, v5
	ds_store_b32 v7, v6
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vcmp_scratch_dst_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           16
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 16
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           vcmp_scratch_dst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         vcmp_scratch_dst_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
