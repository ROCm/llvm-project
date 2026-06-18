; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=mbcnt_hi_source_wave_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Regression guard for wave32->wave64 lifting of v_mbcnt_hi_u32_b32.
; The hi-half mask is empty for all source lanes 0..31, making mbcnt_hi
; a pass-through of src1. The lifted IR must not contain a user-level
; mbcnt.hi call (the raiser's own lane-ID helpers do emit mbcnt.hi, but
; those use a distinct SSA name that does not match the CHECK-NOT pattern).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	mbcnt_hi_source_wave_kernel
	.p2align	8
	.type	mbcnt_hi_source_wave_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @mbcnt_hi_source_wave_kernel(
mbcnt_hi_source_wave_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	; v_mbcnt_lo lifts to an add.
	; CHECK: %mbcnt_lo_srcwave{{[0-9]*}} = add i32 %mbcnt_pop{{[0-9]*}}, 0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	; v_mbcnt_hi is a source-wave pass-through here, not a target mbcnt.hi.
	; CHECK-NOT: %mbcnt_hi{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.hi
	v_mbcnt_hi_u32_b32 v1, -1, v1
	; The pass-through value flows through the lane-active phi into the store.
	; CHECK: phi i32 [ %mbcnt_lo_srcwave{{[0-9]*}},
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_hi_source_wave_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mbcnt_hi_source_wave_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         mbcnt_hi_source_wave_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
