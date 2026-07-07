; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=mbcnt_cmpx_exec_elect_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WN
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=mbcnt_cmpx_exec_elect_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=MODREP
;
; Wave32 single-lane elect: mbcnt(exec) == 0 updates EXEC. WaveNative must
; slice the current source-wave EXEC for mbcnt, then ballot V_CMPX to i64.
; MODREP keeps refusing this C4 shape.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	mbcnt_cmpx_exec_elect_kernel
	.p2align	8
	.type	mbcnt_cmpx_exec_elect_kernel,@function
; WN-LABEL: define amdgpu_kernel void @mbcnt_cmpx_exec_elect_kernel(
mbcnt_cmpx_exec_elect_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
; WN: %exec_srcwave_mask_base = and i32 %{{[^,]+}}, -32
; WN: %exec_srcwave_mask_at_srcwave = lshr i64 %{{[^,]+}}, %exec_srcwave_mask_shift
; WN: %exec_srcwave_mask = trunc i64 %exec_srcwave_mask_at_srcwave to i32
; WN: %mbcnt_masked{{[0-9]*}} = and i32 %exec_srcwave_mask, %mbcnt_below_mask{{[0-9]*}}
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	v_mbcnt_hi_u32_b32 v1, exec_hi, v1
; WN: %[[MBCNT_VGPR:.*]] = phi i32 [ %mbcnt_lo_srcwave{{[0-9]*}},
; WN: %[[CMP:.*]] = icmp eq i32 %[[MBCNT_VGPR]], 0
; WN-NEXT: %cmpx_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %[[CMP]])
; WN-NEXT: %cmpx_exec = and i64 {{[^,]+}}, %cmpx_ballot
; MODREP: cross-wave-lane-predicated-exec
; MODREP: CmpxFromLaneId
; MODREP: outcome: (c) refuse
	v_cmpx_eq_u32_e64 v1, 0
; WN: lshr i64 %cmpx_exec,
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_cmpx_exec_elect_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mbcnt_cmpx_exec_elect_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         mbcnt_cmpx_exec_elect_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
