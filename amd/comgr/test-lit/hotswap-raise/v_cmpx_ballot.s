; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --enable-wave-native \
; RUN:     --emit-ir=v_cmpx_ballot_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_cmpx wave-mask ballot + exec update (--enable-wave-native).
; CHECK-LABEL: define amdgpu_kernel void @v_cmpx_ballot_kernel(
; CHECK:      %[[CMPX_CMP:[^ ]+]] = icmp ult i32 %{{[^ ,]+}}, 64
; CHECK-NEXT: %cmpx_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %[[CMPX_CMP]])
; CHECK-NEXT: %cmpx_exec = and i64 {{[^,]+}}, %cmpx_ballot
; CHECK:      %[[VCMP:[^ ]+]] = icmp ult i32 %{{[^ ,]+}}, 96
; CHECK-NEXT: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %[[VCMP]])
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32
; CHECK-NOT: sext i1 %{{[^ ]+}} to i32
; CHECK-NOT: trunc i64 %cmpx_ballot to i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cmpx_ballot_kernel
	.p2align	8
	.type	v_cmpx_ballot_kernel,@function
v_cmpx_ballot_kernel:                   ; @v_cmpx_ballot_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0xcc
	v_cmpx_lt_u32_e64 v0, 64
	v_mov_b32 v1, 0xAA
	s_mov_b32 exec_lo, -1
	v_cmp_lt_u32_e64 s4, v0, 96
	
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cmpx_ballot_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 5
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
    .name:           v_cmpx_ballot_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         v_cmpx_ballot_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
