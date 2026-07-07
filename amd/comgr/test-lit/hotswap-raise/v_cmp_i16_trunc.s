; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmp_i16_trunc_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_cmp_lt_i16 signed i16 compare (trunc operands)+ballot lift.
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_i16_trunc_kernel(
; CHECK: [[SRC:%[A-Za-z0-9_.]+]] = trunc i32 {{.*}} to i16
; CHECK: [[CMP:%[^ ]+]] = icmp slt i16 -1, [[SRC]]
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cmp_i16_trunc_kernel
	.p2align	8
	.type	v_cmp_i16_trunc_kernel,@function
v_cmp_i16_trunc_kernel:
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, 0xbfce
	v_cmp_lt_i16_e64 s4, -1, v1
	v_cndmask_b32_e64 v2, 0, 1, s4
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cmp_i16_trunc_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
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
    .max_flat_workgroup_size: 32
    .name: v_cmp_i16_trunc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 5
    .symbol: v_cmp_i16_trunc_kernel.kd
    .vgpr_count: 3
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
