; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_store_short_sentinel_srd_kernel,buffer_store_short_finite_srd_kernel,buffer_store_short_allones_srd_kernel,buffer_store_short_ambiguous_srd_kernel 2>/dev/null | %FileCheck %s

; Buffer store b16 SRD stride/num-records word reconstruction.
; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_sentinel_srd_kernel(
; CHECK: icmp eq i32 {{.*}}16777215
; CHECK: select i1 {{.*}}, i32 2147483646, i32 16777215
; CHECK-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; CHECK: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(
; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_finite_srd_kernel(
; CHECK: icmp eq i32 {{.*}}16777215
; CHECK: select i1 {{.*}}, i32 2147483646, i32 4096
; CHECK-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; CHECK: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(
; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_allones_srd_kernel(
; CHECK: icmp eq i32 {{.*}}16777215
; CHECK: select i1 {{.*}}, i32 2147483646, i32 {{(-1|4294967295)}}
; CHECK-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; CHECK: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(
; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_ambiguous_srd_kernel(
; CHECK: icmp eq i32 1, 0
; CHECK: icmp eq i32 1, 131072
; CHECK: icmp eq i32 1, 147456
; CHECK: icmp eq i32 1, 159744
; CHECK: select i1 {{.*}}, i32 2147483646, i32 16777215
; CHECK-NOT: select i1 true, i32 2147483646, i32 16777215
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	buffer_store_short_sentinel_srd_kernel
	.p2align	8
	.type	buffer_store_short_sentinel_srd_kernel,@function
buffer_store_short_sentinel_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_or_b32 s1, s1, 0xfc000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_finite_srd_kernel
	.p2align	8
	.type	buffer_store_short_finite_srd_kernel,@function
buffer_store_short_finite_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 0
	s_mov_b32 s2, 4096
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_allones_srd_kernel
	.p2align	8
	.type	buffer_store_short_allones_srd_kernel,@function
buffer_store_short_allones_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 0
	s_mov_b32 s2, -1
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_ambiguous_srd_kernel
	.p2align	8
	.type	buffer_store_short_ambiguous_srd_kernel,@function
buffer_store_short_ambiguous_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 1
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_short_sentinel_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_finite_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_allones_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_ambiguous_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_sentinel_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_sentinel_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_finite_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_finite_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_allones_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_allones_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_ambiguous_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_ambiguous_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
