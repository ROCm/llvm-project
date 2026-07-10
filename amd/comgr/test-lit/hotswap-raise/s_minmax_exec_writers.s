; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_minmax_exec_writers_kernel 2>&1 \
; RUN:   | %FileCheck %s

; Scalar min/max ops can legally write EXEC_LO as an explicit scalar
; destination. They are safe for SPE only if their handlers route that explicit
; EXEC write through storeExec instead of bypassing the EXEC alloca.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_minmax_exec_writers_kernel
	.p2align	8
	.type	s_minmax_exec_writers_kernel,@function
; CHECK-NOT: SPE-unmodeled-EXEC-writer
; CHECK-LABEL: define amdgpu_kernel void @s_minmax_exec_writers_kernel(
s_minmax_exec_writers_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	s_mov_b32 s2, 0x0000ffff
	s_mov_b32 s3, exec_lo
; CHECK: %smin = select i1
	s_min_u32 exec_lo, s2, s3
	s_mov_b32 exec_lo, s3
; CHECK: %smax = select i1
	s_max_u32 exec_lo, s2, s3
	s_mov_b32 exec_lo, s3
; CHECK: %smin{{[0-9]*}} = select i1
	s_min_i32 exec_lo, s2, s3
	s_mov_b32 exec_lo, s3
; CHECK: %smax{{[0-9]*}} = select i1
	s_max_i32 exec_lo, s2, s3
; CHECK: %wn_lane_active = icmp ne i64 %wn_exec_bit, 0
	v_mov_b32_e32 v1, 1
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_minmax_exec_writers_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: s_minmax_exec_writers_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 4
    .symbol: s_minmax_exec_writers_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
