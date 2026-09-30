; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=binary_float_kernel,scc_kernel \
; RUN:   | %FileCheck %s
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=unhandled_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: s_min_num_f32

; Each opcode carries its own mode guard, so each needs its own kernel: the
; raise stops at the first refusal.
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=round_mul_kernel,round_fmac_kernel,round_fmaak_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ROUND
; ROUND:      unsupported-floating-point-mode: s_mul_f32 [SOP2]
; ROUND-SAME: f32 rounding mode 1 is unsupported
; ROUND:      unsupported-floating-point-mode: s_fmac_f32 [SOP2]
; ROUND-SAME: f32 rounding mode 1 is unsupported
; ROUND:      unsupported-floating-point-mode: s_fmaak_f32 [SOP2]
; ROUND-SAME: f32 rounding mode 1 is unsupported

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	binary_float_kernel
	.p2align	8
	.type	binary_float_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @binary_float_kernel(
binary_float_kernel:
; CHECK: [[MUL_A:%.+]] = bitcast i32 {{.+}} to float
; CHECK-NEXT: [[MUL_B:%.+]] = bitcast i32 {{.+}} to float
; CHECK-NEXT: [[MUL:%.+]] = fmul float [[MUL_A]], [[MUL_B]]
; CHECK-NEXT: [[MUL_BITS:%.+]] = bitcast float [[MUL]] to i32
	s_mul_f32 s2, s0, s1
; CHECK: [[FMAC_A:%.+]] = bitcast i32 {{.+}} to float
; CHECK-NEXT: [[FMAC_B:%.+]] = bitcast i32 {{.+}} to float
; CHECK-NEXT: [[FMAC_ACC:%.+]] = bitcast i32 [[MUL_BITS]] to float
; CHECK-NEXT: [[FMAC:%.+]] = call float @llvm.fma.f32(float [[FMAC_A]], float [[FMAC_B]], float [[FMAC_ACC]])
; CHECK-NEXT: bitcast float [[FMAC]] to i32
	s_fmac_f32 s2, s3, s4
; CHECK: [[AAK_A:%.+]] = bitcast i32 {{.+}} to float
; CHECK-NEXT: [[AAK_B:%.+]] = bitcast i32 {{.+}} to float
; CHECK-NEXT: [[AAK:%.+]] = call float @llvm.fma.f32(float [[AAK_A]], float [[AAK_B]], float f0x40490FDB)
; CHECK-NEXT: bitcast float [[AAK]] to i32
	s_fmaak_f32 s2, s0, s1, 0x40490fdb
	s_endpgm

	.globl	scc_kernel
	.p2align	8
	.type	scc_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @scc_kernel(
scc_kernel:
; The scalar float operations preserve the SCC value produced by s_not_b32.
	s_mov_b32 s2, 7
; CHECK: [[NOT:%.+]] = xor i32 {{.+}}, -1
; CHECK: [[SCC:%.+]] = icmp ne i32 [[NOT]], 0
	s_not_b32 s0, s1
	s_mul_f32 s3, s0, s0
	s_fmac_f32 s3, s0, s0
	s_fmaak_f32 s3, s0, s0, 0x3f800000
; CHECK: select i1 [[SCC]], i32 {{.+}}, i32 7
	s_cmov_b32 s2, s3
	s_endpgm

	.globl	unhandled_kernel
	.p2align	8
	.type	unhandled_kernel,@function
unhandled_kernel:
	s_min_num_f32 s2, s0, s1
	s_endpgm

	.globl	round_mul_kernel
	.p2align	8
	.type	round_mul_kernel,@function
round_mul_kernel:
	s_mul_f32 s2, s0, s1
	s_endpgm

	.globl	round_fmac_kernel
	.p2align	8
	.type	round_fmac_kernel,@function
round_fmac_kernel:
	s_fmac_f32 s2, s0, s1
	s_endpgm

	.globl	round_fmaak_kernel
	.p2align	8
	.type	round_fmaak_kernel,@function
round_fmaak_kernel:
	s_fmaak_f32 s2, s0, s1, 0x3f800000
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel binary_float_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel scc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.amdhsa_kernel unhandled_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.amdhsa_kernel round_mul_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
		.amdhsa_float_round_mode_32 1
	.end_amdhsa_kernel
	.amdhsa_kernel round_fmac_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
		.amdhsa_float_round_mode_32 1
	.end_amdhsa_kernel
	.amdhsa_kernel round_fmaak_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
		.amdhsa_float_round_mode_32 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           binary_float_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         binary_float_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           scc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         scc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unhandled_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         unhandled_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           round_mul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         round_mul_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           round_fmac_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         round_fmac_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           round_fmaak_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         round_fmaak_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
