; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vopc_predicates | %FileCheck %s --check-prefix=PRED

; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vopc_exec_write | %FileCheck %s --check-prefix=EXEC

; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vopc_vop3_encoding 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=VOP3-REFUSE
; VOP3-REFUSE: unsupported-instruction-form: v_cmpx_gt_i32 [VOP3]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopc_predicates
	.p2align	8
	.type	vopc_predicates,@function
; PRED-LABEL: define amdgpu_kernel void @vopc_predicates(
vopc_predicates:
; PRED: icmp slt i32
	v_cmp_lt_i32_e32 vcc_lo, v0, v1
; PRED: icmp eq i32
	v_cmp_eq_i32_e32 vcc_lo, v0, v1
; PRED: icmp sle i32
	v_cmp_le_i32_e32 vcc_lo, v0, v1
; PRED: icmp sgt i32
	v_cmp_gt_i32_e32 vcc_lo, v0, v1
; PRED: icmp ne i32
	v_cmp_ne_i32_e32 vcc_lo, v0, v1
; PRED: icmp sge i32
	v_cmp_ge_i32_e32 vcc_lo, v0, v1
; PRED: icmp ult i32
	v_cmp_lt_u32_e32 vcc_lo, v0, v1
; PRED: icmp eq i32
	v_cmp_eq_u32_e32 vcc_lo, v0, v1
; PRED: icmp ule i32
	v_cmp_le_u32_e32 vcc_lo, v0, v1
; PRED: icmp ugt i32
	v_cmp_gt_u32_e32 vcc_lo, v0, v1
; PRED: icmp ne i32
	v_cmp_ne_u32_e32 vcc_lo, v0, v1
; PRED: icmp uge i32
	v_cmp_ge_u32_e32 vcc_lo, v0, v1
; PRED: ret void
	s_endpgm

	.globl	vopc_exec_write
	.p2align	8
	.type	vopc_exec_write,@function
; EXEC-LABEL: define amdgpu_kernel void @vopc_exec_write(
vopc_exec_write:
; The comparison reaches EXEC as a wave-level ballot, taken at the width of the
; wave the gfx1250 source believes it runs on and so narrowed from the gfx942
; target ballot, and ANDed into the EXEC the raiser is tracking.
; EXEC: [[CMP:%.+]] = icmp sgt i32 {{.+}}, {{.+}}
; EXEC: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; EXEC: [[MASK:%.+]] = trunc i64 [[BALLOT]] to i32
; EXEC: [[NARROWED:%.+]] = and i32 -1, [[MASK]]
	v_cmpx_gt_i32_e32 v0, v1
; The vector write that follows is predicated on the narrowed EXEC, which is
; what makes the store observable: the lane-active bit is recomputed from it
; rather than from the EXEC in force before the comparison.
; EXEC: [[LANE:%.+]] = lshr i32 [[NARROWED]], {{.+}}
; EXEC: [[BIT:%.+]] = and i32 [[LANE]], 1
; EXEC: [[ACTIVE:%.+]] = icmp ne i32 [[BIT]], 0
; EXEC: br i1 [[ACTIVE]]
	v_add_f32_e32 v2, v0, v1
; EXEC: ret void
	s_endpgm

	.globl	vopc_vop3_encoding
	.p2align	8
	.type	vopc_vop3_encoding,@function
vopc_vop3_encoding:
	v_cmpx_gt_i32_e64 v0, v1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopc_predicates
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel vopc_exec_write
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_denorm_mode_32 1
		.amdhsa_float_denorm_mode_16_64 3
	.end_amdhsa_kernel
	.amdhsa_kernel vopc_vop3_encoding
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vopc_predicates
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vopc_predicates.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vopc_exec_write
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vopc_exec_write.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vopc_vop3_encoding
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vopc_vop3_encoding.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
