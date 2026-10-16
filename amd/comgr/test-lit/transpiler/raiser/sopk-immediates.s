; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx950 --emit-ir=sopk_immediates \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=unsupported_sopk 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	sopk_immediates
	.p2align	8
	.type	sopk_immediates,@function
; IR-LABEL: define amdgpu_kernel void @sopk_immediates(
sopk_immediates:
	; IR: mul i32 {{.+}}, -2
	s_mulk_i32 s0, 0xfffe
	; IR: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 {{.+}}, i32 -1)
	s_addk_i32 s0, 0xffff
	; IR: icmp eq i32 {{.+}}, -1
	s_cmpk_eq_i32 s0, 0xffff
	; IR: icmp ne i32 {{.+}}, -1
	s_cmpk_lg_i32 s0, 0xffff
	; IR: icmp sgt i32 {{.+}}, -1
	s_cmpk_gt_i32 s0, 0xffff
	; IR: icmp sge i32 {{.+}}, -1
	s_cmpk_ge_i32 s0, 0xffff
	; IR: icmp slt i32 {{.+}}, -1
	s_cmpk_lt_i32 s0, 0xffff
	; IR: icmp sle i32 {{.+}}, -1
	s_cmpk_le_i32 s0, 0xffff
	; IR: icmp eq i32 {{.+}}, 65535
	s_cmpk_eq_u32 s0, 0xffff
	; IR: icmp ne i32 {{.+}}, 65535
	s_cmpk_lg_u32 s0, 0xffff
	; IR: icmp ugt i32 {{.+}}, 65535
	s_cmpk_gt_u32 s0, 0xffff
	; IR: icmp uge i32 {{.+}}, 65535
	s_cmpk_ge_u32 s0, 0xffff
	; IR: icmp ult i32 {{.+}}, 65535
	s_cmpk_lt_u32 s0, 0xffff
	; IR: icmp ule i32 {{.+}}, 65535
	s_cmpk_le_u32 s0, 0xffff
	v_nop
	v_nop_e64
	s_movk_i32 s0, 0xffff
	; IR: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 -1, i32 1)
	s_addk_i32 s0, 1
	; IR: ret void
	s_endpgm

	.globl	unsupported_sopk
	.p2align	8
	.type	unsupported_sopk,@function
unsupported_sopk:
	; REFUSE: unsupported-instruction-form
	; REFUSE-SAME: s_cmovk_i32
	s_cmovk_i32 s0, 1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sopk_immediates
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel unsupported_sopk
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
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
    .name:           sopk_immediates
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sopk_immediates.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unsupported_sopk
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         unsupported_sopk.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
