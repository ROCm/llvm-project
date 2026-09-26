; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=exec_destination 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=EXEC-DESTINATION
; EXEC-DESTINATION: unsupported comparison mask destination

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=m0_destination 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=M0-DESTINATION
; M0-DESTINATION: unsupported comparison mask destination

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=vcc_hi_destination 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=VCC-HI-DESTINATION
; VCC-HI-DESTINATION: unsupported comparison mask destination

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=exec_hi_destination 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=EXEC-HI-DESTINATION
; EXEC-HI-DESTINATION: unsupported comparison mask destination

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=trap_destination 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=TRAP-DESTINATION
; TRAP-DESTINATION: unsupported comparison mask destination

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=dpp_vopc 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=DPP-VOPC
; DPP-VOPC: unsupported-instruction-form: v_cmp_eq_u32

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=dpp_vop3 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=DPP-VOP3
; DPP-VOP3: unsupported-instruction-form: v_cmp_eq_u32

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=u16 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=U16
; U16: unsupported-instruction-form: v_cmp_eq_u16

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=i64 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=I64
; I64: unsupported-instruction-form: v_cmpx_lt_i64

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=u64 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=U64
; U64: unsupported-instruction-form: v_cmp_ne_u64

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=float_modifier 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=FLOAT-MODIFIER
; FLOAT-MODIFIER: integer source modifiers are not supported

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=float_class 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=FLOAT-CLASS
; FLOAT-CLASS: unsupported-instruction-form: v_cmp_class_f32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6

.macro comparison_kernel name
	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel \name
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.text
	.globl \name
	.p2align 8
	.type \name,@function
\name:
.endm

	comparison_kernel exec_destination
	v_cmp_eq_u32_e64 exec_lo, v0, v1
	s_endpgm

	comparison_kernel m0_destination
	v_cmp_eq_u32_e64 m0, v0, v1
	s_endpgm

	comparison_kernel vcc_hi_destination
	v_cmp_eq_u32_e64 vcc_hi, v0, v1
	s_endpgm

	comparison_kernel exec_hi_destination
	v_cmp_eq_u32_e64 exec_hi, v0, v1
	s_endpgm

	comparison_kernel trap_destination
	v_cmp_eq_u32_e64 ttmp0, v0, v1
	s_endpgm

	comparison_kernel dpp_vopc
	v_cmp_eq_u32_dpp vcc_lo, v0, v1 quad_perm:[1,0,3,2]
	s_endpgm

	comparison_kernel dpp_vop3
	v_cmp_eq_u32_e64_dpp s4, v0, v1 quad_perm:[1,0,3,2]
	s_endpgm

	comparison_kernel u16
	v_cmp_eq_u16_e64 s4, v0, v1
	s_endpgm

	comparison_kernel i64
	v_cmpx_lt_i64_e64 v[0:1], v[2:3]
	s_endpgm

	comparison_kernel u64
	v_cmp_ne_u64_e32 vcc_lo, v[0:1], v[2:3]
	s_endpgm

	comparison_kernel float_modifier
	v_cmp_lt_f32_e64 s4, -v0, v1
	s_endpgm

	comparison_kernel float_class
	v_cmp_class_f32_e32 vcc_lo, v0, v1
	s_endpgm

	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: exec_destination
    .symbol: exec_destination.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: m0_destination
    .symbol: m0_destination.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: vcc_hi_destination
    .symbol: vcc_hi_destination.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: exec_hi_destination
    .symbol: exec_hi_destination.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: trap_destination
    .symbol: trap_destination.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: dpp_vopc
    .symbol: dpp_vopc.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: dpp_vop3
    .symbol: dpp_vop3.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: u16
    .symbol: u16.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: i64
    .symbol: i64.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: u64
    .symbol: u64.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: float_modifier
    .symbol: float_modifier.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
  - .name: float_class
    .symbol: float_class.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 4
    .max_flat_workgroup_size: 1024
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
