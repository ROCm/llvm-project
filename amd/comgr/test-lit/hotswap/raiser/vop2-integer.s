; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=vop2_integer \
; RUN:   --target-isa=gfx942 \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=carry_out_kernel \
; RUN:   --target-isa=gfx942 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CARRY-OUT
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=carry_in_kernel \
; RUN:   --target-isa=gfx942 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CARRY-IN
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=cndmask_kernel \
; RUN:   --target-isa=gfx942 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CNDMASK

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	vop2_integer
	.p2align	8
	.type	vop2_integer,@function
; IR-LABEL: define amdgpu_kernel void @vop2_integer(
vop2_integer:
; IR: = add i32 1, {{.+}}
	v_add_u32_e32 v2, 1, v1
; IR: = sub i32 2, {{.+}}
	v_sub_u32_e32 v3, 2, v1
; The subrev opcodes compute src1 - src0, so the constant is the right operand.
; IR: = sub i32 {{.+}}, 3
	v_subrev_u32_e32 v4, 3, v1
; IR: = and i32 4, {{.+}}
	v_and_b32_e32 v5, 4, v1
; IR: = or i32 5, {{.+}}
	v_or_b32_e32 v6, 5, v1
; IR: = xor i32 6, {{.+}}
	v_xor_b32_e32 v7, 6, v1
; IR: [[XNOR_XOR:%.+]] = xor i32 7, {{.+}}
; IR-NEXT: = xor i32 [[XNOR_XOR]], -1
	v_xnor_b32_e32 v8, 7, v1

; The shift amount is masked to the width the hardware reads, because an
; unmasked LLVM shift is poison once the amount reaches the operand width.
; IR: [[SHL_AMT:%.+]] = and i32 {{.+}}, 31
; IR-NEXT: = shl i32 {{.+}}, [[SHL_AMT]]
	v_lshlrev_b32_e32 v9, v0, v1
; IR: [[SHR_AMT:%.+]] = and i32 {{.+}}, 31
; IR-NEXT: = lshr i32 {{.+}}, [[SHR_AMT]]
	v_lshrrev_b32_e32 v10, v0, v1
; IR: [[ASHR_AMT:%.+]] = and i32 {{.+}}, 31
; IR-NEXT: = ashr i32 {{.+}}, [[ASHR_AMT]]
	v_ashrrev_i32_e32 v11, v0, v1

; IR: = call i32 @llvm.smin.i32(i32 8, i32 {{.+}})
	v_min_i32_e32 v12, 8, v1
; IR: = call i32 @llvm.smax.i32(i32 9, i32 {{.+}})
	v_max_i32_e32 v13, 9, v1
; IR: = call i32 @llvm.umin.i32(i32 10, i32 {{.+}})
	v_min_u32_e32 v14, 10, v1
; IR: = call i32 @llvm.umax.i32(i32 11, i32 {{.+}})
	v_max_u32_e32 v15, 11, v1

; The 24-bit multiplies read only the low 24 bits of each source, with the
; signedness the opcode names.
; IR: = trunc i32 {{.+}} to i24
; IR-NEXT: = sext i24 {{.+}} to i32
; IR: = mul i32
	v_mul_i32_i24_e32 v16, v0, v1
; IR: = trunc i32 {{.+}} to i24
; IR-NEXT: = zext i24 {{.+}} to i32
; IR: = mul i32
	v_mul_u32_u24_e32 v17, v0, v1
; The high halves take the 48-bit product apart at 64 bits, shifting the way
; the source signedness demands.
; IR: = sext i24 {{.+}} to i64
; IR: [[HI_WIDE:%.+]] = mul i64
; IR-NEXT: [[HI:%.+]] = ashr i64 [[HI_WIDE]], 32
; IR-NEXT: = trunc i64 [[HI]] to i32
	v_mul_hi_i32_i24_e32 v18, v0, v1
; IR: = zext i24 {{.+}} to i64
; IR: [[HIU_WIDE:%.+]] = mul i64
; IR-NEXT: [[HIU:%.+]] = lshr i64 [[HIU_WIDE]], 32
; IR-NEXT: = trunc i64 [[HIU]] to i32
	v_mul_hi_u32_u24_e32 v19, v0, v1
; IR: ret void
	s_endpgm

; The carry-propagating opcodes read or write VCC as a further operand, which
; this handler does not model.
	.globl	carry_out_kernel
	.p2align	8
	.type	carry_out_kernel,@function
carry_out_kernel:
; CARRY-OUT: unsupported-instruction-form: v_add_co_u32 [VOP2]
	v_add_co_u32_e32 v2, vcc, v0, v1
	s_endpgm

	.globl	carry_in_kernel
	.p2align	8
	.type	carry_in_kernel,@function
carry_in_kernel:
; CARRY-IN: unsupported-instruction-form: v_addc_co_u32 [VOP2]
	v_addc_co_u32_e32 v2, vcc, v0, v1, vcc
	s_endpgm

	.globl	cndmask_kernel
	.p2align	8
	.type	cndmask_kernel,@function
cndmask_kernel:
; CNDMASK: unsupported-instruction-form: v_cndmask_b32 [VOP2]
	v_cndmask_b32_e32 v2, v0, v1, vcc
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop2_integer
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 20
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel carry_out_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel carry_in_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel cndmask_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
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
    .name:           vop2_integer
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vop2_integer.kd
    .vgpr_count:     20
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           carry_out_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         carry_out_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           carry_in_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         carry_in_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cndmask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         cndmask_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
