; REQUIRES: comgr-has-hotswap-transpile

; The scalar float opcodes exist from gfx12 on, so this fixture is gfx1250.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=scalar_float_kernel,scc_kernel \
; RUN:   | %FileCheck %s

; s_rfe_i64 has no lowering, and SOP1 refuses an opcode it does not lift rather
; than letting it through unlowered.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: s_rfe_i64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	scalar_float_kernel
	.p2align	8
	.type	scalar_float_kernel,@function
; Each opcode reads the register the one before it wrote, so the checks chain
; through: an arm that drops its write or picks up the wrong source breaks the
; chain at the next opcode.
; CHECK-LABEL: define amdgpu_kernel void @scalar_float_kernel(
scalar_float_kernel:
; The rounding opcodes leave their result in floating-point form, so it goes
; back to the destination SGPR as an f32 bit pattern.
; CHECK: [[CEIL_IN:%.+]] = bitcast i32 {{.+}} to float
; CHECK: [[CEIL:%.+]] = call float @llvm.ceil.f32(float [[CEIL_IN]])
; CHECK: [[CEIL_BITS:%.+]] = bitcast float [[CEIL]] to i32
	s_ceil_f32 s1, s0
; CHECK: [[FLOOR_IN:%.+]] = bitcast i32 [[CEIL_BITS]] to float
; CHECK: [[FLOOR:%.+]] = call float @llvm.floor.f32(float [[FLOOR_IN]])
; CHECK: [[FLOOR_BITS:%.+]] = bitcast float [[FLOOR]] to i32
	s_floor_f32 s2, s1
; CHECK: [[TRUNC_IN:%.+]] = bitcast i32 [[FLOOR_BITS]] to float
; CHECK: [[TRUNC:%.+]] = call float @llvm.trunc.f32(float [[TRUNC_IN]])
; CHECK: [[TRUNC_BITS:%.+]] = bitcast float [[TRUNC]] to i32
	s_trunc_f32 s3, s2
; CHECK: [[RNDNE_IN:%.+]] = bitcast i32 [[TRUNC_BITS]] to float
; CHECK: [[RNDNE:%.+]] = call float @llvm.roundeven.f32(float [[RNDNE_IN]])
; CHECK: [[RNDNE_BITS:%.+]] = bitcast float [[RNDNE]] to i32
	s_rndne_f32 s4, s3
; CHECK: [[SITOFP:%.+]] = sitofp i32 [[RNDNE_BITS]] to float
; CHECK: [[SITOFP_BITS:%.+]] = bitcast float [[SITOFP]] to i32
	s_cvt_f32_i32 s5, s4
; CHECK: [[UITOFP:%.+]] = uitofp i32 [[SITOFP_BITS]] to float
; CHECK: [[UITOFP_BITS:%.+]] = bitcast float [[UITOFP]] to i32
	s_cvt_f32_u32 s6, s5
; An out-of-range input saturates and a NaN converts to zero, which is what the
; saturating intrinsics give and what plain fptosi / fptoui make poison.
; CHECK: [[TOSI_IN:%.+]] = bitcast i32 [[UITOFP_BITS]] to float
; CHECK: [[TOSI:%.+]] = call i32 @llvm.fptosi.sat.i32.f32(float [[TOSI_IN]])
	s_cvt_i32_f32 s7, s6
; CHECK: [[TOUI_IN:%.+]] = bitcast i32 [[TOSI]] to float
; CHECK: [[TOUI:%.+]] = call i32 @llvm.fptoui.sat.i32.f32(float [[TOUI_IN]])
	s_cvt_u32_f32 s8, s7
; A half result lands in the low half of the destination and zeroes its high
; half.
; CHECK: [[CVTF16_IN:%.+]] = bitcast i32 [[TOUI]] to float
; CHECK: [[CVTF16:%.+]] = fptrunc float [[CVTF16_IN]] to half
; CHECK: [[CVTF16_BITS:%.+]] = bitcast half [[CVTF16]] to i16
; CHECK: [[CVTF16_DST:%.+]] = zext i16 [[CVTF16_BITS]] to i32
	s_cvt_f16_f32 s9, s8
; s_cvt_f32_f16 reads the low half of its source, so the truncation takes the
; source itself rather than a shifted copy of it.
; CHECK: [[LO:%.+]] = trunc i32 [[CVTF16_DST]] to i16
; CHECK: [[LO_HALF:%.+]] = bitcast i16 [[LO]] to half
; CHECK: [[LO_F32:%.+]] = fpext half [[LO_HALF]] to float
; CHECK: [[LO_BITS:%.+]] = bitcast float [[LO_F32]] to i32
	s_cvt_f32_f16 s10, s9
; s_cvt_hi_f32_f16 reads the high half instead.
; CHECK: [[SHIFTED:%.+]] = lshr i32 [[LO_BITS]], 16
; CHECK: [[HI:%.+]] = trunc i32 [[SHIFTED]] to i16
; CHECK: [[HI_HALF:%.+]] = bitcast i16 [[HI]] to half
; CHECK: [[HI_F32:%.+]] = fpext half [[HI_HALF]] to float
; CHECK: [[HI_BITS:%.+]] = bitcast float [[HI_F32]] to i32
	s_cvt_hi_f32_f16 s11, s10
; The f16 roundings read the low half of the source and zero the high half of
; the destination on both ends.
; CHECK: [[CEILH_LO:%.+]] = trunc i32 [[HI_BITS]] to i16
; CHECK: [[CEILH_IN:%.+]] = bitcast i16 [[CEILH_LO]] to half
; CHECK: [[CEILH:%.+]] = call half @llvm.ceil.f16(half [[CEILH_IN]])
; CHECK: [[CEILH_BITS:%.+]] = bitcast half [[CEILH]] to i16
; CHECK: [[CEILH_DST:%.+]] = zext i16 [[CEILH_BITS]] to i32
	s_ceil_f16 s12, s11
; CHECK: [[FLOORH_LO:%.+]] = trunc i32 [[CEILH_DST]] to i16
; CHECK: [[FLOORH_IN:%.+]] = bitcast i16 [[FLOORH_LO]] to half
; CHECK: [[FLOORH:%.+]] = call half @llvm.floor.f16(half [[FLOORH_IN]])
; CHECK: [[FLOORH_BITS:%.+]] = bitcast half [[FLOORH]] to i16
; CHECK: [[FLOORH_DST:%.+]] = zext i16 [[FLOORH_BITS]] to i32
	s_floor_f16 s13, s12
; CHECK: [[TRUNCH_LO:%.+]] = trunc i32 [[FLOORH_DST]] to i16
; CHECK: [[TRUNCH_IN:%.+]] = bitcast i16 [[TRUNCH_LO]] to half
; CHECK: [[TRUNCH:%.+]] = call half @llvm.trunc.f16(half [[TRUNCH_IN]])
; CHECK: [[TRUNCH_BITS:%.+]] = bitcast half [[TRUNCH]] to i16
; CHECK: [[TRUNCH_DST:%.+]] = zext i16 [[TRUNCH_BITS]] to i32
	s_trunc_f16 s14, s13
; CHECK: [[RNDNEH_LO:%.+]] = trunc i32 [[TRUNCH_DST]] to i16
; CHECK: [[RNDNEH_IN:%.+]] = bitcast i16 [[RNDNEH_LO]] to half
; CHECK: [[RNDNEH:%.+]] = call half @llvm.roundeven.f16(half [[RNDNEH_IN]])
; CHECK: [[RNDNEH_BITS:%.+]] = bitcast half [[RNDNEH]] to i16
; CHECK: zext i16 [[RNDNEH_BITS]] to i32
	s_rndne_f16 s15, s14
	s_endpgm

	.globl	scc_kernel
	.p2align	8
	.type	scc_kernel,@function
; s_not_b32 writes SCC and s_cmov_b32 reads it. Every scalar float opcode runs
; in between, so the select still taking the bit s_not_b32 produced is what
; says none of them touched SCC along the way.
; CHECK-LABEL: define amdgpu_kernel void @scc_kernel(
scc_kernel:
; The value s_cmov_b32 preserves when SCC is clear.
	s_mov_b32 s2, 7
; CHECK: [[NOT:%.+]] = xor i32 {{.+}}, -1
; CHECK: [[SCC:%.+]] = icmp ne i32 [[NOT]], 0
	s_not_b32 s0, s1
	s_ceil_f32 s3, s0
	s_floor_f32 s3, s3
	s_trunc_f32 s3, s3
	s_rndne_f32 s3, s3
	s_cvt_f32_i32 s3, s3
	s_cvt_f32_u32 s3, s3
	s_cvt_i32_f32 s3, s3
	s_cvt_u32_f32 s3, s3
	s_cvt_f16_f32 s3, s3
	s_cvt_f32_f16 s3, s3
	s_cvt_hi_f32_f16 s3, s3
	s_ceil_f16 s3, s3
	s_floor_f16 s3, s3
	s_trunc_f16 s3, s3
	s_rndne_f16 s3, s3
; CHECK: select i1 [[SCC]], i32 {{.+}}, i32 7
	s_cmov_b32 s2, s3
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
	s_rfe_i64 s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scalar_float_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.amdhsa_kernel scc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
    .name:           scalar_float_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         scalar_float_kernel.kd
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
    .name:           rfe_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         rfe_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
