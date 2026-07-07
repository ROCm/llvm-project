; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>/dev/null \
; RUN:   | %FileCheck --check-prefix=CROSS %s
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel_sel2 2>&1 \
; RUN:   | %FileCheck --check-prefix=REFUSE-SAME %s
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel_sel2 2>&1 \
; RUN:   | %FileCheck --check-prefix=REFUSE-CROSS %s

; v_cvt_scale_pk8_bf16_fp4 native passthrough vs cross-target mxfp4 dequant + scale_sel!=0 refusal.
; CHECK-LABEL: define amdgpu_kernel void @v_cvt_scale_pk8_bf16_fp4_kernel(
; CHECK: %cvt_scale_pk8_bf16_fp4 = call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %{{.*}}, i32 %{{.*}}, i32 0)
; CHECK: bitcast <8 x bfloat> %cvt_scale_pk8_bf16_fp4 to i128
; CHECK-DAG: declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32, i32, i32 immarg range(i32 0, 16))
; CHECK-NOT: scale_sel != 0
; CHECK-NOT: amdgcn.cvt.pk.f32
; CHECK-NOT: amdgcn.cvt.f32.fp8
; CHECK-NOT: mxfp4_nibble
; CROSS-LABEL: define amdgpu_kernel void @v_cvt_scale_pk8_bf16_fp4_kernel(
; CROSS-NOT: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4
; CROSS: %mxfp4_scale_byte = and i32 %{{.*}}, 255
; CROSS: %mxfp4_is_scale_nan = icmp eq i32 %mxfp4_scale_byte, 255
; CROSS-DAG: %mxfp4_nibble = and i32 %{{.*}}, 15
; CROSS-DAG: %mxfp4_sign = and i32 %{{.*}}, 1
; CROSS-DAG: %mxfp4_exp_fp4 = and i32 %{{.*}}, 3
; CROSS-DAG: %mxfp4_mant_fp4 = and i32 %mxfp4_nibble, 1
; CROSS-DAG: %mxfp4_bf16_exp_norm = add i32 %mxfp4_exp_fp4, 126
; CROSS-DAG: %mxfp4_is_fp4_sub = icmp eq i32 %mxfp4_exp_fp4, 0
; CROSS-DAG: %mxfp4_exp_plus_scale = add i32 %mxfp4_bf16_exp, %mxfp4_scale_byte
; CROSS-DAG: %mxfp4_new_exp = sub i32 %mxfp4_exp_plus_scale, 127
; CROSS-DAG: %mxfp4_is_overflow = icmp sge i32 %mxfp4_new_exp, 255
; CROSS-DAG: %mxfp4_inf_bits = or i32 %mxfp4_sign_field, 32640
; CROSS-DAG: %mxfp4_implicit_1_mant = or i32 128, %mxfp4_bf16_mant
; CROSS-DAG: %mxfp4_shift_amt = sub i32 1, %mxfp4_new_exp
; CROSS-DAG: %mxfp4_shift_too_big = icmp sge i32 %mxfp4_shift_amt, 8
; CROSS-DAG: %mxfp4_lane_i16 = trunc i32 %mxfp4_lane_i32 to i16
; CROSS-DAG: %mxfp4_lane_bf16 = bitcast i16 %mxfp4_lane_i16 to bfloat
; CROSS: insertelement <8 x bfloat>
; CROSS: bitcast <8 x bfloat> %{{.*}} to i128
; CROSS-NOT: private constant [16 x i16]
; CROSS-NOT: v_cvt_scale_pk8_bf16_fp4 is a gfx1250-only VOP3
; CROSS-NOT: no corpus kernel exercises today
; REFUSE-SAME: scale_sel != 0 is outside the declared support set
; REFUSE-SAME-SAME: matrix-translation.md §7.4
; REFUSE-SAME-NOT: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4
; REFUSE-CROSS: scale_sel != 0 is outside the declared support set
; REFUSE-CROSS-SAME: matrix-translation.md §7.4
; REFUSE-CROSS-NOT: mxfp4_nibble
; REFUSE-CROSS-NOT: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_scale_pk8_bf16_fp4_kernel
	.p2align	8
	.type	v_cvt_scale_pk8_bf16_fp4_kernel,@function
v_cvt_scale_pk8_bf16_fp4_kernel:        ; @v_cvt_scale_pk8_bf16_fp4_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v0, s0
	s_add_co_i32 s2, s0, 4
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cvt_scale_pk8_bf16_fp4 v[0:3], v0, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0xffff, v0
	v_lshl_or_b32 v0, v0, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_mov_b32_e32 v3, v0
	global_store_b128 v4, v[0:3], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_scale_pk8_bf16_fp4_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 3
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.globl	v_cvt_scale_pk8_bf16_fp4_kernel_sel2
	.p2align	8
	.type	v_cvt_scale_pk8_bf16_fp4_kernel_sel2,@function
v_cvt_scale_pk8_bf16_fp4_kernel_sel2:   ; @v_cvt_scale_pk8_bf16_fp4_kernel_sel2
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v0, s0
	s_add_co_i32 s2, s0, 4
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cvt_scale_pk8_bf16_fp4 v[0:3], v0, s2 scale_sel:2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0xffff, v0
	v_lshl_or_b32 v0, v0, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_mov_b32_e32 v3, v0
	global_store_b128 v4, v[0:3], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_scale_pk8_bf16_fp4_kernel_sel2
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 3
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
    .name:           v_cvt_scale_pk8_bf16_fp4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         v_cvt_scale_pk8_bf16_fp4_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_scale_pk8_bf16_fp4_kernel_sel2
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         v_cvt_scale_pk8_bf16_fp4_kernel_sel2.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
