; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native --emit-ir=wmma_f32_16x16x64_fp8_fp8_kernel \
; RUN:   | %FileCheck %s --implicit-check-not=e5m2_fnuz

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_f32_16x16x64_fp8_fp8_kernel
	.p2align	8
	.type	wmma_f32_16x16x64_fp8_fp8_kernel,@function
wmma_f32_16x16x64_fp8_fp8_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[24:27], s[0:1], 0x0
	s_load_b64 s[28:29], s[0:1], 0x10
	v_mov_b32_e32 v24, 0
	s_wait_kmcnt 0x0
	s_load_b256 s[0:7], s[24:25], 0x0
	s_load_b256 s[8:15], s[26:27], 0x0
	s_load_b256 s[16:23], s[28:29], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	s_delay_alu instid0(VALU_DEP_1)
; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x64_fp8_fp8_kernel(
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()
; Both operands are fp8 (E4M3), one conversion per A and B dword.
; CHECK-COUNT-16: trunc <4 x i32> %e4m3_fnuz{{[0-9]*}} to <4 x i8>
; Every A/B dword the redistribution reads must be a CONVERTED one. The raiser
; runs no DCE, so a conversion CHECK not tied to a consumer still matches when
; the def-use edge is dropped and OCP bytes reach the FNUZ MFMA.
; CHECK-COUNT-32: call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_{{lo|hi}}, i32 %fp8_conv_dw{{[0-9]*}})
; The remaining redistribution in this pass is the accumulator, which is f32.
; CHECK-NOT: ds.bpermute(i32 %addr_{{lo|hi}}, i32 %fp8_conv_dw
; CHECK: %{{.*}} = bitcast <2 x i32> %{{.*}} to i64
; CHECK: %mfma1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %mfma1, i32 0, i32 0, i32 0)
; CHECK: %mfma1{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %mfma1{{[0-9]+}}, i32 0, i32 0, i32 0)
; CHECK-NOT: call i32 @llvm.amdgcn.strict.wwm.i32(
; CHECK-NOT: call {{.*}} @llvm.amdgcn.strict.wwm
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8
	v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23]
	s_clause 0x1
	global_store_b128 v24, v[20:23], s[28:29] offset:16
	global_store_b128 v24, v[16:19], s[28:29]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_f32_16x16x64_fp8_fp8_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 25
		.amdhsa_next_free_sgpr 30
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           wmma_f32_16x16x64_fp8_fp8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         wmma_f32_16x16x64_fp8_fp8_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
