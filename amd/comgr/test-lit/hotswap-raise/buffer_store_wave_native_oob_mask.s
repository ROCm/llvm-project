; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=buffer_store_wave_native_oob_mask_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=WN
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:   --emit-ir=buffer_store_wave_native_oob_mask_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=MR

; Buffer store OOB-lane masking under wave-native vs modulo-replication.
; WN-LABEL: define amdgpu_kernel void @buffer_store_wave_native_oob_mask_kernel(
; WN: call i1 @llvm.amdgcn.init.whole.wave()
; WN: call void @llvm.amdgcn.raw.buffer.store.i32(
; MR-LABEL: define amdgpu_kernel void @buffer_store_wave_native_oob_mask_kernel(
; MR: br i1 %{{[^,]+}}, label %spe_do{{[0-9]*}}, label %spe_skip{{[0-9]*}}
; MR: spe_do{{[0-9]*}}:
; MR: call void @llvm.amdgcn.raw.buffer.store.i32(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_store_wave_native_oob_mask_kernel
	.p2align	8
	.type	buffer_store_wave_native_oob_mask_kernel,@function
buffer_store_wave_native_oob_mask_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v2, 64
	v_mov_b32_e32 v3, 0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_cndmask_b32_e32 v1, v2, v3, vcc_lo
	s_mov_b32 s2, 4
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	buffer_store_dword v0, v1, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_wave_native_oob_mask_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
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
    .max_flat_workgroup_size: 64
    .name:           buffer_store_wave_native_oob_mask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_wave_native_oob_mask_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
