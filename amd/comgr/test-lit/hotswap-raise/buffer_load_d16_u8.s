; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_load_d16_u8_kernel 2>/dev/null | %FileCheck %s

; Buffer d16 lo/hi u8 load lift with 16-bit lane packing.
; CHECK-LABEL: define amdgpu_kernel void @buffer_load_d16_u8_kernel(
; CHECK-DAG: call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8
; CHECK-DAG: call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8
; CHECK-DAG: and i32 {{.*}}, -65536
; CHECK-DAG: or {{(disjoint )?}}i32 {{.*}}, %{{.*}}
; CHECK-DAG: and i32 {{.*}}, 65535
; CHECK-DAG: shl i32 {{.*}}, 16
; CHECK-DAG: or {{(disjoint )?}}i32 {{.*}}, %{{.*}}
; CHECK-NOT: call i16 @llvm.amdgcn.raw.ptr.buffer.load
; CHECK-NOT: call i32 @llvm.amdgcn.raw.ptr.buffer.load

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_load_d16_u8_kernel
	.p2align	8
	.type	buffer_load_d16_u8_kernel,@function
buffer_load_d16_u8_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	v_dual_mov_b32 v2, 0xcafefeed :: v_dual_lshlrev_b32 v1, 2, v0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	v_mov_b32_e32 v3, 0xcafefeed
	s_wait_kmcnt 0x0
	s_mov_b32 s4, s2
	s_mov_b32 s5, s3
	buffer_load_d16_u8 v2, v1, s[4:7], null offen scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	buffer_load_d16_hi_u8 v3, v1, s[4:7], null offen scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	global_store_b64 v0, v[2:3], s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_load_d16_u8_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           buffer_load_d16_u8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         buffer_load_d16_u8_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
