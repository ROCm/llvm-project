; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_load_b32_scale_offset_kernel 2>/dev/null | %FileCheck %s

; s_load_b32 scale_offset scaled-offset GEP lowering.
; CHECK-LABEL: define amdgpu_kernel void @s_load_b32_scale_offset_kernel(
; CHECK: call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
; CHECK: %smem_roff = zext i32 %{{[^ ,]+}} to i64
; CHECK: %smem_roff_scaled = mul i64 %smem_roff, 4
; CHECK: getelementptr inbounds i8, ptr addrspace(1) %{{[^ ,]+}}, i64 %smem_roff_scaled
; CHECK: %smem_load{{[0-9]*}} = load i32, ptr addrspace(1) %{{[^ ,]+}}, align 4
; CHECK-NOT: getelementptr inbounds i8, ptr addrspace(1) %{{[^ ,]+}}, i64 %smem_roff{{$}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_load_b32_scale_offset_kernel
	.p2align	8
	.type	s_load_b32_scale_offset_kernel,@function
s_load_b32_scale_offset_kernel:         ; @s_load_b32_scale_offset_kernel
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_load_b32 s0, s[6:7], s2 offset:0x0 scale_offset
	s_wait_kmcnt 0
	
	v_mov_b32_e32 v1, s0
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_load_b32_scale_offset_kernel
		.amdhsa_kernarg_size 20
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .actual_access:  write_only, .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
      - { .offset:         16, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .max_flat_workgroup_size: 1024
    .name:           s_load_b32_scale_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_load_b32_scale_offset_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
