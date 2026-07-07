; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=smem_sgpr_imm_scale_offset_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; SMEM SGPR+imm offset with scale_offset lowered to a scaled GEP.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	smem_sgpr_imm_scale_offset_kernel
	.p2align	8
	.type	smem_sgpr_imm_scale_offset_kernel,@function
smem_sgpr_imm_scale_offset_kernel:
; CHECK-LABEL: define amdgpu_kernel void @smem_sgpr_imm_scale_offset_kernel(
	s_load_b64 s[4:5], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x8
	s_wait_loadcnt 0x0
; CHECK: %smem_roff{{[0-9]*}} = zext i32
; CHECK: %smem_roff_scaled{{[0-9]*}} = mul i64 %smem_roff{{[0-9]*}}, 4
; CHECK: %smem_roff_plus_imm{{[0-9]*}} = add i64 %smem_roff_scaled{{[0-9]*}}, 2056
; CHECK: getelementptr inbounds i8, ptr addrspace(1) %{{[^,]+}}, i64 %smem_roff_plus_imm
	s_load_b32 s3, s[4:5], s2 offset:0x808 scale_offset
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v1, s3
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_sgpr_imm_scale_offset_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .offset: 8, .size: 4, .value_kind: by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .max_flat_workgroup_size: 1024
    .name: smem_sgpr_imm_scale_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 6
    .symbol: smem_sgpr_imm_scale_offset_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
