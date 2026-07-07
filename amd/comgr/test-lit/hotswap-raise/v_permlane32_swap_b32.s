; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_permlane32_swap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_permlane32_swap_b32 cross-lane swap lift.
; CHECK-LABEL: define amdgpu_kernel void @v_permlane32_swap_b32_kernel(
; CHECK: %pls32_partner{{[0-9]*}} = xor i32 %{{[^,]+}}, 32
; CHECK: %pls32_addr{{[0-9]*}} = shl i32 %pls32_partner{{[0-9]*}}, 2
; CHECK: %pls32_new_vdst{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls32_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: %pls32_new_src0_out{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls32_addr{{[0-9]*}}, i32 %{{[^)]+}})

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	v_permlane32_swap_b32_kernel
	.p2align	8
	.type	v_permlane32_swap_b32_kernel,@function
v_permlane32_swap_b32_kernel:           ; @v_permlane32_swap_b32_kernel
; %bb.0:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_add_u32_e32 v1, 0x3e8, v0
	v_mov_b32_e32 v2, v0
	v_permlane32_swap_b32 v2, v1
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v2, s[0:1]
	global_store_dword v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_permlane32_swap_b32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_tg_split 0
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_permlane32_swap_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_permlane32_swap_b32_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
