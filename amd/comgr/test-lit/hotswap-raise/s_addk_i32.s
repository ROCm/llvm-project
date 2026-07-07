; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_addk_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_addk_co_i32 scalar add-with-signed-immediate (with overflow) lift.
; CHECK-LABEL: define amdgpu_kernel void @s_addk_i32_kernel(
; CHECK: %addk = add i32 %{{[^,]+}}, 1024
; CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{[^,]+}}, i32 1024)
; CHECK: %addk{{[0-9]*}} = add i32 %{{[^,]+}}, -12288
; CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{[^,]+}}, i32 -12288)
; CHECK-NOT: %addk = add i32 %{{[^,]+}}, %{{[^,]+}}
; CHECK-NOT: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{[^,]+}}, i32 %{{[^,]+}})

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_addk_i32_kernel
	.p2align	8
	.type	s_addk_i32_kernel,@function
s_addk_i32_kernel:                      ; @s_addk_i32_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s2, ttmp9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s3, s3, s2
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s2, ttmp9, s3
	s_addk_co_i32 s2, 0x400
	s_addk_co_i32 s2, 0xd000
	
	v_add_nc_u32_e32 v1, s2, v0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_addk_i32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 5
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
    .name:           s_addk_i32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         s_addk_i32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
