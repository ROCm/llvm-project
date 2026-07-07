; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=setpc_set_dispatch_set_kernel 2>/dev/null | %FileCheck %s

; s_set_pc_i64 enumerated dispatch-set lowered to icmp/branch chain.
; CHECK-LABEL: define amdgpu_kernel void @setpc_set_dispatch_set_kernel(
; CHECK-DAG: 60, %bb_0x18
; CHECK-DAG: 68, %bb_0x28
; CHECK: %ret_pc_marker = or i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK-NEXT: %dispatch_0x38_cmp_0 = icmp eq i64 %ret_pc_marker, 60
; CHECK-NEXT: br i1 %dispatch_0x38_cmp_0, label %bb_0x3C, label %dispatch_0x38_1
; CHECK: dispatch_0x38_unreachable:
; CHECK-NEXT: unreachable
; CHECK: dispatch_0x38_1:
; CHECK-NEXT: %dispatch_0x38_cmp_1 = icmp eq i64 %ret_pc_marker, 68
; CHECK-NEXT: br i1 %dispatch_0x38_cmp_1, label %bb_0x44, label %dispatch_0x38_unreachable
; CHECK-NOT: indirectbr
; CHECK-NOT: blockaddress(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setpc_set_dispatch_set_kernel
	.p2align	8
	.type	setpc_set_dispatch_set_kernel,@function
setpc_set_dispatch_set_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	s_cmp_eq_u32 s2, s3
	s_cbranch_scc1 4
	s_get_pc_i64 s[10:11]
	s_add_co_u32 s10, s10, 32
	s_add_co_ci_u32 s11, s11, 0
	s_branch 4
	s_get_pc_i64 s[10:11]
	s_add_co_u32 s10, s10, 24
	s_add_co_ci_u32 s11, s11, 0
	s_branch 0
	s_set_pc_i64 s[10:11]
	v_mov_b32 v1, 0xCAFE0001
	v_mov_b32 v1, 0xDEAD0001
	s_endpgm
	
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setpc_set_dispatch_set_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
    .max_flat_workgroup_size: 1024
    .name:           setpc_set_dispatch_set_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         setpc_set_dispatch_set_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
