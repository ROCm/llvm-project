; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=global_load_tr6_b96_kernel | %FileCheck %s

; global_load_tr6_b96 transpose lift: ds.bpermute lane-shuffle emulation on gfx942.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_tr6_b96_kernel
	.p2align	8
	.type	global_load_tr6_b96_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @global_load_tr6_b96_kernel(
global_load_tr6_b96_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 4, v0
	s_wait_kmcnt 0x0
; CHECK: [[LANELO:%[a-zA-Z_0-9]+]] = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK: call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 [[LANELO]])
; CHECK: = load <3 x i32>, ptr addrspace(1)
; CHECK-COUNT-48: call i32 @llvm.amdgcn.ds.bpermute(
; CHECK-NOT: call i32 @llvm.amdgcn.ds.bpermute(
; CHECK: zext i32 %{{[^ ]+}} to i64
; CHECK: shl i64 %{{[^,]+}}, 32
; CHECK: %tr6_win{{[0-9]*}} = or i64
; CHECK: %tr6_elem{{[0-9]*}} = trunc i64
; CHECK-NOT: call {{.*}}@llvm.amdgcn.global.load.tr6.b96
; CHECK-NOT: call {{.*}}@llvm.amdgcn.global.load.tr.b96
	global_load_tr6_b96 v[2:4], v1, s[0:1]
	s_wait_loadcnt 0x0
	global_store_b96 v0, v[2:4], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_tr6_b96_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 2
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
    .name:           global_load_tr6_b96_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_tr6_b96_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
