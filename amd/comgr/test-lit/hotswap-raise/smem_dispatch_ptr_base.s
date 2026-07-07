; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=smem_dispatch_ptr_base_kernel \
; RUN:   | %FileCheck %s

; Dispatch-ptr SGPR seeding for scalar loads through s[0:1].
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	smem_dispatch_ptr_base_kernel
	.p2align	8
	.type	smem_dispatch_ptr_base_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @smem_dispatch_ptr_base_kernel(
smem_dispatch_ptr_base_kernel:
; CHECK: [[DISPATCH:%[^ ]+]] = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK: ptrtoint ptr addrspace(4) [[DISPATCH]] to i64
; CHECK-NOT: zext i32 undef to i64
; CHECK: load i32, ptr addrspace(1) %{{[^,]+}}, align 4
	s_load_b64 s[2:3], s[0:1], 0x4
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s2, s3
	v_mov_b32_e32 v0, s4
; CHECK: ret void
; CHECK-NOT: zext i32 undef to i64
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_dispatch_ptr_base_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
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
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 256
    .name:           smem_dispatch_ptr_base_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         smem_dispatch_ptr_base_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
