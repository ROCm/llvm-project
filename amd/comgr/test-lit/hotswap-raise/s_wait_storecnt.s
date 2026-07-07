; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_wait_storecnt_kernel 2>/dev/null | %FileCheck %s

; s_wait_storecnt store-count wait lowered to s.waitcnt.
; CHECK-LABEL: define amdgpu_kernel void @s_wait_storecnt_kernel(
; CHECK: store i32 {{.*}}, ptr addrspace(1) %{{[^,]+}}
; CHECK: call void @llvm.amdgcn.s.waitcnt(i32 0)
; CHECK: store i32 {{.*}}, ptr addrspace(1) %{{[^,]+}}
; CHECK: call void @llvm.amdgcn.s.waitcnt(i32 0)
; CHECK: store i32 {{.*}}, ptr addrspace(1) %{{[^,]+}}
; CHECK: call void @llvm.amdgcn.s.waitcnt(i32 0)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_wait_storecnt_kernel
	.p2align	8
	.type	s_wait_storecnt_kernel,@function
s_wait_storecnt_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, 0x11111111
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_wait_storecnt 0
	v_mov_b32_e32 v1, 0x22222222
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_wait_storecnt 1
	v_mov_b32_e32 v1, 0x33333333
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_wait_storecnt 3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_wait_storecnt_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
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
    .name:           s_wait_storecnt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         s_wait_storecnt_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
