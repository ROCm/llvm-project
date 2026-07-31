; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && env HSA_HOTSWAP_STRICT=1 %raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:     --emit-ir=hidden_arg_rebased_pointer \
; RUN:   | %FileCheck %s

; Hidden read through an in-place constant rebase of the kernarg pair.
; CHECK-LABEL: define amdgpu_kernel void @hidden_arg_rebased_pointer(
; CHECK-NOT: call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	hidden_arg_rebased_pointer
	.p2align	8
	.type	hidden_arg_rebased_pointer,@function
hidden_arg_rebased_pointer:
	s_add_nc_u64 s[0:1], s[0:1], 0x50
; CHECK: load i32, ptr addrspace(1)
	s_load_b64 s[4:5], s[0:1], 0x8
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s4
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel hidden_arg_rebased_pointer
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .offset: 88, .size: 8, .value_kind: hidden_hostcall_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           hidden_arg_rebased_pointer
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         hidden_arg_rebased_pointer.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
