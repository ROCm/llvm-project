; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && env HSA_HOTSWAP_STRICT=1 raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=source_hidden_sub_const_same_bb 2>/dev/null \
; RUN:   | %FileCheck %s

; Constant subtraction from the entry kernarg pointer stays an Entry+Const fact, remapped to target implicitarg.ptr offset 80.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	source_hidden_sub_const_same_bb
	.p2align	8
	.type	source_hidden_sub_const_same_bb,@function
; CHECK-LABEL: define amdgpu_kernel void @source_hidden_sub_const_same_bb(
source_hidden_sub_const_same_bb:
	s_add_nc_u64 s[0:1], s[0:1], 0x20
	s_sub_nc_u64 s[0:1], s[0:1], 0x10
; CHECK: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; CHECK: getelementptr inbounds i8, ptr addrspace(4) %{{[^,]+}}, i32 80
; CHECK-NOT: getelementptr inbounds i8, ptr addrspace(4) %{{[^,]+}}, i32 56
; CHECK: load i64, ptr addrspace(4) %{{[^,]+}}, align 8
	s_load_b64 s[2:3], s[0:1], 0x28
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel source_hidden_sub_const_same_bb
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           8
        .value_kind:     hidden_hostcall_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .max_flat_workgroup_size: 1024
    .name:           source_hidden_sub_const_same_bb
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .sgpr_spill_count: 0
    .symbol:         source_hidden_sub_const_same_bb.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
