; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: env HSA_HOTSWAP_STRICT=1 raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --assume-hip-global-offset-zero \
; RUN:   --emit-ir=kernarg_hiddenarg_full_pair_same_block,kernarg_hiddenarg_full_pair_successor \
; RUN:   | %FileCheck %s

; NonEntry kernarg provenance: s_load rebases the whole pair, lifting hidden-arg loads to ordinary global memory (same-block + successor).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kernarg_hiddenarg_full_pair_same_block
	.p2align	8
	.type	kernarg_hiddenarg_full_pair_same_block,@function
; CHECK-LABEL: define amdgpu_kernel void @kernarg_hiddenarg_full_pair_same_block(
kernarg_hiddenarg_full_pair_same_block:
	s_load_b64 s[0:1], s[0:1], 0x8
	s_wait_kmcnt 0x0
; CHECK-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; CHECK: inttoptr i64
; CHECK: load i32, ptr addrspace(1)
	s_load_b32 s2, s[0:1], 0x8
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s2
	s_endpgm

	.globl	kernarg_hiddenarg_full_pair_successor
	.p2align	8
	.type	kernarg_hiddenarg_full_pair_successor,@function
; CHECK-LABEL: define amdgpu_kernel void @kernarg_hiddenarg_full_pair_successor(
kernarg_hiddenarg_full_pair_successor:
	s_load_b64 s[0:1], s[0:1], 0x8
	s_wait_kmcnt 0x0
	s_branch .Lload_successor
	s_nop 0
.Lload_successor:
; CHECK-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; CHECK: inttoptr i64
; CHECK: load i32, ptr addrspace(1)
	s_load_b32 s2, s[0:1], 0x8
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s2
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_hiddenarg_full_pair_same_block
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_hiddenarg_full_pair_successor
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     hidden_global_offset_x
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 32
    .name:           kernarg_hiddenarg_full_pair_same_block
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         kernarg_hiddenarg_full_pair_same_block.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     hidden_global_offset_x
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 32
    .name:           kernarg_hiddenarg_full_pair_successor
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         kernarg_hiddenarg_full_pair_successor.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
