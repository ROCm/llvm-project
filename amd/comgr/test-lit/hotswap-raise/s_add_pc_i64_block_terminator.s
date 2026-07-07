; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=sapc_block_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; s_add_pc_i64 as a block terminator resolved to a direct branch.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	sapc_block_kernel
	.p2align	8
	.type	sapc_block_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @sapc_block_kernel(
sapc_block_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_cbranch_scc1 2
	s_add_pc_i64 lit64(8)
; CHECK: br label %bb_0x20
; CHECK-NOT: indirectbr
	s_nop 0
	s_nop 0
; CHECK: bb_0x20:
	s_wait_kmcnt 0x0
	v_mov_b32 v1, 0xCAFE0002
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sapc_block_kernel
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
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           sapc_block_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         sapc_block_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
