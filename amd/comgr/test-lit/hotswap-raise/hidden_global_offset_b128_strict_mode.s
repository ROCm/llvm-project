; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && env HSA_HOTSWAP_STRICT=1 raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --assume-hip-global-offset-zero \
; RUN:     --emit-ir=hidden_global_offset_b128_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Wide SMEM hidden-arg synthesis: AISE kernels may read two adjacent 64-bit
; hidden_global_offset fields with one s_load_b128. The source ABI value is
; HIP's always-zero global work offset, so each dword is synthesized and strict
; mode must not fall back to the target hidden-arg block.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.protected	hidden_global_offset_b128_kernel
	.globl	hidden_global_offset_b128_kernel
	.p2align	8
	.type	hidden_global_offset_b128_kernel,@function
hidden_global_offset_b128_kernel:
; CHECK-LABEL: define amdgpu_kernel void @hidden_global_offset_b128_kernel(
	s_clause 0x1
	s_load_b64 s[8:9], s[0:1], 0x0
	s_load_b128 s[4:7], s[0:1], 0x8
; CHECK-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; CHECK: phi i32 [ 0, %{{[a-zA-Z_0-9]+}} ], [ %tid, %{{[a-zA-Z_0-9]+}} ]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v0, s4 :: v_dual_mov_b32 v1, 0
	global_store_b32 v1, v0, s[8:9]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel hidden_global_offset_b128_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 10
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
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
      - .offset:         16
        .size:           8
        .value_kind:     hidden_global_offset_y
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           hidden_global_offset_b128_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         hidden_global_offset_b128_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
