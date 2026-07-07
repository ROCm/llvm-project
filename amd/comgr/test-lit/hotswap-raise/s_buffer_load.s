; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=s_buffer_load_success_kernel \
; RUN:   | %FileCheck %s --check-prefix=SUCCESS
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --write-hsaco=%t.gfx942.hsaco --kernel=s_buffer_load_success_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=PIPE
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=s_buffer_load_scope_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SCOPE
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=s_buffer_load_unrepresentable_base_kernel \
; RUN:   | %FileCheck %s --check-prefix=TRAP
;
; gfx12 S_BUFFER_LOAD consumes a four-SGPR buffer resource descriptor. The
; raiser decodes the source descriptor fields used by scalar buffer loads,
; rebuilds a target raw-buffer resource with the source byte extent, and lets
; target buffer hardware return zero for out-of-bounds load elements.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

; SUCCESS-LABEL: define amdgpu_kernel void @s_buffer_load_success_kernel(
; PIPE: raise_cli: wrote {{[0-9]+}} byte HSACO for kernel 's_buffer_load_success_kernel'
	.globl	s_buffer_load_success_kernel
	.p2align	8
	.type	s_buffer_load_success_kernel,@function
s_buffer_load_success_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
; SUCCESS: [[EXTENT:%[^ ]+]] = mul i64
; SUCCESS: call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %{{[^,]+}}, i16 0, i64 [[EXTENT]], i32 0)
; SUCCESS: call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(
	s_buffer_load_b64 s[8:9], s[4:7], 0x0
; SUCCESS: call <3 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v3i32(
	s_buffer_load_b96 s[12:14], s[4:7], 0x8
; SUCCESS: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
	s_buffer_load_b128 s[16:19], s[4:7], 0x10
	s_mov_b32 s20, 0x20
; SUCCESS: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
; SUCCESS: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
	s_buffer_load_b256 s[24:31], s[4:7], s20 offset:0x20
; SUCCESS: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
; SUCCESS: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
; SUCCESS: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
; SUCCESS: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
	s_buffer_load_b512 s[32:47], s[4:7], 0x40
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s8
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_buffer_load_success_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 48
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

; SCOPE: S_BUFFER_LOAD cache-policy/scope bits are not modelled
	.globl	s_buffer_load_scope_refuse_kernel
	.p2align	8
	.type	s_buffer_load_scope_refuse_kernel,@function
s_buffer_load_scope_refuse_kernel:
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0
	s_mov_b32 s6, 2
	s_mov_b32 s7, 0x1000
	s_buffer_load_b64 s[8:9], s[4:7], 0x0 scope:SCOPE_DEV
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s8
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_buffer_load_scope_refuse_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 10
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480

; TRAP-LABEL: define amdgpu_kernel void @s_buffer_load_unrepresentable_base_kernel(
; TRAP: trunc i64 {{%[^ ]+}} to i48
; TRAP: sext i48 {{%[^ ]+}} to i64
; TRAP: icmp eq i64
; TRAP: call void @llvm.trap()
	.globl	s_buffer_load_unrepresentable_base_kernel
	.p2align	8
	.type	s_buffer_load_unrepresentable_base_kernel,@function
s_buffer_load_unrepresentable_base_kernel:
	s_mov_b32 s4, 0
	; base bit 48 is outside gfx942's signed 48-bit V# base range, while
	; bit 57 also provides a finite one-byte source extent for the test.
	s_mov_b32 s5, 0x02010000
	s_mov_b32 s6, 0
	s_mov_b32 s7, 0
	s_buffer_load_b64 s[8:9], s[4:7], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s8
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_buffer_load_unrepresentable_base_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 10
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
      - { .offset:         0, .size:          16, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 256
    .name:           s_buffer_load_success_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .symbol:         s_buffer_load_success_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 256
    .name:           s_buffer_load_scope_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         s_buffer_load_scope_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 256
    .name:           s_buffer_load_unrepresentable_base_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         s_buffer_load_unrepresentable_base_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
