; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: env HSA_HOTSWAP_STRICT=1 raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:   --emit-ir=kernarg_runtime_offset_ordinary_strict \
; RUN:   | %FileCheck %s --check-prefix=STRICT

; Kernarg load through a register (non-immediate) offset.
; STRICT-LABEL: define amdgpu_kernel void @kernarg_runtime_offset_ordinary_strict(
; STRICT-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; STRICT: inttoptr i64
; STRICT: load i32, ptr addrspace(1)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	kernarg_runtime_offset_ordinary_strict
	.p2align	8
	.type	kernarg_runtime_offset_ordinary_strict,@function
kernarg_runtime_offset_ordinary_strict:
	s_and_b32 s4, ttmp9, 0xffff
	s_load_b32 s2, s[0:1], s4
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s2
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_runtime_offset_ordinary_strict
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 10
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           kernarg_runtime_offset_ordinary_strict
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         kernarg_runtime_offset_ordinary_strict.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
