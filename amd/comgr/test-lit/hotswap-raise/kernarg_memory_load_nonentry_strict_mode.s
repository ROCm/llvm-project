; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: env HSA_HOTSWAP_STRICT=1 raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=kernarg_memory_load_nonentry_strict_mode \
; RUN:   | %FileCheck %s --check-prefix=STRICT
;
; Models a Triton pattern where an explicit pointer argument is loaded through
; the entry kernarg pair and reused in the same physical SGPR pair. After the
; first load, s[0:1] carries a pointer value read from the kernarg buffer, so
; the later high-offset load must use ordinary memory lowering rather than
; source hidden-arg lowering.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	kernarg_memory_load_nonentry_strict_mode
	.p2align	8
	.type	kernarg_memory_load_nonentry_strict_mode,@function
; STRICT-LABEL: define amdgpu_kernel void @kernarg_memory_load_nonentry_strict_mode(
kernarg_memory_load_nonentry_strict_mode:
	s_load_b128 s[0:3], s[0:1], 0x0 nv
	s_wait_kmcnt 0x0
; STRICT-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; STRICT: inttoptr i64
; STRICT: load i32, ptr addrspace(1)
	s_load_b64 s[0:1], s[0:1], 0x90
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_memory_load_nonentry_strict_mode
		.amdhsa_kernarg_size 32
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
      - .offset:         0
        .size:           32
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 32
    .name:           kernarg_memory_load_nonentry_strict_mode
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         kernarg_memory_load_nonentry_strict_mode.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
