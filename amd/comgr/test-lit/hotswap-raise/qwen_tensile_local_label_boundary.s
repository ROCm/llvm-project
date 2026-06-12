; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=qwen_tensile_local_label_boundary_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Tensile code objects often set the public kernel function's ELF size to zero
; and use many local labels inside the function body. The symbol-extent fallback
; must bound decode by the next metadata kernel symbol, not by the next local
; label or helper function, or only part of the selected kernel is raised.
;
; CHECK-LABEL: define amdgpu_kernel void @qwen_tensile_local_label_boundary_kernel
; CHECK: store
; CHECK-NOT: NonCommutativeAtomic
; CHECK-NOT: global_atomic_cmpswap
; CHECK: ret void

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	qwen_tensile_local_label_boundary_kernel
	.p2align	8
	.type	qwen_tensile_local_label_boundary_kernel,@function
qwen_tensile_local_label_boundary_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
.Linternal_label:
	v_mov_b32_e32 v0, 7
	s_wait_kmcnt 0x0
	global_store_b32 v0, v0, s[2:3]

	.globl	qwen_tensile_next_kernel
	.p2align	8
	.type	qwen_tensile_next_kernel,@function
qwen_tensile_next_kernel:
	v_mov_b64_e32 v[0:1], 0
	v_mov_b64_e32 v[2:3], 0
	global_atomic_cmpswap_b32 v2, v[0:1], s[0:1] scope:SCOPE_DEV
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel qwen_tensile_local_label_boundary_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
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
    .max_flat_workgroup_size: 256
    .name:           qwen_tensile_local_label_boundary_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         qwen_tensile_local_label_boundary_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           qwen_tensile_next_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         qwen_tensile_next_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
