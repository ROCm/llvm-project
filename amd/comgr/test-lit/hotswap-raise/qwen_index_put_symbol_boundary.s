; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=qwen_index_put_boundary_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; PyTorch boolean-mask index_put objects can place many related kernels in one
; `.text` section. The selected symbol ends with a backward branch and no
; trailing `s_endpgm`; a later, different put/take symbol contains
; `global_atomic_cmpswap_b32`. Decoding must stop at the selected symbol's ELF
; size, otherwise obstruction analysis reports the later symbol's CAS as if it
; belonged to the selected index_put kernel.
;
; CHECK-LABEL: define amdgpu_kernel void @qwen_index_put_boundary_kernel
; CHECK-NOT: NonCommutativeAtomic
; CHECK-NOT: global_atomic_cmpswap

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	qwen_index_put_boundary_kernel
	.p2align	8
	.type	qwen_index_put_boundary_kernel,@function
qwen_index_put_boundary_kernel:
	s_branch -1
.Lqwen_index_put_boundary_end:
	.size	qwen_index_put_boundary_kernel, .Lqwen_index_put_boundary_end-qwen_index_put_boundary_kernel

	.globl	qwen_unselected_later_cas_kernel
	.p2align	8
	.type	qwen_unselected_later_cas_kernel,@function
qwen_unselected_later_cas_kernel:
	v_mov_b64_e32 v[0:1], 0
	v_mov_b64_e32 v[2:3], 0
	global_atomic_cmpswap_b32 v2, v[0:1], s[0:1] scope:SCOPE_DEV
	s_endpgm
	.size	qwen_unselected_later_cas_kernel, .-qwen_unselected_later_cas_kernel

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel qwen_index_put_boundary_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
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
    .max_flat_workgroup_size: 256
    .name:           qwen_index_put_boundary_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         qwen_index_put_boundary_kernel.kd
    .vgpr_count:     0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
