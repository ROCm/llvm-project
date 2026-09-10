; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --dump-meta | %FileCheck %s

; --dump-meta exercises the code-object metadata extraction: the ISA from the
; ELF e_flags, the per-kernel ABI from the MsgPack notes (including a
; global_buffer argument with a string .address_space and a by_value argument
; with none), the descriptor register fields read from the meta_kernel.kd blob
; in .rodata, the .text extent, and the .text bytes.
; CHECK: isa: amdgcn-amd-amdhsa--gfx942
; CHECK-NEXT: kernel: meta_kernel kernarg=16 group=32 maxflat=256 has_kd=1 rsrc1=0x00ac0040 rsrc2=0x00000086 code_props=0x0008 preload=0x0001 extent_size=4
; CHECK-NEXT: arg: name=out offset=0 size=8 kind=global_buffer address_space=global
; CHECK-NEXT: arg: name=scale offset=8 size=4 kind=by_value address_space=<none>
; CHECK-NEXT: text_bytes: 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	meta_kernel
	.p2align	8
	.type	meta_kernel,@function
meta_kernel:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel meta_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_group_segment_fixed_size 32
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_kernarg_preload_length 1
		.amdhsa_user_sgpr_kernarg_preload_offset 0
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .name: out
        .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: scale
        .offset: 8
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 32
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 256
    .name:           meta_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         meta_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
