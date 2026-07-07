; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_load_saddr_nonkernarg_after_merge_kernel 2>/dev/null | %FileCheck %s

; global_load_b32 SADDR from non-kernarg (merged) base: saddr_vaddr synthesis.
; CHECK-LABEL: define amdgpu_kernel void @global_load_saddr_nonkernarg_after_merge_kernel(
; CHECK: saddr_vaddr
; CHECK: load float, ptr addrspace(1)
; CHECK: store i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_saddr_nonkernarg_after_merge_kernel
	.p2align	8
	.type	global_load_saddr_nonkernarg_after_merge_kernel,@function
global_load_saddr_nonkernarg_after_merge_kernel:
	s_load_b512 s[4:19], s[0:1], 0x0
	s_load_b256 s[20:27], s[0:1], 0x58
	s_wait_kmcnt 0x0
	s_cmp_eq_u32 s4, s4
	s_cbranch_scc1 .Lmerge
	s_mov_b32 s30, 0
.Lmerge:
	v_mov_b32_e32 v5, 0
	s_add_nc_u64 s[0:1], s[10:11], s[20:21]
	global_load_b32 v0, v5, s[0:1]
	s_wait_loadcnt 0x0
	global_store_b32 v0, v0, s[18:19]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_saddr_nonkernarg_after_merge_kernel
		.amdhsa_kernarg_size 120
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 31
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
      - { .actual_access:  read_only,  .address_space:  global, .offset:         0, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:         8, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:        16, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:        24, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:        32, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:        40, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:        48, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  write_only, .address_space:  global, .offset:        56, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:        88, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:        96, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:       104, .size: 8, .value_kind: global_buffer }
      - { .actual_access:  read_only,  .address_space:  global, .offset:       112, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 120
    .max_flat_workgroup_size: 1024
    .name:           global_load_saddr_nonkernarg_after_merge_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         global_load_saddr_nonkernarg_after_merge_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
