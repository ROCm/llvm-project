; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: env HSA_HOTSWAP_STRICT=1 %not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=kernarg_rebased_sload_b128 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STRICT
; RUN: raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=kernarg_rebased_sload_b128 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=PERMISSIVE
;
; Triton may load an explicit by-value argument pointer into the physical
; kernarg SGPR pair, rebase it, and then issue more s_load_b128 operations
; through that pointer. Without a dataflow proof that the new value is
; definitely non-entry, strict mode refuses instead of treating a physical SGPR
; write as proof that source hidden-arg handling is no longer relevant.

; STRICT: implicit-arg offsets may be applied to the target runtime hidden-arg block on some CFG paths

; PERMISSIVE-LABEL: define amdgpu_kernel void @kernarg_rebased_sload_b128(
; PERMISSIVE-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; PERMISSIVE: inttoptr i64
; PERMISSIVE: load i32, ptr addrspace(1)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	kernarg_rebased_sload_b128
	.p2align	8
	.type	kernarg_rebased_sload_b128,@function
kernarg_rebased_sload_b128:
	s_load_b128 s[0:3], s[0:1], 0x0 nv
	s_wait_kmcnt 0x0
	s_mov_b32 s6, 64
	s_mov_b32 s7, 0
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]
	s_cmp_eq_u32 s6, 64
	s_cbranch_scc1 .Lload
	s_nop 0
.Lload:
	s_load_b128 s[4:7], s[0:1], 0x30
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s4
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_rebased_sload_b128
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 4
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           40
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 32
    .name:           kernarg_rebased_sload_b128
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         kernarg_rebased_sload_b128.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
