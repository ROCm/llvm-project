; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && env HSA_HOTSWAP_STRICT=1 raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --assume-hip-global-offset-zero \
; RUN:     --emit-ir=kernarg_unreachable_fallthrough 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Dead bytes after an unconditional branch must not be included in the
; predecessor block's kernarg-pointer provenance summary. The live target still
; sees the entry kernarg pointer, so strict mode may synthesize the known source
; hidden_global_offset field instead of treating the target as clobbered.

; CHECK-LABEL: define amdgpu_kernel void @kernarg_unreachable_fallthrough(
; CHECK-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; CHECK: phi i32 [ 0, %{{[a-zA-Z_0-9]+}} ], [ %tid, %{{[a-zA-Z_0-9]+}} ]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kernarg_unreachable_fallthrough
	.p2align	8
	.type	kernarg_unreachable_fallthrough,@function
kernarg_unreachable_fallthrough:
	s_branch .Ltarget
	s_load_b64 s[0:1], s[0:1], 0
.Ltarget:
	s_load_b64 s[4:5], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x8
	s_wait_kmcnt 0
	v_dual_mov_b32 v0, s2 :: v_dual_mov_b32 v1, 0
	global_store_b32 v1, v0, s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_unreachable_fallthrough
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           kernarg_unreachable_fallthrough
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         kernarg_unreachable_fallthrough.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
