; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:        --emit-ir=global_inv_dev_kernel,global_inv_sys_kernel,global_inv_cu_kernel \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; RUN: %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_inv_se_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-SE

; global_inv scope lowering: SCOPE_DEV -> agent-scoped acquire fence;
; SCOPE_SYS -> system-scoped acquire fence; SCOPE_CU no-op; SCOPE_SE refused.
; Mirrors global_wb (the release/writeback counterpart). The three lowerable
; scopes are raised in a single run (one kernel each) so per-kernel CHECK-LABELs
; anchor them; SCOPE_SE keeps its own %not run because it aborts the raise.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

; IR-LABEL: define amdgpu_kernel void @global_inv_dev_kernel(
; IR: fence syncscope("agent") acquire
	.globl	global_inv_dev_kernel
	.p2align	8
	.type	global_inv_dev_kernel,@function
global_inv_dev_kernel:
	global_inv scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	s_endpgm

; IR-LABEL: define amdgpu_kernel void @global_inv_sys_kernel(
; IR: fence acquire
	.globl	global_inv_sys_kernel
	.p2align	8
	.type	global_inv_sys_kernel,@function
global_inv_sys_kernel:
	global_inv scope:SCOPE_SYS
	s_endpgm

; IR-LABEL: define amdgpu_kernel void @global_inv_cu_kernel(
; IR-NOT: fence
	.globl	global_inv_cu_kernel
	.p2align	8
	.type	global_inv_cu_kernel,@function
global_inv_cu_kernel:
	global_inv
	s_endpgm

; REFUSE-SE: failed to raise: unsupported-instruction-form: global_inv [FLAT]
; REFUSE-SE: SCOPE_SE cannot be represented
	.globl	global_inv_se_kernel
	.p2align	8
	.type	global_inv_se_kernel,@function
global_inv_se_kernel:
	global_inv scope:SCOPE_SE
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_inv_dev_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel

	.amdhsa_kernel global_inv_sys_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel

	.amdhsa_kernel global_inv_cu_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel

	.amdhsa_kernel global_inv_se_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel

	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: global_inv_dev_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: global_inv_dev_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: global_inv_sys_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: global_inv_sys_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: global_inv_cu_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: global_inv_cu_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: global_inv_se_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: global_inv_se_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
