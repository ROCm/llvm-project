; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>/dev/null | %FileCheck %s
;
; Lift test for the LDS ds_add F64 atomics: the no-return form (ds_add_f64)
; and the return form (ds_add_rtn_f64), which additionally writes back the
; pre-op value. Both lift to an addrspace(3) `atomicrmw fadd double`.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_add_f64_kernel
	.p2align	8
	.type	ds_add_f64_kernel,@function
ds_add_f64_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0x3ff00000
; CHECK-LABEL: define amdgpu_kernel void @ds_add_f64_kernel(
; No-return form: plain addrspace(3) atomicrmw fadd.
; CHECK: atomicrmw fadd ptr addrspace(3) %{{[^,]+}}, double %{{[^ ]+}}
	ds_add_f64 v0, v[2:3]
; Return form: same atomicrmw, plus the returned value bitcast back to i64.
; CHECK: %[[RMW:.+]] = atomicrmw fadd ptr addrspace(3) %{{[^,]+}}, double %{{[^ ]+}}
; CHECK: bitcast double %[[RMW]] to i64
	ds_add_rtn_f64 v[4:5], v0, v[2:3]
	s_endpgm
; CHECK-NOT: atomicrmw fadd float
; CHECK-NOT: atomicrmw fadd ptr addrspace(0)
; CHECK-NOT: atomicrmw fadd ptr addrspace(1)
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_add_f64_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_group_segment_fixed_size 16
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 16
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           ds_add_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         ds_add_f64_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
