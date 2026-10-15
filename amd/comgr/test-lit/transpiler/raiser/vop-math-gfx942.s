; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=legacy_float | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	legacy_float
	.p2align	8
	.type	legacy_float,@function
; CHECK-LABEL: define amdgpu_kernel void @legacy_float(
legacy_float:
; CHECK: call float @llvm.maximumnum.f32
; CHECK: and i32 {{.+}}, 4194304
; CHECK: select i1 {{.+}}, float {{.+}}, float
	v_max3_f32 v0, v1, v2, v3
	global_store_dword v[4:5], v0, off
; CHECK: call float @llvm.minimumnum.f32
; CHECK: and i32 {{.+}}, 4194304
; CHECK: select i1 {{.+}}, float {{.+}}, float
	v_min3_f32 v0, v1, v2, v3
	global_store_dword v[4:5], v0, off
; CHECK: call float @llvm.amdgcn.fmed3.f32
; CHECK: call float @llvm.minimumnum.f32
; CHECK: select i1 {{.+}}, float {{.+}}, float
	v_med3_f32 v0, v1, v2, v3
	global_store_dword v[4:5], v0, off
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel legacy_float
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_ieee_mode 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           legacy_float
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         legacy_float.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
