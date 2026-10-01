; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_scope | %FileCheck %s

; An access whose scope reaches past its own compute unit takes part in a
; handshake with another workgroup, so it lifts to a volatile one and the
; optimizer may neither drop nor reorder it. SCOPE_CU shares its encoding with
; a default cache policy, so the plain accesses below cover it.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	global_scope
	.p2align	8
	.type	global_scope,@function
; CHECK-LABEL: define amdgpu_kernel void @global_scope(
global_scope:
; CHECK: load volatile i32, ptr addrspace(1) {{%.+}}, align 4
	global_load_b32 v1, v0, s[0:1] scope:SCOPE_DEV

; CHECK: load volatile i64, ptr addrspace(1) {{%.+}}, align 4
	global_load_b64 v[2:3], v0, s[0:1] scope:SCOPE_SYS

; CHECK: load volatile <3 x i32>, ptr addrspace(1) {{%.+}}, align 4
	global_load_b96 v[4:6], v0, s[0:1] scope:SCOPE_DEV

; The scope survives alongside the addressing modifiers, which the corpus
; pairs it with.
; CHECK: [[SCALE:%.+]] = mul i64 {{%.+}}, 16
; CHECK: [[PTR:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; CHECK: load volatile <4 x i32>, ptr addrspace(1) [[PTR]], align 4
	global_load_b128 v[8:11], v0, s[0:1] scale_offset scope:SCOPE_DEV

; CHECK: store volatile i32 {{.+}}, ptr addrspace(1) {{%.+}}, align 4
	global_store_b32 v0, v1, s[0:1] scope:SCOPE_DEV

; CHECK: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) {{%.+}}, i64 32
; CHECK: store volatile i64 {{.+}}, ptr addrspace(1) [[OFFSET]], align 4
	global_store_b64 v0, v[2:3], s[0:1] offset:32 scope:SCOPE_DEV

; CHECK: store volatile <4 x i32> {{.+}}, ptr addrspace(1) {{%.+}}, align 4
	global_store_b128 v0, v[8:11], s[0:1] scope:SCOPE_SYS

; A default cache policy keeps the plain, optimizable access.
; CHECK: load i32, ptr addrspace(1) {{%.+}}, align 4
	global_load_b32 v1, v0, s[0:1]

; CHECK: store i32 {{.+}}, ptr addrspace(1) {{%.+}}, align 4
	global_store_b32 v0, v1, s[0:1]
; CHECK: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_scope
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           global_scope
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_scope.kd
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
