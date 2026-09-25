; REQUIRES: comgr-has-transpiler

; RUN: set -e; for case in LOAD_SCOPE LOAD_SCALED_SCOPE STORE_TH STORE_SCALED_TH NV SCALED_NV; do \
; RUN:   %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=$case=1 %s -o %t.o; \
; RUN:   %ld.lld -shared %t.o -o %t.hsaco; \
; RUN:   not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_invalid 2>&1 \
; RUN:     | %FileCheck %s --check-prefix=POLICY; \
; RUN: done
; POLICY: in kernel 'global_invalid'
; POLICY-SAME: non-default cache policy is not modeled

; RUN: set -e; for case in LOAD_NO_SADDR STORE_NO_SADDR; do \
; RUN:   %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=$case=1 %s -o %t.o; \
; RUN:   %ld.lld -shared %t.o -o %t.hsaco; \
; RUN:   not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_invalid 2>&1 \
; RUN:     | %FileCheck %s --check-prefix=NO-SADDR; \
; RUN: done
; NO-SADDR: in kernel 'global_invalid'
; NO-SADDR-SAME: scale_offset requires an saddr base

; RUN: set -e; for case in SUBDWORD ATOMIC; do \
; RUN:   %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=$case=1 %s -o %t.o; \
; RUN:   %ld.lld -shared %t.o -o %t.hsaco; \
; RUN:   not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_invalid 2>&1 \
; RUN:     | %FileCheck %s --check-prefix=OPERATION; \
; RUN: done
; OPERATION: in kernel 'global_invalid'
; OPERATION-SAME: unsupported flat memory operation

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl global_invalid
	.p2align 8
	.type global_invalid,@function
global_invalid:
.ifdef LOAD_SCOPE
	global_load_b32 v1, v0, s[0:1] scope:SCOPE_DEV
.endif
.ifdef LOAD_SCALED_SCOPE
	global_load_b32 v1, v0, s[0:1] scale_offset scope:SCOPE_DEV
.endif
.ifdef STORE_TH
	global_store_b64 v0, v[2:3], s[0:1] th:TH_STORE_NT
.endif
.ifdef STORE_SCALED_TH
	global_store_b64 v0, v[2:3], s[0:1] scale_offset th:TH_STORE_NT
.endif
.ifdef NV
	global_load_b32 v1, v0, s[0:1] nv
.endif
.ifdef SCALED_NV
	global_load_b32 v1, v0, s[0:1] scale_offset nv
.endif
; LLVM's assembler rejects scale_offset without SADDR. Use raw encodings to
; exercise the raiser's refusal for these forms.
.ifdef LOAD_NO_SADDR
; global_load_b32 v1, v[2:3], off scale_offset
	.long 0xee05007c, 0x00010001, 0x00000002
.endif
.ifdef STORE_NO_SADDR
; global_store_b32 v[2:3], v1, off scale_offset
	.long 0xee06807c, 0x00810000, 0x00000002
.endif
.ifdef SUBDWORD
	global_load_u16 v1, v0, s[0:1] scale_offset
.endif
.ifdef ATOMIC
	global_atomic_add_u32 v0, v1, s[0:1] scale_offset
.endif
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel global_invalid
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: global_invalid
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: global_invalid.kd
    .vgpr_count: 4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
