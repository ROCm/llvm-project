; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CASE=0 \
; RUN:   %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_invalid 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=POLICY

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CASE=1 \
; RUN:   %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_invalid 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=POLICY

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CASE=2 \
; RUN:   %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_invalid 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=POLICY

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CASE=4 \
; RUN:   %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_invalid 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=POLICY

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CASE=5 \
; RUN:   %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_invalid 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=POLICY

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CASE=7 \
; RUN:   %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_invalid 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=POLICY

; POLICY: in kernel 'global_subdword_invalid'
; POLICY-SAME: non-default cache policy is not modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CASE=8 \
; RUN:   %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_invalid 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=D16
; D16: in kernel 'global_subdword_invalid'
; D16-SAME: unsupported flat memory operation

.amdhsa_code_object_version 6
.text
.globl global_subdword_invalid
.p2align 8
.type global_subdword_invalid,@function
global_subdword_invalid:

.if CASE == 0
global_load_u8 v4, v0, s[0:1] scope:SCOPE_DEV th:TH_LOAD_NT
.endif
.if CASE == 1
global_load_i8 v4, v0, s[0:1] scale_offset scope:SCOPE_SYS th:TH_LOAD_NT
.endif
.if CASE == 2
global_load_u16 v4, v0, s[0:1] th:TH_LOAD_NT
.endif
.if CASE == 4
global_store_b8 v0, v4, s[0:1] scope:SCOPE_DEV th:TH_STORE_NT
.endif
.if CASE == 5
global_store_b16 v0, v4, s[0:1] scale_offset th:TH_STORE_NT
.endif
.if CASE == 7
global_store_d16_hi_b16 v0, v4, s[0:1] scale_offset scope:SCOPE_DEV th:TH_STORE_NT
.endif
.if CASE == 8
global_load_d16_u8 v4, v0, s[0:1]
.endif
s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel global_subdword_invalid
  .amdhsa_kernarg_size 0
  .amdhsa_next_free_vgpr 8
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
    .name: global_subdword_invalid
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: global_subdword_invalid.kd
    .vgpr_count: 8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
