; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir | %opt -passes=verify -disable-output
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch > %t.ll
; RUN: %opt -passes=verify -disable-output %t.ll
; RUN: %FileCheck %s < %t.ll

; CHECK: ; launch: ord\0Anary kind=unchanged max_workgroup_size=1024
; CHECK-LABEL: define amdgpu_kernel void @"ord\0Anary"(

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text
.globl ordinary
.p2align 8
.type ordinary,@function
ordinary:
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ordinary
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: "ord\nnary"
    .symbol: ordinary.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 1
    .vgpr_count: 1
    .wavefront_size: 32
...
.end_amdgpu_metadata
