; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -defsym=CASE=0 -filetype=obj %s -o %t.0.o
; RUN: %ld.lld -shared %t.0.o -o %t.0.hsaco
; RUN: not %transpile_cli %t.0.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=LAYOUT
; LAYOUT: buffer source descriptor layout is not modeled
; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -defsym=CASE=1 -filetype=obj %s -o %t.1.o
; RUN: %ld.lld -shared %t.1.o -o %t.1.hsaco
; RUN: not %transpile_cli %t.1.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=FORMAT
; FORMAT: unsupported buffer opcode or addressing form
; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -defsym=CASE=2 -filetype=obj %s -o %t.2.o
; RUN: %ld.lld -shared %t.2.o -o %t.2.hsaco
; RUN: not %transpile_cli %t.2.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=LDS
; LDS: unsupported buffer opcode or addressing form
; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -defsym=CASE=3 -filetype=obj %s -o %t.3.o
; RUN: %ld.lld -shared %t.3.o -o %t.3.hsaco
; RUN: not %transpile_cli %t.3.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=TYPED
; TYPED: unsupported-instruction-form: tbuffer_load_format_x [Unknown]
; RUN: not %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=4 -filetype=null %s 2>&1 | %FileCheck %s --check-prefix=ADDR64
; ADDR64: error: invalid operand for instruction
; ADDR64-NEXT: buffer_load_dword v2, v[0:1], s[4:7], 0 addr64

.amdhsa_code_object_version 6
.text
.globl buffer_neighbor
.p2align 8
.type buffer_neighbor,@function
buffer_neighbor:
.if CASE == 0
  buffer_load_dword v1, off, s[4:7], 0
.elseif CASE == 1
  buffer_load_format_x v1, off, s[4:7], 0
.elseif CASE == 2
  buffer_load_dword off, s[4:7], s8 lds
.elseif CASE == 3
  tbuffer_load_format_x v1, off, s[4:7], 0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT]
.else
  buffer_load_dword v2, v[0:1], s[4:7], 0 addr64
.endif
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.if CASE != 4
.amdhsa_kernel buffer_neighbor
  .amdhsa_accum_offset 4
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 9
.end_amdhsa_kernel
.else
.amdhsa_kernel buffer_neighbor
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 9
.end_amdhsa_kernel
.endif
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: buffer_neighbor
    .symbol: buffer_neighbor.kd
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 9
    .vgpr_count: 3
    .wavefront_size: 64
...
.end_amdgpu_metadata
