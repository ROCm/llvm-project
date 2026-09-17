; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s

.amdhsa_code_object_version 6
.text
.globl buffer_bounds
.p2align 8
.type buffer_bounds,@function
; CHECK-LABEL: define amdgpu_kernel void @buffer_bounds(
buffer_bounds:
  s_load_b128 s[4:7], s[0:1], 0
  s_load_b128 s[8:11], s[0:1], 16
  s_wait_kmcnt 0
; CHECK: and i32 {{.+}}, 63
  s_and_b32 s7, s7, 63
  s_mov_b32 exec_lo, 1
  v_mov_b32 v0, 0
  v_mov_b32 v1, s11
; CHECK: [[SCALAR:%.+]] = zext i32 {{.+}} to i64
; CHECK-NEXT: [[IMMEDIATE:%.+]] = add i64 [[SCALAR]], 3
; CHECK-NEXT: [[VECTOR:%.+]] = zext i32 {{.+}} to i64
; CHECK-NEXT: [[OFFSET:%.+]] = add i64 [[IMMEDIATE]], [[VECTOR]]
; CHECK: [[BASEHI:%.+]] = and i32 {{.+}}, 33554431
; CHECK: [[BASEEXT:%.+]] = zext i32 [[BASEHI]] to i64
; CHECK: [[BASESHIFT:%.+]] = shl i64 [[BASEEXT]], 32
; CHECK: [[BASE:%.+]] = or i64 {{.+}}, [[BASESHIFT]]
; CHECK: [[EXTENTLO:%.+]] = lshr i32 {{.+}}, 25
; CHECK: [[EXTENTLOEXT:%.+]] = zext i32 [[EXTENTLO]] to i64
; CHECK: [[EXTENTMID:%.+]] = shl i64 {{.+}}, 7
; CHECK: [[EXTENTLOW:%.+]] = or i64 [[EXTENTLOEXT]], [[EXTENTMID]]
; CHECK: [[EXTENTHI:%.+]] = and i32 {{.+}}, 63
; CHECK: [[EXTENTHIEXT:%.+]] = zext i32 [[EXTENTHI]] to i64
; CHECK: [[EXTENTHISHIFT:%.+]] = shl i64 [[EXTENTHIEXT]], 39
; CHECK: [[EXTENT:%.+]] = or i64 [[EXTENTLOW]], [[EXTENTHISHIFT]]
; CHECK: [[UNBOUNDED:%.+]] = icmp eq i64 [[EXTENT]], 35184372088831
; CHECK: [[END0:%.+]] = add i64 {{.+}}, 4
; CHECK: [[VALID0:%.+]] = icmp ult i64 [[END0]], [[EXTENT]]
; CHECK: [[LOAD0:%.+]] = or i1 [[UNBOUNDED]], [[VALID0]]
; CHECK: br i1 [[LOAD0]], label {{.+}}, label {{.+}}
; CHECK: load i32, ptr addrspace(1) {{.+}}, align 1
; CHECK: phi i32 [ 0, {{.+}} ], [ {{.+}} ]
; CHECK: icmp ult i64 {{.+}}, [[EXTENT]]
; CHECK: load i32, ptr addrspace(1) {{.+}}, align 1
; CHECK: phi i32 [ 0, {{.+}} ], [ {{.+}} ]
; CHECK: icmp ult i64 {{.+}}, [[EXTENT]]
; CHECK: load i32, ptr addrspace(1) {{.+}}, align 1
; CHECK: phi i32 [ 0, {{.+}} ], [ {{.+}} ]
; CHECK: icmp ult i64 {{.+}}, [[EXTENT]]
; CHECK: load i32, ptr addrspace(1) {{.+}}, align 1
; CHECK: phi i32 [ 0, {{.+}} ], [ {{.+}} ]
  buffer_load_b128 v[4:7], v1, s[4:7], s10 offen offset:3
  s_wait_loadcnt 0
  global_store_b32 v0, v4, s[8:9]
  global_store_b32 v0, v5, s[8:9] offset:4
  global_store_b32 v0, v6, s[8:9] offset:8
  global_store_b32 v0, v7, s[8:9] offset:12
  v_mov_b32 v4, 0x12345678
  v_mov_b32 v5, 0xabcdef01
  v_mov_b32 v6, 0x76543210
  v_mov_b32 v7, 0xfedcba98
; CHECK: icmp ult i64 {{.+}}, {{.+}}
; CHECK: br i1 {{.+}}, label {{.+}}, label {{.+}}
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
; CHECK: icmp ult i64 {{.+}}, {{.+}}
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
; CHECK: icmp ult i64 {{.+}}, {{.+}}
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
; CHECK: icmp ult i64 {{.+}}, {{.+}}
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b128 v[4:7], v1, s[4:7], s10 offen offset:3
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel buffer_bounds
  .amdhsa_kernarg_size 32
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: buffer_bounds
    .symbol: buffer_bounds.kd
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .sgpr_count: 12
    .vgpr_count: 8
    .wavefront_size: 32
...
.end_amdgpu_metadata
