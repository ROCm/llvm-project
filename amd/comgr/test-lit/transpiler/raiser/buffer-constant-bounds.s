; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=0 -filetype=obj %s -o %t.0.o
; RUN: %ld.lld -shared %t.0.o -o %t.0.hsaco
; RUN: %transpile_cli %t.0.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ZERO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=9 -filetype=obj %s -o %t.1.o
; RUN: %ld.lld -shared %t.1.o -o %t.1.hsaco
; RUN: %transpile_cli %t.1.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ZERO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=10 -filetype=obj %s -o %t.2.o
; RUN: %ld.lld -shared %t.2.o -o %t.2.hsaco
; RUN: %transpile_cli %t.2.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ZERO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=11 -filetype=obj %s -o %t.3.o
; RUN: %ld.lld -shared %t.3.o -o %t.3.hsaco
; RUN: %transpile_cli %t.3.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ONE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=14 -filetype=obj %s -o %t.4.o
; RUN: %ld.lld -shared %t.4.o -o %t.4.hsaco
; RUN: %transpile_cli %t.4.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ONE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=15 -filetype=obj %s -o %t.5.o
; RUN: %ld.lld -shared %t.5.o -o %t.5.hsaco
; RUN: %transpile_cli %t.5.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,TWO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=18 -filetype=obj %s -o %t.6.o
; RUN: %ld.lld -shared %t.6.o -o %t.6.hsaco
; RUN: %transpile_cli %t.6.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,TWO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=19 -filetype=obj %s -o %t.7.o
; RUN: %ld.lld -shared %t.7.o -o %t.7.hsaco
; RUN: %transpile_cli %t.7.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,THREE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=22 -filetype=obj %s -o %t.8.o
; RUN: %ld.lld -shared %t.8.o -o %t.8.hsaco
; RUN: %transpile_cli %t.8.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,THREE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=23 -filetype=obj %s -o %t.9.o
; RUN: %ld.lld -shared %t.9.o -o %t.9.hsaco
; RUN: %transpile_cli %t.9.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,FOUR
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=0x8000000000 -filetype=obj %s -o %t.10.o
; RUN: %ld.lld -shared %t.10.o -o %t.10.hsaco
; RUN: %transpile_cli %t.10.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,FOUR
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=0x1ffffffffffe -filetype=obj %s -o %t.11.o
; RUN: %ld.lld -shared %t.11.o -o %t.11.hsaco
; RUN: %transpile_cli %t.11.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,FOUR
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=0x1fffffffffff -filetype=obj %s -o %t.12.o
; RUN: %ld.lld -shared %t.12.o -o %t.12.hsaco
; RUN: %transpile_cli %t.12.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,FOUR

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=10 -defsym=WIDTH=64 -filetype=obj %s -o %t.64.10.o
; RUN: %ld.lld -shared %t.64.10.o -o %t.64.10.hsaco
; RUN: %transpile_cli %t.64.10.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ZERO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=11 -defsym=WIDTH=64 -filetype=obj %s -o %t.64.11.o
; RUN: %ld.lld -shared %t.64.11.o -o %t.64.11.hsaco
; RUN: %transpile_cli %t.64.11.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ONE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=14 -defsym=WIDTH=96 -filetype=obj %s -o %t.96.14.o
; RUN: %ld.lld -shared %t.96.14.o -o %t.96.14.hsaco
; RUN: %transpile_cli %t.96.14.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ONE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=15 -defsym=WIDTH=96 -filetype=obj %s -o %t.96.15.o
; RUN: %ld.lld -shared %t.96.15.o -o %t.96.15.hsaco
; RUN: %transpile_cli %t.96.15.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,TWO

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=10 -defsym=OFFEN=0 -filetype=obj %s -o %t.offset.10.o
; RUN: %ld.lld -shared %t.offset.10.o -o %t.offset.10.hsaco
; RUN: %transpile_cli %t.offset.10.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ZERO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=11 -defsym=OFFEN=0 -filetype=obj %s -o %t.offset.11.o
; RUN: %ld.lld -shared %t.offset.11.o -o %t.offset.11.hsaco
; RUN: %transpile_cli %t.offset.11.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,ONE

.ifndef WIDTH
.set WIDTH, 128
.endif
.ifndef OFFEN
.set OFFEN, 1
.endif
.amdhsa_code_object_version 6
.text
.globl buffer_constant_bounds
.p2align 8
.type buffer_constant_bounds,@function
; CHECK-LABEL: define amdgpu_kernel void @buffer_constant_bounds(
buffer_constant_bounds:
  s_load_b64 s[8:9], s[0:1], 0
; CHECK: fence
  s_wait_kmcnt 0
  s_mov_b32 s4, 4096
  s_mov_b32 s5, 0x1000000 | ((EXTENT & 127) << 25)
  s_mov_b32 s6, (EXTENT >> 7) & 0xffffffff
  s_mov_b32 s7, EXTENT >> 39
.if OFFEN
  s_mov_b32 s10, 1
.else
  s_mov_b32 s10, 3
.endif
  v_mov_b32 v0, 0
  v_mov_b32 v1, 2
  v_mov_b32 v6, 0
  v_mov_b32 v7, 0
; ZERO-NOT: load i32, ptr addrspace(1)
; ONE: load i32, ptr addrspace(1) inttoptr (i64 72057594037932038 to ptr addrspace(1)), align 1
; ONE-NOT: load i32, ptr addrspace(1)
; TWO-COUNT-2: load i32, ptr addrspace(1) inttoptr (i64 {{.+}} to ptr addrspace(1)), align 1
; TWO-NOT: load i32, ptr addrspace(1)
; THREE-COUNT-3: load i32, ptr addrspace(1) inttoptr (i64 {{.+}} to ptr addrspace(1)), align 1
; THREE-NOT: load i32, ptr addrspace(1)
; FOUR-COUNT-4: load i32, ptr addrspace(1) inttoptr (i64 {{.+}} to ptr addrspace(1)), align 1
; FOUR-NOT: load i32, ptr addrspace(1)
.if !OFFEN
  buffer_load_b128 v[4:7], off, s[4:7], s10 offset:3
.elseif WIDTH == 64
  buffer_load_b64 v[4:5], v1, s[4:7], s10 offen offset:3
.elseif WIDTH == 96
  buffer_load_b96 v[4:6], v1, s[4:7], s10 offen offset:3
.else
  buffer_load_b128 v[4:7], v1, s[4:7], s10 offen offset:3
.endif
  s_wait_loadcnt 0
; CHECK: fence
; ZERO-COUNT-4: store i32 0, ptr addrspace(1)
; ONE: store i32 {{.+}}, ptr addrspace(1)
; ONE-COUNT-3: store i32 0, ptr addrspace(1)
; TWO-COUNT-2: store i32 {{.+}}, ptr addrspace(1)
; TWO-COUNT-2: store i32 0, ptr addrspace(1)
; THREE-COUNT-3: store i32 {{.+}}, ptr addrspace(1)
; THREE: store i32 0, ptr addrspace(1)
; FOUR-COUNT-4: store i32 {{.+}}, ptr addrspace(1)
  global_store_b32 v0, v4, s[8:9]
  global_store_b32 v0, v5, s[8:9] offset:4
  global_store_b32 v0, v6, s[8:9] offset:8
  global_store_b32 v0, v7, s[8:9] offset:12
  v_mov_b32 v4, 42
  v_mov_b32 v5, 42
  v_mov_b32 v6, 42
  v_mov_b32 v7, 42
; ONE: store i32 42, ptr addrspace(1) inttoptr (i64 72057594037932038 to ptr addrspace(1)), align 1
; TWO-COUNT-2: store i32 42, ptr addrspace(1) inttoptr (i64 {{.+}} to ptr addrspace(1)), align 1
; THREE-COUNT-3: store i32 42, ptr addrspace(1) inttoptr (i64 {{.+}} to ptr addrspace(1)), align 1
; FOUR-COUNT-4: store i32 42, ptr addrspace(1) inttoptr (i64 {{.+}} to ptr addrspace(1)), align 1
; CHECK-NOT: store
.if !OFFEN
  buffer_store_b128 v[4:7], off, s[4:7], s10 offset:3
.elseif WIDTH == 64
  buffer_store_b64 v[4:5], v1, s[4:7], s10 offen offset:3
.elseif WIDTH == 96
  buffer_store_b96 v[4:6], v1, s[4:7], s10 offen offset:3
.else
  buffer_store_b128 v[4:7], v1, s[4:7], s10 offen offset:3
.endif
; CHECK: ret void
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel buffer_constant_bounds
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 11
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: buffer_constant_bounds
    .symbol: buffer_constant_bounds.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .sgpr_count: 11
    .vgpr_count: 8
    .wavefront_size: 32
...
.end_amdgpu_metadata
