; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=+sramecc -defsym=OFFSET=4 -defsym=MARGIN=1 -filetype=obj %s -o %t.inside.o
; RUN: %ld.lld -shared %t.inside.o -o %t.inside.hsaco
; RUN: %transpile_cli %t.inside.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,IN
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=+sramecc -defsym=OFFSET=4 -defsym=MARGIN=0 -filetype=obj %s -o %t.equal.o
; RUN: %ld.lld -shared %t.equal.o -o %t.equal.hsaco
; RUN: %transpile_cli %t.equal.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,OOB
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=+sramecc -defsym=OFFSET=4 -defsym=MARGIN=-1 -filetype=obj %s -o %t.past.o
; RUN: %ld.lld -shared %t.past.o -o %t.past.hsaco
; RUN: %transpile_cli %t.past.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,OOB
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=+sramecc -defsym=OFFSET=3 -defsym=MARGIN=1 -filetype=obj %s -o %t.unaligned.inside.o
; RUN: %ld.lld -shared %t.unaligned.inside.o -o %t.unaligned.inside.hsaco
; RUN: %transpile_cli %t.unaligned.inside.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,IN
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=+sramecc -defsym=OFFSET=3 -defsym=MARGIN=0 -filetype=obj %s -o %t.unaligned.equal.o
; RUN: %ld.lld -shared %t.unaligned.equal.o -o %t.unaligned.equal.hsaco
; RUN: %transpile_cli %t.unaligned.equal.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,OOB
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=+sramecc -defsym=OFFSET=3 -defsym=MARGIN=-1 -filetype=obj %s -o %t.unaligned.past.o
; RUN: %ld.lld -shared %t.unaligned.past.o -o %t.unaligned.past.hsaco
; RUN: %transpile_cli %t.unaligned.past.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=CHECK,OOB

.amdhsa_code_object_version 6
.text
.globl buffer_narrow_bounds
.p2align 8
.type buffer_narrow_bounds,@function
; CHECK-LABEL: define amdgpu_kernel void @buffer_narrow_bounds(
buffer_narrow_bounds:
  s_load_b128 s[8:11], s[0:1], 0
; CHECK: fence
  s_wait_kmcnt 0
  s_mov_b32 s4, s8
  s_and_b32 s5, s9, 0x1ffffff
  s_or_b32 s5, s5, (OFFSET + 1 + MARGIN) << 25
  s_mov_b32 s6, 0
  s_mov_b32 s7, 0
  s_mov_b32 s12, 1
  v_mov_b32 v0, 0
  v_mov_b32 v1, 1
  v_mov_b32 v4, 0x12345678
  v_mov_b32 v5, 0x12345678
  v_mov_b32 v6, 0x12345678
  v_mov_b32 v7, 0x12345678
; OOB-NOT: load
; IN: [[BYTE:%.+]] = load i8, ptr addrspace(1) {{.+}}, align 1
; IN: [[U8:%.+]] = zext i8 [[BYTE]] to i32
  buffer_load_u8 v4, v1, s[4:7], s12 offen offset:OFFSET-2
; IN: [[LOW_BYTE:%.+]] = load i8, ptr addrspace(1) {{.+}}, align 1
; IN: [[LOW_U8:%.+]] = zext i8 [[LOW_BYTE]] to i32
  buffer_load_d16_u8 v5, v1, s[4:7], s12 offen offset:OFFSET-2
; IN: [[HIGH_BYTE:%.+]] = load i8, ptr addrspace(1) {{.+}}, align 1
; IN: [[EXT_BYTE:%.+]] = zext i8 [[HIGH_BYTE]] to i32
; IN: [[HIGH_U8:%.+]] = shl{{.+}}i32 [[EXT_BYTE]], 16
  buffer_load_d16_hi_u8 v6, v1, s[4:7], s12 offen offset:OFFSET-2
  s_wait_loadcnt 0
; OOB-COUNT-3: store i32 0, ptr addrspace(1)
; IN: store i32 [[U8]], ptr addrspace(1)
  global_store_b32 v0, v4, s[10:11]
; IN: store i32 [[LOW_U8]], ptr addrspace(1)
  global_store_b32 v0, v5, s[10:11] offset:4
; IN: store i32 [[HIGH_U8]], ptr addrspace(1)
  global_store_b32 v0, v6, s[10:11] offset:8
; OOB-NOT: store i8
; IN: store i8 120, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b8 v7, v1, s[4:7], s12 offen offset:OFFSET-2
; IN: store i8 52, ptr addrspace(1) {{.+}}, align 1
  buffer_store_d16_hi_b8 v7, v1, s[4:7], s12 offen offset:OFFSET-2
  s_wait_storecnt 0
  s_and_b32 s5, s9, 0x1ffffff
  s_or_b32 s5, s5, (OFFSET + 2 + MARGIN) << 25
; OOB-NOT: load
; IN: [[HALF:%.+]] = load i16, ptr addrspace(1) {{.+}}, align 1
; IN: [[U16:%.+]] = zext i16 [[HALF]] to i32
  buffer_load_u16 v4, v1, s[4:7], s12 offen offset:OFFSET-2
; IN: [[LOW_HALF:%.+]] = load i16, ptr addrspace(1) {{.+}}, align 1
; IN: [[LOW_U16:%.+]] = zext i16 [[LOW_HALF]] to i32
  buffer_load_d16_b16 v5, v1, s[4:7], s12 offen offset:OFFSET-2
; IN: [[HIGH_HALF:%.+]] = load i16, ptr addrspace(1) {{.+}}, align 1
; IN: [[EXT_HALF:%.+]] = zext i16 [[HIGH_HALF]] to i32
; IN: [[HIGH_U16:%.+]] = shl{{.+}}i32 [[EXT_HALF]], 16
  buffer_load_d16_hi_b16 v6, v1, s[4:7], s12 offen offset:OFFSET-2
  s_wait_loadcnt 0
; OOB-COUNT-3: store i32 0, ptr addrspace(1)
; IN: store i32 [[U16]], ptr addrspace(1)
  global_store_b32 v0, v4, s[10:11] offset:12
; IN: store i32 [[LOW_U16]], ptr addrspace(1)
  global_store_b32 v0, v5, s[10:11] offset:16
; IN: store i32 [[HIGH_U16]], ptr addrspace(1)
  global_store_b32 v0, v6, s[10:11] offset:20
; IN: store i16 22136, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b16 v7, v1, s[4:7], s12 offen offset:OFFSET-2
; IN: store i16 4660, ptr addrspace(1) {{.+}}, align 1
  buffer_store_d16_hi_b16 v7, v1, s[4:7], s12 offen offset:OFFSET-2
; OOB-NOT: store
; CHECK: ret void
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel buffer_narrow_bounds
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 13
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: buffer_narrow_bounds
    .symbol: buffer_narrow_bounds.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .sgpr_count: 13
    .vgpr_count: 8
    .wavefront_size: 32
...
.end_amdgpu_metadata
