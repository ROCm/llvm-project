; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --dump-decoded=global_subdword | %FileCheck \
; RUN:   %s --check-prefix=DECODE
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_subdword \
; RUN:   | %FileCheck %s --check-prefix=IR

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --dump-decoded=global_subdword | %FileCheck \
; RUN:   %s --check-prefix=DECODE
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_subdword \
; RUN:   | %FileCheck %s --check-prefix=IR

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --dump-decoded=global_subdword | %FileCheck \
; RUN:   %s --check-prefix=DECODE
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_subdword \
; RUN:   | %FileCheck %s --check-prefix=IR

.amdhsa_code_object_version 6
.text
.globl global_subdword
.p2align 8
.type global_subdword,@function
; IR-LABEL: define amdgpu_kernel void @global_subdword(
global_subdword:
v_mov_b32 v4, 0x1234
s_mov_b32 exec_lo, 0x55555555

; DECODE: GLOBAL_LOAD_U8 global_load_u8
; IR: [[FROZEN:%.+]] = freeze i64 {{%.+}}
; IR-NEXT: [[PTR:%.+]] = inttoptr i64 [[FROZEN]] to ptr addrspace(1)
; IR-NEXT: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) [[PTR]], i64 1
; IR: br i1 {{%.+}}, label %[[DO:.+]], label %[[SKIP:.+]]
; IR: [[DO]]:
; IR-NEXT: [[VALUE:%.+]] = load i8, ptr addrspace(1) [[OFFSET]], align 1
; IR-NEXT: [[EXT:%.+]] = zext i8 [[VALUE]] to i32
; IR-NEXT: br label %[[SKIP]]
; IR: [[SKIP]]:
; IR-NEXT: [[PREVIOUS0:%.+]] = phi i32 [ [[EXT]], %[[DO]] ], [ 4660, {{%.+}} ]
global_load_u8 v4, v[2:3], off offset:1

; DECODE: GLOBAL_LOAD_I8 global_load_i8
; IR: [[VALUE:%.+]] = load i8, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = sext i8 [[VALUE]] to i32
; IR: [[PREVIOUS1:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS0]], {{%.+}} ]
global_load_i8 v4, v[2:3], off offset:-1

; DECODE: GLOBAL_LOAD_U16 global_load_u16
; IR: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) {{%.+}}, i64 1
; IR: [[VALUE:%.+]] = load i16, ptr addrspace(1) [[OFFSET]], align 1
; IR-NEXT: [[EXT:%.+]] = zext i16 [[VALUE]] to i32
; IR: [[PREVIOUS2:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS1]], {{%.+}} ]
global_load_u16 v4, v[2:3], off offset:1

; DECODE: GLOBAL_LOAD_I16 global_load_i16
; IR: [[VALUE:%.+]] = load i16, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = sext i16 [[VALUE]] to i32
; IR: [[PREVIOUS3:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS2]], {{%.+}} ]
global_load_i16 v4, v[2:3], off offset:-1

; DECODE: GLOBAL_STORE_B8 global_store_b8
; IR: [[DATA:%.+]] = trunc i32 [[PREVIOUS3]] to i8
; IR: [[FROZEN:%.+]] = freeze i64 {{%.+}}
; IR-NEXT: [[PTR:%.+]] = inttoptr i64 [[FROZEN]] to ptr addrspace(1)
; IR-NEXT: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) [[PTR]], i64 1
; IR: br i1 {{%.+}}, label %[[DO:.+]], label %[[SKIP:.+]]
; IR: [[DO]]:
; IR-NEXT: store i8 [[DATA]], ptr addrspace(1) [[OFFSET]], align 1
; IR-NEXT: br label %[[SKIP]]
; IR: [[SKIP]]:
global_store_b8 v[2:3], v4, off offset:1

; DECODE: GLOBAL_STORE_B16 global_store_b16
; IR: [[DATA:%.+]] = trunc i32 [[PREVIOUS3]] to i16
; IR: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) {{%.+}}, i64 -1
; IR: store i16 [[DATA]], ptr addrspace(1) [[OFFSET]], align 1
global_store_b16 v[2:3], v4, off offset:-1

; DECODE: GLOBAL_STORE_D16_HI_B8 global_store_d16_hi_b8
; IR: [[HIGH:%.+]] = lshr i32 [[PREVIOUS3]], 16
; IR-NEXT: [[DATA:%.+]] = trunc i32 [[HIGH]] to i8
; IR: store i8 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_d16_hi_b8 v[2:3], v4, off offset:1

; DECODE: GLOBAL_STORE_D16_HI_B16 global_store_d16_hi_b16
; IR: [[HIGH:%.+]] = lshr i32 [[PREVIOUS3]], 16
; IR-NEXT: [[DATA:%.+]] = trunc i32 [[HIGH]] to i16
; IR: store i16 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_d16_hi_b16 v[2:3], v4, off offset:-1

; DECODE: GLOBAL_LOAD_U8 global_load_u8
; IR: [[VALUE:%.+]] = load i8, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = zext i8 [[VALUE]] to i32
; IR: [[PREVIOUS4:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS3]], {{%.+}} ]
global_load_u8 v4, v0, s[0:1] offset:1

; DECODE: GLOBAL_LOAD_I8 global_load_i8
; IR: [[VALUE:%.+]] = load i8, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = sext i8 [[VALUE]] to i32
; IR: [[PREVIOUS5:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS4]], {{%.+}} ]
global_load_i8 v4, v0, s[0:1] offset:-1

; DECODE: GLOBAL_LOAD_U16 global_load_u16
; IR: [[VALUE:%.+]] = load i16, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = zext i16 [[VALUE]] to i32
; IR: [[PREVIOUS6:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS5]], {{%.+}} ]
global_load_u16 v4, v0, s[0:1] offset:1

; DECODE: GLOBAL_LOAD_I16 global_load_i16
; IR: [[VALUE:%.+]] = load i16, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = sext i16 [[VALUE]] to i32
; IR: [[PREVIOUS7:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS6]], {{%.+}} ]
global_load_i16 v4, v0, s[0:1] offset:-1

; DECODE: GLOBAL_STORE_B8 global_store_b8
; IR: [[DATA:%.+]] = trunc i32 [[PREVIOUS7]] to i8
; IR: store i8 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_b8 v0, v4, s[0:1] offset:1

; DECODE: GLOBAL_STORE_B16 global_store_b16
; IR: [[DATA:%.+]] = trunc i32 [[PREVIOUS7]] to i16
; IR: store i16 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_b16 v0, v4, s[0:1] offset:-1

; DECODE: GLOBAL_STORE_D16_HI_B8 global_store_d16_hi_b8
; IR: [[HIGH:%.+]] = lshr i32 [[PREVIOUS7]], 16
; IR-NEXT: [[DATA:%.+]] = trunc i32 [[HIGH]] to i8
; IR: store i8 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_d16_hi_b8 v0, v4, s[0:1] offset:1

; DECODE: GLOBAL_STORE_D16_HI_B16 global_store_d16_hi_b16
; IR: [[HIGH:%.+]] = lshr i32 [[PREVIOUS7]], 16
; IR-NEXT: [[DATA:%.+]] = trunc i32 [[HIGH]] to i16
; IR: store i16 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_d16_hi_b16 v0, v4, s[0:1] offset:-1

; IR: ret void
s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel global_subdword
  .amdhsa_kernarg_size 0
  .amdhsa_wavefront_size32 1
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
    .name: global_subdword
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: global_subdword.kd
    .vgpr_count: 8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
