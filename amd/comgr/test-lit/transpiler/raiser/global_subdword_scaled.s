; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --dump-decoded=global_subdword_scaled | \
; RUN:   %FileCheck %s --check-prefix=DECODE
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=global_subdword_scaled | %FileCheck %s --check-prefix=IR

; RUN: not %transpile_cli %t.hsaco --isa=gfx1200 --target-isa=gfx1200 \
; RUN:   --emit-ir=global_subdword_scaled 2>&1 | %FileCheck %s \
; RUN:   --check-prefix=GPU
; GPU: scale_offset is not supported on this GPU

.amdhsa_code_object_version 6
.text
.globl global_subdword_scaled
.p2align 8
.type global_subdword_scaled,@function
; IR-LABEL: define amdgpu_kernel void @global_subdword_scaled(
global_subdword_scaled:
v_mov_b32 v4, 0x1234
s_and_b32 exec_lo, exec_lo, 0x55555555

; DECODE: GLOBAL_LOAD_U8 global_load_u8
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: mul i64 [[LANE]], 1
; IR: [[VALUE:%.+]] = load i8, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = zext i8 [[VALUE]] to i32
; IR: [[PREVIOUS0:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ 4660, {{%.+}} ]
global_load_u8 v4, v0, s[0:1] offset:1 scale_offset

; DECODE: GLOBAL_LOAD_I8 global_load_i8
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: mul i64 [[LANE]], 1
; IR: [[VALUE:%.+]] = load i8, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = sext i8 [[VALUE]] to i32
; IR: [[PREVIOUS1:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS0]], {{%.+}} ]
global_load_i8 v4, v0, s[0:1] offset:-1 scale_offset

; DECODE: GLOBAL_LOAD_U16 global_load_u16
; IR: [[BASE:%.+]] = or i64 {{%.+}}, {{%.+}}
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: [[SCALE:%.+]] = mul i64 [[LANE]], 2
; IR-NEXT: [[ADDR:%.+]] = add i64 [[BASE]], [[SCALE]]
; IR-NEXT: [[FROZEN:%.+]] = freeze i64 [[ADDR]]
; IR-NEXT: [[PTR:%.+]] = inttoptr i64 [[FROZEN]] to ptr addrspace(1)
; IR-NEXT: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) [[PTR]], i64 1
; IR: [[VALUE:%.+]] = load i16, ptr addrspace(1) [[OFFSET]], align 1
; IR-NEXT: [[EXT:%.+]] = zext i16 [[VALUE]] to i32
; IR: [[PREVIOUS2:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS1]], {{%.+}} ]
global_load_u16 v4, v0, s[0:1] offset:1 scale_offset

; DECODE: GLOBAL_LOAD_I16 global_load_i16
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: mul i64 [[LANE]], 2
; IR: [[VALUE:%.+]] = load i16, ptr addrspace(1) {{%.+}}, align 1
; IR-NEXT: [[EXT:%.+]] = sext i16 [[VALUE]] to i32
; IR: [[PREVIOUS3:%.+]] = phi i32 [ [[EXT]], {{%.+}} ], [ [[PREVIOUS2]], {{%.+}} ]
global_load_i16 v4, v0, s[0:1] offset:-1 scale_offset

; DECODE: GLOBAL_STORE_B8 global_store_b8
; IR: [[DATA:%.+]] = trunc i32 [[PREVIOUS3]] to i8
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: mul i64 [[LANE]], 1
; IR: store i8 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_b8 v0, v4, s[0:1] offset:1 scale_offset

; DECODE: GLOBAL_STORE_B16 global_store_b16
; IR: [[DATA:%.+]] = trunc i32 [[PREVIOUS3]] to i16
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: mul i64 [[LANE]], 2
; IR: [[OFFSET:%.+]] = getelementptr i8, ptr addrspace(1) {{%.+}}, i64 -1
; IR: store i16 [[DATA]], ptr addrspace(1) [[OFFSET]], align 1
global_store_b16 v0, v4, s[0:1] offset:-1 scale_offset

; DECODE: GLOBAL_STORE_D16_HI_B8 global_store_d16_hi_b8
; IR: [[HIGH:%.+]] = lshr i32 [[PREVIOUS3]], 16
; IR-NEXT: [[DATA:%.+]] = trunc i32 [[HIGH]] to i8
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: mul i64 [[LANE]], 1
; IR: store i8 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_d16_hi_b8 v0, v4, s[0:1] offset:1 scale_offset

; DECODE: GLOBAL_STORE_D16_HI_B16 global_store_d16_hi_b16
; IR: [[HIGH:%.+]] = lshr i32 [[PREVIOUS3]], 16
; IR-NEXT: [[DATA:%.+]] = trunc i32 [[HIGH]] to i16
; IR: [[LANE:%.+]] = sext i32 {{.+}} to i64
; IR-NEXT: mul i64 [[LANE]], 2
; IR: store i16 [[DATA]], ptr addrspace(1) {{%.+}}, align 1
global_store_d16_hi_b16 v0, v4, s[0:1] offset:-1 scale_offset

; IR: ret void
s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel global_subdword_scaled
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
    .name: global_subdword_scaled
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: global_subdword_scaled.kd
    .vgpr_count: 8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
