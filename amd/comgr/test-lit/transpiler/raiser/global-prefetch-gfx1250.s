; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CASE=0 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch,prefetch_data > %t.ll
; RUN: %opt -passes='default<O2>' -verify-each -S %t.ll -o %t.opt.ll
; RUN: %FileCheck %s --implicit-check-not="= load " \
; RUN:   --implicit-check-not="store " --implicit-check-not="@llvm.prefetch" < %t.opt.ll
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.opt.ll -o %t.target.o
; RUN: %ld.lld -shared %t.target.o -o %t.target.hsaco

; RUN: not %transpile_cli %t.hsaco --isa=gfx1200 --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch 2>&1 | %FileCheck %s --check-prefix=GPU
; GPU: cannot decode instruction at .text offset 0x0

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CASE=1 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch 2>&1 | %FileCheck %s --check-prefix=REFUSE
; REFUSE: only RT prefetch with SE, DEV or SYS scope and no
; REFUSE-SAME: extra cache-policy bits is modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CASE=2 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch 2>&1 | %FileCheck %s --check-prefix=REFUSE

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CASE=3 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch 2>&1 | %FileCheck %s --check-prefix=REFUSE

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CASE=4 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch 2>&1 | %FileCheck %s --check-prefix=REFUSE

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CASE=5 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch 2>&1 | %FileCheck %s --check-prefix=REFUSE

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CASE=6 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=prefetch 2>&1 | %FileCheck %s --check-prefix=REFUSE

.amdhsa_code_object_version 6
.text
.globl prefetch
.p2align 8
.type prefetch,@function
; CHECK-LABEL: define amdgpu_kernel void @prefetch(
; CHECK: ret void
prefetch:
.if CASE == 0
  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
  global_prefetch_b8 v0, s[0:1] offset:-1 scope:SCOPE_SYS
  s_and_b32 exec_lo, exec_lo, 0x55555555
  global_prefetch_b8 v[2:3], off offset:-8388608 scope:SCOPE_DEV
  s_mov_b32 exec_lo, 0
  global_prefetch_b8 v[2:3], off offset:8388607 scope:SCOPE_SYS
.elseif CASE == 1
  global_prefetch_b8 v0, s[0:1]
.elseif CASE == 2
  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE th:TH_LOAD_NT
.elseif CASE == 3
  global_prefetch_b8 v[2:3], off scope:SCOPE_DEV th:TH_LOAD_LU
.elseif CASE == 4
  global_prefetch_b8 v[2:3], off scope:SCOPE_SYS nv
.elseif CASE == 5
  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE scale_offset
.elseif CASE == 6
  global_prefetch_b8 v[2:3], off scope:SCOPE_DEV th:TH_LOAD_HT
.endif
  s_endpgm

.globl prefetch_data
.p2align 8
.type prefetch_data,@function
; CHECK-LABEL: define amdgpu_kernel void @prefetch_data(
prefetch_data:
; CHECK: {{%.+}} = load i64, ptr addrspace(1) {{%.+}}
  s_load_b64 s[2:3], s[0:1], 0
; CHECK: fence syncscope("agent") seq_cst
  s_wait_kmcnt 0
  s_and_b32 exec_lo, exec_lo, 0x55555555
  v_lshlrev_b32 v0, 2, v0
; CHECK: [[VALUE:%.+]] = load volatile i32, ptr addrspace(1) [[ADDRESS:%.+]],
  global_load_b32 v1, v0, s[2:3] scope:SCOPE_DEV
  global_prefetch_b8 v0, s[2:3] offset:1 scope:SCOPE_SE
  s_wait_xcnt 0
  s_wait_loadcnt 0
  v_mov_b32 v2, v0
  v_mov_b32 v0, -1
  s_mov_b64 s[4:5], -1
  global_prefetch_b8 v0, s[4:5] scope:SCOPE_SYS
  s_wait_xcnt 0
  v_mov_b32 v0, v2
; CHECK: [[SUM:%.+]] = add i32 [[VALUE]], 17
  v_add_nc_u32 v1, 17, v1
; CHECK: fence syncscope("agent") seq_cst
; CHECK: store volatile i32 [[SUM]], ptr addrspace(1) [[ADDRESS]],
  global_store_b32 v0, v1, s[2:3] scope:SCOPE_DEV
; CHECK: ret void
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel prefetch
  .amdhsa_kernarg_size 0
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.p2align 6
.amdhsa_kernel prefetch_data
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 6
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: prefetch
    .symbol: prefetch.kd
    .args: []
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .wavefront_size: 32
    .sgpr_count: 2
    .vgpr_count: 4
  - .name: prefetch_data
    .symbol: prefetch_data.kd
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .wavefront_size: 32
    .sgpr_count: 6
    .vgpr_count: 3
...
.end_amdgpu_metadata
