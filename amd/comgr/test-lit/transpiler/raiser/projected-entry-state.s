; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=unused,overwritten,dispatch_sizes --allow-replicated-dispatch --specialize-workgroup=32,2,1 > %t.ll
; RUN: %FileCheck %s --check-prefix=IR < %t.ll
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=dispatch_sizes --allow-replicated-dispatch > %t.dynamic.ll
; RUN: %FileCheck %s --check-prefix=DYNAMIC < %t.dynamic.ll
; DYNAMIC-LABEL: define amdgpu_kernel void @dispatch_sizes(
; DYNAMIC: call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; DYNAMIC: load i16
; DYNAMIC: udiv i32 {{.+}}, 2
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=partial,path,queue_use,other_field --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym OVERWRITE_HIGH=1 -filetype=obj %s -o %t.high.o
; RUN: %ld.lld -shared %t.high.o -o %t.high.hsaco
; RUN: not %transpile_cli %t.high.hsaco --target-isa=gfx942 --emit-ir=partial --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=PARTIAL
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=outlined --allow-replicated-dispatch | %FileCheck %s --check-prefix=OUTLINED
; OUTLINED-LABEL: define amdgpu_kernel void @outlined(
; OUTLINED-NOT: call {{.+}} @llvm.amdgcn.queue.ptr
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym HELPER_USES_ENTRY=1 -filetype=obj %s -o %t.helper.o
; RUN: %ld.lld -shared %t.helper.o -o %t.helper.hsaco
; RUN: not %transpile_cli %t.helper.hsaco --target-isa=gfx942 --emit-ir=outlined --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=HELPER
; HELPER: unsupported-entry-sgpr-source in kernel 'outlined'
; HELPER-SAME: cannot reproduce consumed entry state 'llvm.amdgcn.queue.ptr'
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text

.globl unused
.p2align 8
.type unused,@function
; IR-LABEL: define amdgpu_kernel void @unused(
; IR-NOT: call {{.+}} @llvm.amdgcn.dispatch.ptr
; IR-NOT: call {{.+}} @llvm.amdgcn.queue.ptr
unused:
  s_mov_b32 exec_lo, -1
  s_mov_b32 s0, 0
  s_mov_b32 s8, s1
  s_mov_b32 s9, s3
  s_endpgm

.globl overwritten
.p2align 8
.type overwritten,@function
; IR-LABEL: define amdgpu_kernel void @overwritten(
; IR-NOT: call {{.+}} @llvm.amdgcn.dispatch.ptr
; IR-NOT: call {{.+}} @llvm.amdgcn.queue.ptr
overwritten:
  s_mov_b32 exec_lo, -1
  s_mov_b64 s[0:1], 42
  s_load_b64 s[6:7], s[4:5], 0
  s_wait_kmcnt 0
  v_mov_b32 v1, s0
  global_store_b32 v0, v1, s[6:7]
  s_endpgm

.globl dispatch_sizes
.p2align 8
.type dispatch_sizes,@function
; IR-LABEL: define amdgpu_kernel void @dispatch_sizes(
; IR-NOT: call {{.+}} @llvm.amdgcn.dispatch.ptr
; IR-NOT: call {{.+}} @llvm.amdgcn.queue.ptr
; The stored packet words are X=32, Y=2, Z=1, reserved=0.
; IR: %[[SIZES:.+]] = phi <2 x i32> {{.*}}bitcast (<1 x i64> splat (i64 4295098400) to <2 x i32>)
; IR-NEXT: %[[LOW:.+]] = extractelement <2 x i32> %[[SIZES]], i64 0
; IR-NEXT: %[[XY:.+]] = call i32 @llvm.amdgcn.readlane.i32(i32 %[[LOW]], i32 0)
; IR-NEXT: %[[HIGH:.+]] = extractelement <2 x i32> %[[SIZES]], i64 1
; IR-NEXT: %[[Z:.+]] = call i32 @llvm.amdgcn.readlane.i32(i32 %[[HIGH]], i32 0)
; IR: %[[XY64:.+]] = zext i32 %[[XY]] to i64
; IR-NEXT: %[[Z64:.+]] = zext i32 %[[Z]] to i64
; IR-NEXT: %[[SHIFT:.+]] = shl nuw i64 %[[Z64]], 32
; IR-NEXT: %[[PACKED:.+]] = or disjoint i64 %[[SHIFT]], %[[XY64]]
; IR: store i64 %[[PACKED]], ptr addrspace(1) {{%.+}}, align 4
dispatch_sizes:
  s_mov_b32 exec_lo, -1
  s_load_b64 s[8:9], s[0:1], 4
  s_load_b64 s[6:7], s[4:5], 0
  s_wait_kmcnt 0
  v_mov_b32 v2, s8
  v_mov_b32 v3, s9
  global_store_b64 v0, v[2:3], s[6:7]
  s_endpgm

.globl partial
.p2align 8
.type partial,@function
; PARTIAL: unsupported-entry-sgpr-source in kernel 'partial'
; PARTIAL-SAME: cannot reproduce consumed entry state
; REFUSE: unsupported-entry-sgpr-source in kernel 'partial'
; REFUSE-SAME: cannot reproduce consumed entry state
partial:
  s_mov_b32 exec_lo, -1
.ifdef OVERWRITE_HIGH
  s_mov_b32 s1, 0
  s_cmp_eq_u32 s0, 0
.else
  s_mov_b32 s0, 0
  s_cmp_eq_u32 s1, 0
.endif
  s_cbranch_scc1 .Lpartial_exit
  s_mov_b32 exec_lo, 0
.Lpartial_exit:
  s_endpgm

.globl path
.p2align 8
.type path,@function
; REFUSE: unsupported-entry-sgpr-source in kernel 'path'
; REFUSE-SAME: cannot reproduce consumed entry state
path:
  s_mov_b32 exec_lo, -1
  s_cmp_eq_u32 ttmp9, 0
  s_cbranch_scc1 .Lpath_use
  s_mov_b64 s[0:1], 0
.Lpath_use:
  s_cmp_eq_u32 s1, 0
  s_cbranch_scc1 .Lpath_exit
  s_mov_b32 exec_lo, 0
.Lpath_exit:
  s_endpgm

.globl queue_use
.p2align 8
.type queue_use,@function
; REFUSE: unsupported-entry-sgpr-source in kernel 'queue_use'
; REFUSE-SAME: cannot reproduce consumed entry state
queue_use:
  s_mov_b32 exec_lo, -1
  s_load_b32 s8, s[2:3], 0
  s_cmp_eq_u32 s8, 0
  s_cbranch_scc1 .Lentry_load_use_1
  s_mov_b32 exec_lo, 0
.Lentry_load_use_1:
  s_endpgm

.globl other_field
.p2align 8
.type other_field,@function
; REFUSE: unsupported-entry-sgpr-source in kernel 'other_field'
; REFUSE-SAME: cannot reproduce consumed entry state
other_field:
  s_mov_b32 exec_lo, -1
  s_load_b32 s8, s[0:1], 12
  s_cmp_eq_u32 s8, 0
  s_cbranch_scc1 .Lentry_load_use_2
  s_mov_b32 exec_lo, 0
.Lentry_load_use_2:
  s_endpgm

.globl outlined
.p2align 8
.type outlined,@function
outlined:
  s_mov_b32 exec_lo, -1
  s_get_pc_i64 s[10:11]
  s_add_u32 s10, s10, entry_helper@rel32@lo+4
  s_addc_u32 s11, s11, entry_helper@rel32@hi+12
  s_swap_pc_i64 s[12:13], s[10:11]
  s_cmp_eq_u32 s8, 0
  s_cbranch_scc1 .Loutlined_exit
  s_mov_b32 exec_lo, 0
.Loutlined_exit:
  s_endpgm
.size outlined, .-outlined

.globl entry_helper
.hidden entry_helper
.p2align 8
.type entry_helper,@function
entry_helper:
.ifdef HELPER_USES_ENTRY
  s_mov_b32 s8, s3
.else
  s_mov_b32 s8, 42
.endif
  s_set_pc_i64 s[12:13]
.size entry_helper, .-entry_helper

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel unused
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdhsa_kernel overwritten
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdhsa_kernel dispatch_sizes
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdhsa_kernel partial
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdhsa_kernel path
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdhsa_kernel queue_use
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdhsa_kernel other_field
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdhsa_kernel outlined
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: unused
    .symbol: unused.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: overwritten
    .symbol: overwritten.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: dispatch_sizes
    .symbol: dispatch_sizes.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: partial
    .symbol: partial.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: path
    .symbol: path.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: queue_use
    .symbol: queue_use.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: other_field
    .symbol: other_field.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
  - .name: outlined
    .symbol: outlined.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 4
    .wavefront_size: 32
...
.end_amdgpu_metadata
