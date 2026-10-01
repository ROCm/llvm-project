; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch 2>&1 | %FileCheck %s
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx90a --emit-ir=dispatch --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=PAIR
; PAIR: unsupported-wave-projection
; PAIR-SAME: wave-size changes are supported only from gfx1250 to gfx942
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text

.globl matrix
.p2align 8
.type matrix,@function
matrix:
; CHECK: unsupported-wave-projection: v_wmma_f32_16x16x32_f16
; CHECK-SAME: in kernel 'matrix'
; CHECK-SAME: replicated dispatch does not support matrix fragments
  s_bfe_u32 s4, ttmp8, 0x50019
  s_cmp_eq_u32 s4, 0
  s_cbranch_scc1 .Lmatrix_exit
  v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 0
.Lmatrix_exit:
  s_endpgm

.globl hardware
.p2align 8
.type hardware,@function
hardware:
; CHECK: unsupported-wave-projection: s_setreg_b32
; CHECK-SAME: in kernel 'hardware'
; CHECK-SAME: replicated dispatch does not support per-wave hardware effects
  s_bfe_u32 s4, ttmp8, 0x50019
  s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 4, 2), s4
  s_endpgm

.globl message
.p2align 8
.type message,@function
message:
; CHECK: unsupported-wave-projection: s_sendmsg
; CHECK-SAME: in kernel 'message'
; CHECK-SAME: replicated dispatch does not support per-wave hardware effects
  s_bfe_u32 s4, ttmp8, 0x50019
  s_cmp_eq_u32 s4, 0
  s_cbranch_scc1 .Lmessage_exit
  s_sendmsg sendmsg(MSG_INTERRUPT)
.Lmessage_exit:
  s_endpgm

.globl dispatch
.p2align 8
.type dispatch,@function
dispatch:
; CHECK: unsupported-wave-projection
; CHECK-SAME: in kernel 'dispatch'
; CHECK-SAME: replicated dispatch cannot expose target dispatch or queue state
  s_mov_b32 exec_lo, -1
  s_endpgm

.globl queue
.p2align 8
.type queue,@function
queue:
; CHECK: unsupported-wave-projection
; CHECK-SAME: in kernel 'queue'
; CHECK-SAME: replicated dispatch cannot expose target dispatch or queue state
  s_mov_b32 exec_lo, -1
  s_endpgm

.globl dispatch_id
.p2align 8
.type dispatch_id,@function
dispatch_id:
; CHECK: unsupported-wave-projection
; CHECK-SAME: in kernel 'dispatch_id'
; CHECK-SAME: replicated dispatch cannot expose target dispatch or queue state
  s_mov_b32 exec_lo, -1
  s_endpgm

.globl hidden
.p2align 8
.type hidden,@function
hidden:
; CHECK: unsupported-wave-projection
; CHECK-SAME: in kernel 'hidden'
; CHECK-SAME: replicated dispatch cannot reproduce hidden argument kind 'hidden_queue_ptr'
  s_mov_b32 exec_lo, -1
  s_endpgm

.globl small
.p2align 8
.type small,@function
small:
; CHECK: unsupported-launch
; CHECK-SAME: in kernel 'small'
; CHECK-SAME: replicated dispatch requires room for a whole source wave
  s_mov_b32 exec_lo, -1
  s_endpgm

.globl unsupported
.p2align 8
.type unsupported,@function
unsupported:
; CHECK: unsupported-instruction-form: v_rcp_iflag_f32
; CHECK-SAME: in kernel 'unsupported'
  s_mov_b32 exec_lo, -1
  v_rcp_iflag_f32 v1, v0
  s_endpgm

.globl synchronize
.p2align 8
.type synchronize,@function
synchronize:
; CHECK: unsupported-instruction-form: s_barrier_signal
; CHECK-SAME: in kernel 'synchronize'
  s_mov_b32 exec_lo, -1
  s_barrier_signal -1
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel matrix
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel hardware
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel message
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel dispatch
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel queue
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel dispatch_id
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel hidden
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel small
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel unsupported
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel synchronize
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: matrix
    .symbol: matrix.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: hardware
    .symbol: hardware.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: message
    .symbol: message.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: dispatch
    .symbol: dispatch.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: queue
    .symbol: queue.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: dispatch_id
    .symbol: dispatch_id.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: hidden
    .symbol: hidden.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 8
        .size: 8
        .value_kind: hidden_queue_ptr
  - .name: small
    .symbol: small.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 16
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: unsupported
    .symbol: unsupported.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: synchronize
    .symbol: synchronize.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
...
.end_amdgpu_metadata
