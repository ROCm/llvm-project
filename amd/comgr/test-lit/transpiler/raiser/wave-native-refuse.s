; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=expand_exec 2>&1 | %FileCheck %s --check-prefix=EXPAND-EXEC
; EXPAND-EXEC: unproven-exec-containment:
; EXPAND-EXEC-SAME: cannot prove that EXEC preserves the kernel entry mask
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=mask_branch 2>&1 | %FileCheck %s --check-prefix=MASK-BRANCH
; MASK-BRANCH: non-uniform-scalar-state:
; MASK-BRANCH-SAME: requires scalar control flow uniform across the target wave
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=yz_branch 2>&1 | %FileCheck %s --check-prefix=YZ-BRANCH
; YZ-BRANCH: non-uniform-scalar-state:
; YZ-BRANCH-SAME: requires scalar control flow uniform across the target wave
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=scalar_address 2>&1 | %FileCheck %s --check-prefix=SCALAR-ADDRESS
; SCALAR-ADDRESS: non-uniform-scalar-state:
; SCALAR-ADDRESS-SAME: requires uniform scalar memory addresses
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=wave_message 2>&1 | %FileCheck %s --check-prefix=WAVE-MESSAGE
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=wave_message --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=WAVE-MESSAGE
; WAVE-MESSAGE: unsupported-wave-projection
; WAVE-MESSAGE-SAME: does not support per-wave hardware side effects
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=mask_loop 2>&1 | %FileCheck %s --check-prefix=MASK-LOOP
; MASK-LOOP: unproven-exec-containment:
; MASK-LOOP-SAME: cannot prove that EXEC preserves the kernel entry mask
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx90a --emit-ir=expand_exec 2>&1 | %FileCheck %s --check-prefix=DIRECTION
; DIRECTION: unsupported-wave-projection
; DIRECTION-SAME: wave-size changes are supported only from gfx1250 to gfx942

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=masked_matrix 2>&1 | %FileCheck %s --check-prefix=MATRIX
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=masked_matrix --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=MATRIX
; MATRIX: unsupported-wave-projection: v_wmma_f32_16x16x32_f16
; MATRIX-SAME: requires the entry EXEC mask at WMMA instructions

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=WAVE32=0 -filetype=obj %s -o %t.wave64.o
; RUN: %ld.lld -shared %t.wave64.o -o %t.wave64.hsaco
; RUN: not %transpile_cli %t.wave64.hsaco --target-isa=gfx942 --emit-ir=expand_exec 2>&1 | %FileCheck %s --check-prefix=WAVE64
; WAVE64: unsupported-wave-projection
; WAVE64-SAME: requires a wave32 source kernel descriptor

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=hardware_register 2>&1 | %FileCheck %s --check-prefix=HWREG
; HWREG: non-uniform-scalar-state: s_setreg_b32
; HWREG-SAME: requires uniform hardware register writes

.ifndef WAVE32
.set WAVE32, 1
.endif

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text

.globl expand_exec
.p2align 8
.type expand_exec,@function
expand_exec:
  s_mov_b32 exec_lo, -1
  s_endpgm

.globl mask_branch
.p2align 8
.type mask_branch,@function
mask_branch:
  v_cmp_eq_u32 vcc_lo, 0, v0
  s_cbranch_vccz .Lexit_mask
  v_mov_b32 v1, 1
.Lexit_mask:
  s_endpgm

.globl yz_branch
.p2align 8
.type yz_branch,@function
yz_branch:
  v_lshrrev_b32 v1, 10, v0
  v_readfirstlane_b32 s4, v1
  s_cmp_eq_u32 s4, 0
  s_cbranch_scc1 .Lexit_yz
  v_mov_b32 v2, 1
.Lexit_yz:
  s_endpgm

.globl scalar_address
.p2align 8
.type scalar_address,@function
scalar_address:
  v_readfirstlane_b32 s4, v0
  s_mov_b32 s5, 0
  s_load_b32 s6, s[4:5], 0
  s_endpgm

.globl hardware_register
.p2align 8
.type hardware_register,@function
hardware_register:
  s_bfe_u32 s4, ttmp8, 0x50019
  s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 4, 2), s4
  s_endpgm

.globl wave_message
.p2align 8
.type wave_message,@function
wave_message:
  s_sendmsg sendmsg(MSG_INTERRUPT)
  s_endpgm

.globl mask_loop
.p2align 8
.type mask_loop,@function
mask_loop:
  s_mov_b32 s4, 4
.Lloop_mask:
  s_and_b32 exec_lo, exec_lo, 0x55555555
  s_sub_u32 s4, s4, 1
  s_cbranch_scc1 .Lloop_mask
  s_endpgm

.globl masked_matrix
.p2align 8
.type masked_matrix,@function
masked_matrix:
  s_mov_b32 exec_lo, 0
  v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 0
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel expand_exec
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 WAVE32
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel mask_branch
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel yz_branch
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel scalar_address
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel hardware_register
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel wave_message
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel mask_loop
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel


.amdhsa_kernel masked_matrix
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: expand_exec
    .symbol: expand_exec.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: mask_branch
    .symbol: mask_branch.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: yz_branch
    .symbol: yz_branch.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: scalar_address
    .symbol: scalar_address.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: hardware_register
    .symbol: hardware_register.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: wave_message
    .symbol: wave_message.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: mask_loop
    .symbol: mask_loop.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .args:
      - .name: output
        .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name: masked_matrix
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .symbol: masked_matrix.kd
    .vgpr_count: 32
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
