; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vbuffer_vgpr_msb_roles_kernel \
; RUN:   | %FileCheck %s

; s_set_vgpr_msb selects the high-VGPR bank per operand role for the vbuffer store.

    .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
    .amdhsa_code_object_version 6
    .text
    .globl  vbuffer_vgpr_msb_roles_kernel
    .p2align 8
    .type   vbuffer_vgpr_msb_roles_kernel,@function
vbuffer_vgpr_msb_roles_kernel:
    s_load_b64 s[0:1], s[0:1], 0
    v_mov_b32_e32 v0, 7
    v_mov_b32_e32 v200, 11
    s_set_vgpr_msb 0xc0
    v_mov_b32_e32 v200, 42
    s_set_vgpr_msb 0x3
; CHECK-LABEL: define amdgpu_kernel void @vbuffer_vgpr_msb_roles_kernel(
; CHECK-NOT: add i32 0, 11
; CHECK: [[VOFF:%[0-9]+]] = add i32 0, 42
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i32(i32 %{{[^,]+}}, <4 x i32> %{{[^,]+}}, i32 [[VOFF]], i32 0, i32 0)
    buffer_store_dword v0, v200, s[0:3], null offen
    s_set_vgpr_msb 0
    s_endpgm

    .section .rodata,"a",@progbits
    .p2align 6, 0x0
    .amdhsa_kernel vbuffer_vgpr_msb_roles_kernel
        .amdhsa_kernarg_size 8
        .amdhsa_user_sgpr_count 2
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_next_free_vgpr 1024
        .amdhsa_next_free_sgpr 4
        .amdhsa_wavefront_size32 1
    .end_amdhsa_kernel
    .amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: vbuffer_vgpr_msb_roles_kernel
    .symbol: vbuffer_vgpr_msb_roles_kernel.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 4
    .vgpr_count: 1024
    .max_flat_workgroup_size: 64
    .args:
      - { .name: out, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
    .end_amdgpu_metadata
