; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vcc_hi_carryout_kernel 2>&1 | %FileCheck %s
;
; A V_CMP may name vcc_hi / exec_hi as a destination, but the wave32 mask/carry
; rules forbid them as a carry-out destination. Such an encoding must be refused
; loudly, not folded into the real VCC.

; CHECK: kernel 'vcc_hi_carryout_kernel'
; CHECK-SAME: vcc_hi / exec_hi scratch as a carry-out destination is forbidden

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  vcc_hi_carryout_kernel
        .p2align        8
        .type   vcc_hi_carryout_kernel,@function
vcc_hi_carryout_kernel:
        v_add_co_u32_e64 v5, vcc_hi, v0, v1
        ds_store_b32 v6, v5
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel vcc_hi_carryout_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_next_free_vgpr 7
                .amdhsa_next_free_sgpr 8
                .amdhsa_wavefront_size32 1
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel
        .text
        .p2alignl 7, 3214868480
        .fill 96, 4, 3214868480
        .amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vcc_hi_carryout_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         vcc_hi_carryout_kernel.kd
    .vgpr_count:     7
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
        .end_amdgpu_metadata
