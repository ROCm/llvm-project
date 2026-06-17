; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vcc_hi_vopd_kernel | %FileCheck %s
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=exec_hi_vopd_cond_kernel | %FileCheck %s --check-prefix=EXECHI
;
; On a wave32 source vcc_hi / exec_hi are free scratch scalars; a VOPD
; component that names either as a source or as the cndmask condition must route
; to its own scratch slot, not the real VCC / EXEC. See
; ParsedReg::VCC_HI_SCRATCH.

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  vcc_hi_vopd_kernel
        .p2align        8
        .type   vcc_hi_vopd_kernel,@function
vcc_hi_vopd_kernel:
        s_mov_b32 vcc_hi, 42
        v_cmp_lt_i32 vcc_lo, v0, v1
; The dual cndmask still consumes the real VCC:
; CHECK: %vcmp = icmp slt
; CHECK: %vopd_cndmask = select i1 %vcmp
; The dual mov reads the vcc_hi scratch (constant 42), independent of the VCC:
; CHECK: store i32 42, ptr addrspace(3)
        v_dual_cndmask_b32 v5, v0, v1 :: v_dual_mov_b32 v8, vcc_hi
        ds_store_b32 v6, v5
        ds_store_b32 v7, v8
        s_endpgm

        .globl  exec_hi_vopd_cond_kernel
        .p2align        8
        .type   exec_hi_vopd_cond_kernel,@function
exec_hi_vopd_cond_kernel:
        s_mov_b32 exec_hi, s4
        v_cmp_lt_i32 vcc_lo, v0, v1
; The dual cndmask condition is the per-lane bit of the exec_hi scratch, not the
; real VCC compare %vcmp:
; EXECHI: %vcmp = icmp slt
; EXECHI: %[[LANEBIT:wn_mask_lane_i1[0-9]*]] = icmp ne i64 %{{.*}}, 0
; EXECHI: %vopd_cndmask = select i1 %[[LANEBIT]]
; EXECHI-NOT: %vopd_cndmask = select i1 %vcmp
        v_dual_cndmask_b32 v5, v0, v1, exec_hi :: v_dual_mov_b32 v8, v2
        ds_store_b32 v6, v5
        ds_store_b32 v7, v8
        s_endpgm

        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel vcc_hi_vopd_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_next_free_vgpr 9
                .amdhsa_next_free_sgpr 8
                .amdhsa_wavefront_size32 1
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel
        .amdhsa_kernel exec_hi_vopd_cond_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_next_free_vgpr 9
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
    .name:           vcc_hi_vopd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         vcc_hi_vopd_kernel.kd
    .vgpr_count:     9
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           exec_hi_vopd_cond_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         exec_hi_vopd_cond_kernel.kd
    .vgpr_count:     9
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
        .end_amdgpu_metadata
