; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_rcp_iflag_f32_clamp_refuse_kernel,v_rcp_f32_omod_refuse_kernel,v_exp_f32_clamp_refuse_kernel,v_log_f32_omod_refuse_kernel,v_ldexp_f32_clamp_refuse_kernel,v_sqrt_f32_omod_refuse_kernel,v_rsq_f32_clamp_refuse_kernel,v_floor_f32_omod_refuse_kernel,v_ceil_f32_clamp_refuse_kernel,v_trunc_f32_omod_refuse_kernel,v_rndne_f32_clamp_refuse_kernel,v_fract_f32_omod_refuse_kernel 2>&1 | %FileCheck %s

; Refuse f32 unary ops carrying non-default clamp/omod output modifiers.
; CHECK-DAG: kernel 'v_rcp_iflag_f32_clamp_refuse_kernel'
; CHECK-DAG: V_RCP_IFLAG_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_rcp_f32_omod_refuse_kernel'
; CHECK-DAG: V_RCP_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_exp_f32_clamp_refuse_kernel'
; CHECK-DAG: V_EXP_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_log_f32_omod_refuse_kernel'
; CHECK-DAG: V_LOG_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_ldexp_f32_clamp_refuse_kernel'
; CHECK-DAG: V_LDEXP_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_sqrt_f32_omod_refuse_kernel'
; CHECK-DAG: V_SQRT_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_rsq_f32_clamp_refuse_kernel'
; CHECK-DAG: V_RSQ_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_floor_f32_omod_refuse_kernel'
; CHECK-DAG: V_FLOOR_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_ceil_f32_clamp_refuse_kernel'
; CHECK-DAG: V_CEIL_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_trunc_f32_omod_refuse_kernel'
; CHECK-DAG: V_TRUNC_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_rndne_f32_clamp_refuse_kernel'
; CHECK-DAG: V_RNDNE_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped
; CHECK-DAG: kernel 'v_fract_f32_omod_refuse_kernel'
; CHECK-DAG: V_FRACT_F32 with non-default clamp/omod is not yet lifted
; CHECK-DAG: output modifier semantics must not be silently dropped

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  v_rcp_iflag_f32_clamp_refuse_kernel
        .p2align        8
        .type   v_rcp_iflag_f32_clamp_refuse_kernel,@function
v_rcp_iflag_f32_clamp_refuse_kernel:
        v_rcp_iflag_f32_e64 v0, v0 clamp
        s_endpgm

        .globl  v_rcp_f32_omod_refuse_kernel
        .p2align        8
        .type   v_rcp_f32_omod_refuse_kernel,@function
v_rcp_f32_omod_refuse_kernel:
        v_rcp_f32_e64 v0, v0 mul:2
        s_endpgm

        .globl  v_exp_f32_clamp_refuse_kernel
        .p2align        8
        .type   v_exp_f32_clamp_refuse_kernel,@function
v_exp_f32_clamp_refuse_kernel:
        v_exp_f32_e64 v0, v0 clamp
        s_endpgm

        .globl  v_log_f32_omod_refuse_kernel
        .p2align        8
        .type   v_log_f32_omod_refuse_kernel,@function
v_log_f32_omod_refuse_kernel:
        v_log_f32_e64 v0, v0 mul:2
        s_endpgm

        .globl  v_ldexp_f32_clamp_refuse_kernel
        .p2align        8
        .type   v_ldexp_f32_clamp_refuse_kernel,@function
v_ldexp_f32_clamp_refuse_kernel:
        v_ldexp_f32 v0, v0, v0 clamp
        s_endpgm

        .globl  v_sqrt_f32_omod_refuse_kernel
        .p2align        8
        .type   v_sqrt_f32_omod_refuse_kernel,@function
v_sqrt_f32_omod_refuse_kernel:
        v_sqrt_f32_e64 v0, v0 mul:2
        s_endpgm

        .globl  v_rsq_f32_clamp_refuse_kernel
        .p2align        8
        .type   v_rsq_f32_clamp_refuse_kernel,@function
v_rsq_f32_clamp_refuse_kernel:
        v_rsq_f32_e64 v0, v0 clamp
        s_endpgm

        .globl  v_floor_f32_omod_refuse_kernel
        .p2align        8
        .type   v_floor_f32_omod_refuse_kernel,@function
v_floor_f32_omod_refuse_kernel:
        v_floor_f32_e64 v0, v0 mul:2
        s_endpgm

        .globl  v_ceil_f32_clamp_refuse_kernel
        .p2align        8
        .type   v_ceil_f32_clamp_refuse_kernel,@function
v_ceil_f32_clamp_refuse_kernel:
        v_ceil_f32_e64 v0, v0 clamp
        s_endpgm

        .globl  v_trunc_f32_omod_refuse_kernel
        .p2align        8
        .type   v_trunc_f32_omod_refuse_kernel,@function
v_trunc_f32_omod_refuse_kernel:
        v_trunc_f32_e64 v0, v0 mul:2
        s_endpgm

        .globl  v_rndne_f32_clamp_refuse_kernel
        .p2align        8
        .type   v_rndne_f32_clamp_refuse_kernel,@function
v_rndne_f32_clamp_refuse_kernel:
        v_rndne_f32_e64 v0, v0 clamp
        s_endpgm

        .globl  v_fract_f32_omod_refuse_kernel
        .p2align        8
        .type   v_fract_f32_omod_refuse_kernel,@function
v_fract_f32_omod_refuse_kernel:
        v_fract_f32_e64 v0, v0 mul:2
        s_endpgm

        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_rcp_iflag_f32_clamp_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_rcp_f32_omod_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_exp_f32_clamp_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_log_f32_omod_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_ldexp_f32_clamp_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_sqrt_f32_omod_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_rsq_f32_clamp_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_floor_f32_omod_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_ceil_f32_clamp_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_trunc_f32_omod_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_rndne_f32_clamp_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel

        .p2align        6, 0x0
        .amdhsa_kernel v_fract_f32_omod_refuse_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 1
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel
        .text
        .p2alignl 7, 3214868480
        .fill 96, 4, 3214868480
        .text
        .amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_rcp_iflag_f32_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_rcp_iflag_f32_clamp_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_rcp_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_rcp_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_exp_f32_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_exp_f32_clamp_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_log_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_log_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_ldexp_f32_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_ldexp_f32_clamp_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_sqrt_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_sqrt_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_rsq_f32_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_rsq_f32_clamp_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_floor_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_floor_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_ceil_f32_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_ceil_f32_clamp_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_trunc_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_trunc_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_rndne_f32_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_rndne_f32_clamp_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fract_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_fract_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

        .end_amdgpu_metadata
