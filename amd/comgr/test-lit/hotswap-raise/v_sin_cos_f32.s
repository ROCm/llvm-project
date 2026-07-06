; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_sin_cos_f32_kernel | %FileCheck %s
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_sin_f32_clamp_refuse_kernel 2>&1 | %FileCheck %s --check-prefix=SIN-REFUSE
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cos_f32_omod_refuse_kernel 2>&1 | %FileCheck %s --check-prefix=COS-REFUSE
;
; The ISA manual defines v_sin_f32/v_cos_f32 as F32 -> F32 vector TRANS
; instructions computing sin/cos(src * 2*pi), with full-range inputs and
; denormal support. Preserve that hardware contract through the AMDGPU
; intrinsics rather than generic LLVM radian-domain sin/cos intrinsics.

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text

; CHECK-LABEL: define amdgpu_kernel void @v_sin_cos_f32_kernel(
; CHECK: call float @llvm.amdgcn.sin.f32(float {{.*}})
; CHECK: call float @llvm.amdgcn.cos.f32(float {{.*}})
; CHECK: [[NEG:%[^ ]+]] = fneg float {{.*}}
; CHECK: [[NEGBITS:%[^ ]+]] = bitcast float [[NEG]] to i32
; CHECK: [[NEGSRC:%[^ ]+]] = bitcast i32 [[NEGBITS]] to float
; CHECK: call float @llvm.amdgcn.sin.f32(float [[NEGSRC]])
; CHECK: [[ABS:%[^ ]+]] = call float @llvm.fabs.f32(float {{.*}})
; CHECK: [[ABSBITS:%[^ ]+]] = bitcast float [[ABS]] to i32
; CHECK: [[ABSSRC:%[^ ]+]] = bitcast i32 [[ABSBITS]] to float
; CHECK: call float @llvm.amdgcn.cos.f32(float [[ABSSRC]])
; CHECK-NOT: call {{.*}}@llvm.sin.f32
; CHECK-NOT: call {{.*}}@llvm.cos.f32
; CHECK: declare {{.*}}float @llvm.amdgcn.sin.f32(float)
; CHECK: declare {{.*}}float @llvm.amdgcn.cos.f32(float)
; CHECK: declare {{.*}}float @llvm.fabs.f32(float)
        .globl  v_sin_cos_f32_kernel
        .p2align        8
        .type   v_sin_cos_f32_kernel,@function
v_sin_cos_f32_kernel:
        v_sin_f32 v0, v0
        v_cos_f32 v1, v1
        v_sin_f32_e64 v2, -v2
        v_cos_f32_e64 v3, |v3|
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_sin_cos_f32_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 4
                .amdhsa_next_free_sgpr 8
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel
        .text
        .p2alignl 7, 3214868480
        .fill 96, 4, 3214868480

; SIN-REFUSE-DAG: kernel 'v_sin_f32_clamp_refuse_kernel'
; SIN-REFUSE-DAG: V_SIN_F32 with non-default clamp/omod is not yet lifted
; SIN-REFUSE-DAG: output modifier semantics must not be silently dropped
        .globl  v_sin_f32_clamp_refuse_kernel
        .p2align        8
        .type   v_sin_f32_clamp_refuse_kernel,@function
v_sin_f32_clamp_refuse_kernel:
        v_sin_f32_e64 v0, v0 clamp
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_sin_f32_clamp_refuse_kernel
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

; COS-REFUSE-DAG: kernel 'v_cos_f32_omod_refuse_kernel'
; COS-REFUSE-DAG: V_COS_F32 with non-default clamp/omod is not yet lifted
; COS-REFUSE-DAG: output modifier semantics must not be silently dropped
        .globl  v_cos_f32_omod_refuse_kernel
        .p2align        8
        .type   v_cos_f32_omod_refuse_kernel,@function
v_cos_f32_omod_refuse_kernel:
        v_cos_f32_e64 v0, v0 mul:2
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_cos_f32_omod_refuse_kernel
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
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_sin_cos_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_sin_cos_f32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_sin_f32_clamp_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_sin_f32_clamp_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cos_f32_omod_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cos_f32_omod_refuse_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

        .end_amdgpu_metadata
