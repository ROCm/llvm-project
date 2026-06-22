; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_minimum_f64_kernel \
; RUN:   | %FileCheck %s

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  v_minimum_f64_kernel
        .p2align        8
        .type   v_minimum_f64_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_minimum_f64_kernel(
v_minimum_f64_kernel:
; CHECK: call double @llvm.minimum.f64(double {{.*}}, double {{.*}})
        v_minimum_f64 v[0:1], v[0:1], v[2:3]
; CHECK: [[NEG:%[^ ]+]] = fneg double %{{.+}}
; CHECK: call double @llvm.minimum.f64(double [[NEG]], double {{.*}})
        v_minimum_f64 v[4:5], -v[0:1], v[2:3]
; CHECK: [[ABS:%[^ ]+]] = call double @llvm.fabs.f64(double %{{.+}})
; CHECK: call double @llvm.minimum.f64(double [[ABS]], double {{.*}})
        v_minimum_f64 v[6:7], |v[0:1]|, v[2:3]
; CHECK: declare {{.*}}double @llvm.minimum.f64(double, double)
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_minimum_f64_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 8
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
    .name:           v_minimum_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_minimum_f64_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

        .end_amdgpu_metadata
