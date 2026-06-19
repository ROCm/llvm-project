; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_frexp_mant_f64_kernel | %FileCheck %s
;
; Lift v_frexp_mant_f64 to llvm.amdgcn.frexp.mant.f64, not the
; struct-returning llvm.frexp (different inf/NaN semantics). The declare
; check pins the f64 overload, and the second instruction checks an
; abs/neg source modifier is applied.

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  v_frexp_mant_f64_kernel
        .p2align        8
        .type   v_frexp_mant_f64_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_frexp_mant_f64_kernel(
v_frexp_mant_f64_kernel:
; CHECK: call double @llvm.amdgcn.frexp.mant.f64(double %{{.+}})
        v_frexp_mant_f64 v[0:1], v[0:1]
; CHECK: [[NEG:%[^ ]+]] = fneg double %{{.+}}
; CHECK: call double @llvm.amdgcn.frexp.mant.f64(double [[NEG]])
        v_frexp_mant_f64_e64 v[2:3], -v[0:1]
; CHECK: declare double @llvm.amdgcn.frexp.mant.f64(double)
; CHECK-NOT: @llvm.frexp.f64
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_frexp_mant_f64_kernel
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
        .text
        .amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_frexp_mant_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_frexp_mant_f64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

        .end_amdgpu_metadata
