; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_div_scale_f64_kernel | %FileCheck %s
;
; Lift v_div_scale_f64 to the struct-returning
; llvm.amdgcn.div.scale.f64. Both decode arms are exercised:
;   (denom, denom, numer)  src0==src1 -> scale-denominator, i1 false
;   (numer, denom, numer)  src0==src2 -> scale-numerator,   i1 true
; The per-lane i1 carry-out is routed to the SDST (vcc_lo here).

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  v_div_scale_f64_kernel
        .p2align        8
        .type   v_div_scale_f64_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_div_scale_f64_kernel(
v_div_scale_f64_kernel:
; CHECK: %[[R:[^ ]+]] = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 false)
; CHECK: extractvalue { double, i1 } %[[R]], 0
; CHECK: extractvalue { double, i1 } %[[R]], 1
        v_div_scale_f64 v[0:1], vcc_lo, v[2:3], v[2:3], v[0:1]
; CHECK: %[[N:[^ ]+]] = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %{{.+}}, double %{{.+}}, i1 true)
; CHECK: extractvalue { double, i1 } %[[N]], 0
; CHECK: extractvalue { double, i1 } %[[N]], 1
        v_div_scale_f64 v[6:7], vcc_lo, v[2:3], v[4:5], v[2:3]
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_div_scale_f64_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_wavefront_size32 1
                .amdhsa_next_free_vgpr 8
                .amdhsa_next_free_sgpr 8
                .amdhsa_reserve_vcc 1
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
    .name:           v_div_scale_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_div_scale_f64_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

        .end_amdgpu_metadata
