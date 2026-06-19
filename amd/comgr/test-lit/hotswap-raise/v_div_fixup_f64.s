; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_div_fixup_f64_kernel | %FileCheck %s
;
; Lift v_div_fixup_f64 (final IEEE-divide special-case fixup) to
; llvm.amdgcn.div.fixup.f64. The second instruction checks that an
; abs/neg source modifier is applied (not silently dropped).

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  v_div_fixup_f64_kernel
        .p2align        8
        .type   v_div_fixup_f64_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_div_fixup_f64_kernel(
v_div_fixup_f64_kernel:
; CHECK: call double @llvm.amdgcn.div.fixup.f64(double %{{.+}}, double %{{.+}}, double %{{.+}})
        v_div_fixup_f64 v[0:1], v[0:1], v[2:3], v[4:5]
; CHECK: [[NEG:%[^ ]+]] = fneg double %{{.+}}
; CHECK: call double @llvm.amdgcn.div.fixup.f64(double [[NEG]], double %{{.+}}, double %{{.+}})
        v_div_fixup_f64 v[6:7], -v[0:1], v[2:3], v[4:5]
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel v_div_fixup_f64_kernel
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
    .name:           v_div_fixup_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_div_fixup_f64_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

        .end_amdgpu_metadata
