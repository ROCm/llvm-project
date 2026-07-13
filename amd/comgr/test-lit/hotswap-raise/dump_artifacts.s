; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
;
; Without HSA_HOTSWAP_DUMP_INPUT the persistent dump dir gets the .ll and .s
; artifacts but no source .dis (the raiser skips the disasm string build).
; RUN: rm -rf %t.d1 && HSA_HOTSWAP_DUMP_DIR=%t.d1 %raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --write-hsaco=%t.out --kernel=dump_artifacts_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PIPE
; RUN: %FileCheck %s --check-prefix=LL < %t.d1/hotswap-*/dump_artifacts_kernel.ll
; RUN: %FileCheck %s --check-prefix=ASM < %t.d1/hotswap-*/dump_artifacts_kernel.s
; RUN: %not ls %t.d1/hotswap-*/dump_artifacts_kernel.dis
;
; With HSA_HOTSWAP_DUMP_INPUT=1 the source disassembly .dis is also written.
; RUN: rm -rf %t.d2 && HSA_HOTSWAP_DUMP_INPUT=1 HSA_HOTSWAP_DUMP_DIR=%t.d2 %raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --write-hsaco=%t.out2 --kernel=dump_artifacts_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PIPE
; RUN: %FileCheck %s --check-prefix=DIS < %t.d2/hotswap-*/dump_artifacts_kernel.dis

; PIPE: raise_cli: wrote
; PIPE-SAME: dump_artifacts_kernel

; LL: define amdgpu_kernel void @dump_artifacts_kernel(
; LL: call float @llvm.roundeven.f32(float {{.*}})

; ASM: dump_artifacts_kernel:

; DIS: v_rndne_f32

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  dump_artifacts_kernel
        .p2align        8
        .type   dump_artifacts_kernel,@function
dump_artifacts_kernel:
        v_rndne_f32 v0, v0
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel dump_artifacts_kernel
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
    .name:           dump_artifacts_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         dump_artifacts_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

        .end_amdgpu_metadata
