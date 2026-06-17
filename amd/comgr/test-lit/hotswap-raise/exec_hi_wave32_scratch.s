; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=exec_hi_cndmask_kernel | %FileCheck %s
;
; A wave32 exec_hi is a scratch scalar symmetric to vcc_hi; consumers must route
; it to its own scratch slot, not the real EXEC or VCC. See
; ParsedReg::VCC_HI_SCRATCH.

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  exec_hi_cndmask_kernel
        .p2align        8
        .type   exec_hi_cndmask_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @exec_hi_cndmask_kernel(
; The exec_hi scratch is promoted to SSA, not left as a private alloca:
; CHECK-NOT: %ExecHiScratch = alloca
exec_hi_cndmask_kernel:
        s_mov_b32 exec_hi, 42
; CHECK: %vcmp = icmp slt
        v_cmp_lt_i32 vcc_lo, v0, v1
; The cndmask condition is the per-lane bit of the exec_hi scratch, not %vcmp:
; CHECK: %[[LANEBIT:wn_mask_lane_i1[0-9]*]] = icmp ne i64 %{{.*}}, 0
; CHECK: %cndmask = select i1 %[[LANEBIT]], i32 {{.*}}, i32 %tid
; CHECK-NOT: %cndmask = select i1 %vcmp
        v_cndmask_b32 v5, v0, v1, exec_hi
; The exec_hi read-back propagates the scratch constant 42:
; CHECK: store i32 42, ptr addrspace(3)
        v_mov_b32 v2, exec_hi
        ds_store_b32 v3, v2
        ds_store_b32 v6, v5
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel exec_hi_cndmask_kernel
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
    .name:           exec_hi_cndmask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         exec_hi_cndmask_kernel.kd
    .vgpr_count:     7
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
        .end_amdgpu_metadata
