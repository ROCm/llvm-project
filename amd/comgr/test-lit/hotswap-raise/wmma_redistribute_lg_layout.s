; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=wmma_redistribute_lg_layout_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; wmma_f32_16x16x32_bf16 lane-group (lg) layout redistribution via ds.bpermute for wave translation.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	wmma_redistribute_lg_layout_kernel
	.p2align	8
	.type	wmma_redistribute_lg_layout_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @wmma_redistribute_lg_layout_kernel(
wmma_redistribute_lg_layout_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
	; CHECK: %addr_lo = shl i32 %{{.+}}, 2
	; CHECK: %addr_hi = shl i32 %{{.+}}, 2
	; CHECK: %lane_grp = lshr i32 %{{.+}}, 4
	; CHECK: %[[V0:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_lo, i32 %[[DWG:dw[0-9]*]])
	; CHECK-NEXT: %[[V1:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_lo, i32 %[[DWG2:dw[0-9]*]])
	; CHECK-NEXT: %[[V2:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_hi, i32 %[[DWG]])
	; CHECK-NEXT: %[[V3:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_hi, i32 %[[DWG2]])
	; CHECK-NEXT: %[[EQ2:[0-9]+]] = icmp eq i32 %lane_grp, 2
	; CHECK-NEXT: %{{[0-9]+}} = select i1 %[[EQ2]], i32 %[[V2]], i32 %[[V3]]
	; CHECK-NEXT: %[[EQ1:[0-9]+]] = icmp eq i32 %lane_grp, 1
	; CHECK-NEXT: %{{[0-9]+}} = select i1 %[[EQ1]], i32 %[[V1]], i32 %{{[0-9]+}}
	; CHECK-NEXT: %[[EQ0:[0-9]+]] = icmp eq i32 %lane_grp, 0
	; CHECK-NEXT: %{{[0-9]+}} = select i1 %[[EQ0]], i32 %[[V0]], i32 %{{[0-9]+}}
	global_store_b128 v24, v[16:19], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_redistribute_lg_layout_kernel
		.amdhsa_next_free_vgpr 25
		.amdhsa_next_free_sgpr 2
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wmma_redistribute_lg_layout_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         wmma_redistribute_lg_layout_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
