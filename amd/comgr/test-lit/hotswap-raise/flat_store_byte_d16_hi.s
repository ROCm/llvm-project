; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=flat_store_byte_d16_hi_kernel 2>&1 \
; RUN:   | %FileCheck %s

; flat_store_d16_hi_b8: D16-hi byte store lowering (lshr 16 / trunc to i8),
; the byte-store sibling of flat_store_short_d16_hi. Surfaces bits [23:16].

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	flat_store_byte_d16_hi_kernel
	.p2align	8
	.type	flat_store_byte_d16_hi_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @flat_store_byte_d16_hi_kernel(
flat_store_byte_d16_hi_kernel:
; CHECK: [[SHIFT:%.+]] = lshr i32 %{{.+}}, 16
; CHECK: [[TRUNC:%.+]] = trunc i32 [[SHIFT]] to i8
; CHECK: store i8 [[TRUNC]], ptr %{{.+}}
; CHECK-NOT: store i32 [[TRUNC]], ptr %
; CHECK-NOT: store i16 [[TRUNC]], ptr %
	v_mov_b32 v2, v0
	flat_store_d16_hi_b8 v[0:1], v2
	s_endpgm
	.rodata
	.p2align	6, 0x0
	.amdhsa_kernel flat_store_byte_d16_hi_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           flat_store_byte_d16_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         flat_store_byte_d16_hi_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
