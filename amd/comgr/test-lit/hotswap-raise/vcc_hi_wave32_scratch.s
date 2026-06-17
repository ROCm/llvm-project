; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vcc_hi_scratch_kernel 2>/dev/null | %FileCheck %s
;
; Wave32 source raised to a wave64 target.  Exercises the wave32 VCC_HI
; handling: parseReg (raise-context.cpp) routes VCC_HI to its own
; VCC_HI_SCRATCH slot, and collectAllocas (reg-file.cpp) promotes that slot,
; so a VCC-writing v_cmp cannot clobber the scratch value.  Per-instruction
; CHECKs are inline in the kernel body below.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	vcc_hi_scratch_kernel
	.p2align	8
	.type	vcc_hi_scratch_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @vcc_hi_scratch_kernel(
vcc_hi_scratch_kernel:
	s_mov_b32 vcc_hi, 42
	; The scratch slot is promoted, never left as a private alloca:
	; CHECK-NOT: %VccHiScratch = alloca
	v_cmp_lt_i32 vcc_lo, v0, v1
	; CHECK: %[[VCMP:[a-zA-Z0-9_.]+]] = icmp slt
	v_cndmask_b32 v5, v0, v1, vcc_lo
	; CHECK: select i1 %[[VCMP]]
	v_mov_b32 v2, vcc_hi
	ds_store_b32 v3, v2
	; vcc_hi still reads 42 -- distinct from the VCC the v_cmp wrote -- and
	; reaches the store as a propagated constant (proving it was promoted):
	; CHECK: store i32 42, ptr addrspace(3)
	ds_store_b32 v6, v5
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vcc_hi_scratch_kernel
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 8
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
    .name:           vcc_hi_scratch_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         vcc_hi_scratch_kernel.kd
    .vgpr_count:     7
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
