; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=sopc_integer \
; RUN:   | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	sopc_integer
	.p2align	8
	.type	sopc_integer,@function
; IR-LABEL: define amdgpu_kernel void @sopc_integer(
sopc_integer:
	; IR: icmp eq i32 {{.*}}
	s_cmp_eq_u32 s0, s1
	; IR: icmp ne i32 {{.*}}
	s_cmp_lg_u32 s0, s1
	; IR: icmp ugt i32 {{.*}}
	s_cmp_gt_u32 s0, s1
	; IR: icmp uge i32 {{.*}}
	s_cmp_ge_u32 s0, s1
	; IR: icmp ult i32 {{.*}}
	s_cmp_lt_u32 s0, s1
	; IR: icmp ule i32 {{.*}}
	s_cmp_le_u32 s0, s1
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: icmp eq i32 {{.*}}
	s_cmp_eq_i32 s0, s1
	; IR: icmp ne i32 {{.*}}
	s_cmp_lg_i32 s0, s1
	; IR: icmp sgt i32 {{.*}}
	s_cmp_gt_i32 s0, s1
	; IR: icmp sge i32 {{.*}}
	s_cmp_ge_i32 s0, s1
	; IR: icmp slt i32 {{.*}}
	s_cmp_lt_i32 s0, s1
	; IR: icmp sle i32 {{.*}}
	s_cmp_le_i32 s0, s1
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: icmp eq i64 {{.*}}
	s_cmp_eq_u64 s[0:1], s[2:3]
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: icmp ne i64 {{.*}}
	s_cmp_lg_u64 s[0:1], s[2:3]
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: [[BIT0_32_AMOUNT:%.*]] = and i32 {{.*}}, 31
	; IR-NEXT: [[BIT0_32_BIT:%.*]] = shl i32 1, [[BIT0_32_AMOUNT]]
	; IR-NEXT: [[BIT0_32_MASK:%.*]] = and i32 {{.*}}, [[BIT0_32_BIT]]
	; IR-NEXT: icmp eq i32 [[BIT0_32_MASK]], 0
	s_bitcmp0_b32 s0, s1
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: [[BIT1_32_AMOUNT:%.*]] = and i32 {{.*}}, 31
	; IR-NEXT: [[BIT1_32_BIT:%.*]] = shl i32 1, [[BIT1_32_AMOUNT]]
	; IR-NEXT: [[BIT1_32_MASK:%.*]] = and i32 {{.*}}, [[BIT1_32_BIT]]
	; IR-NEXT: icmp ne i32 [[BIT1_32_MASK]], 0
	s_bitcmp1_b32 s0, s1
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: [[BIT0_64_AMOUNT32:%.*]] = and i32 {{.*}}, 63
	; IR-NEXT: [[BIT0_64_AMOUNT:%.*]] = zext i32 [[BIT0_64_AMOUNT32]] to i64
	; IR-NEXT: [[BIT0_64_BIT:%.*]] = shl i64 1, [[BIT0_64_AMOUNT]]
	; IR-NEXT: [[BIT0_64_MASK:%.*]] = and i64 {{.*}}, [[BIT0_64_BIT]]
	; IR-NEXT: icmp eq i64 [[BIT0_64_MASK]], 0
	s_bitcmp0_b64 s[0:1], s2
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: [[BIT1_64_AMOUNT32:%.*]] = and i32 {{.*}}, 63
	; IR-NEXT: [[BIT1_64_AMOUNT:%.*]] = zext i32 [[BIT1_64_AMOUNT32]] to i64
	; IR-NEXT: [[BIT1_64_BIT:%.*]] = shl i64 1, [[BIT1_64_AMOUNT]]
	; IR-NEXT: [[BIT1_64_MASK:%.*]] = and i64 {{.*}}, [[BIT1_64_BIT]]
	; IR-NEXT: icmp ne i64 [[BIT1_64_MASK]], 0
	s_bitcmp1_b64 s[0:1], s2
	; IR: select i1 {{%.*}}, i32 1, i32 0
	s_cselect_b32 s4, 1, 0
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sopc_integer
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 5
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sopc_integer
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         sopc_integer.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
