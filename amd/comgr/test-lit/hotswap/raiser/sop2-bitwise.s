; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=sop2_bitwise \
; RUN:   | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	sop2_bitwise
	.p2align	8
	.type	sop2_bitwise,@function
; IR-LABEL: define amdgpu_kernel void @sop2_bitwise(
sop2_bitwise:
	; IR: %[[AND32:and]] = and i32 {{.*}}
	; IR: icmp ne i32 %[[AND32]], 0
	s_and_b32 s2, s0, s1
	; IR: %[[AND64:and64]] = and i64 {{%.*}}, {{%.*}}
	; IR: icmp ne i64 %[[AND64]], 0
	s_and_b64 s[2:3], s[0:1], s[4:5]
	; IR: %[[OR32:or]] = or i32 {{.*}}
	; IR: icmp ne i32 %[[OR32]], 0
	s_or_b32 s2, s0, s1
	; IR: %[[OR64:or64]] = or i64 {{%.*}}, {{%.*}}
	; IR: icmp ne i64 %[[OR64]], 0
	s_or_b64 s[2:3], s[0:1], s[4:5]
	; IR: %[[XOR32:xor]] = xor i32 {{.*}}
	; IR: icmp ne i32 %[[XOR32]], 0
	s_xor_b32 s2, s0, s1
	; IR: %[[XOR64:xor64]] = xor i64 {{%.*}}, {{%.*}}
	; IR: icmp ne i64 %[[XOR64]], 0
	s_xor_b64 s[2:3], s[0:1], s[4:5]
	; IR: [[ANDN32_NOT:%.*]] = xor i32 {{.*}}, -1
	; IR-NEXT: %[[ANDN32:andn2]] = and i32 {{.*}}, [[ANDN32_NOT]]
	; IR: icmp ne i32 %[[ANDN32]], 0
	s_andn2_b32 s2, s0, s1
	; IR: [[ANDN64_NOT:%.*]] = xor i64 {{.*}}, -1
	; IR-NEXT: %[[ANDN64:andn2_64]] = and i64 {{.*}}, [[ANDN64_NOT]]
	; IR: icmp ne i64 %[[ANDN64]], 0
	s_andn2_b64 s[2:3], s[0:1], s[4:5]
	; IR: [[ORN32_NOT:%.*]] = xor i32 {{.*}}, -1
	; IR-NEXT: [[ORN32:%.*]] = or i32 {{.*}}, [[ORN32_NOT]]
	; IR: icmp ne i32 [[ORN32]], 0
	s_orn2_b32 s2, s0, s1
	; IR: [[ORN64_NOT:%.*]] = xor i64 {{.*}}, -1
	; IR-NEXT: [[ORN64:%.*]] = or i64 {{.*}}, [[ORN64_NOT]]
	; IR: icmp ne i64 [[ORN64]], 0
	s_orn2_b64 s[2:3], s[0:1], s[4:5]
	; IR: [[NAND32_AND:%.*]] = and i32 {{.*}}
	; IR-NEXT: [[NAND32:%.*]] = xor i32 [[NAND32_AND]], -1
	; IR: icmp ne i32 [[NAND32]], 0
	s_nand_b32 s2, s0, s1
	; IR: [[NAND64_AND:%.*]] = and i64 {{%.*}}, {{%.*}}
	; IR-NEXT: [[NAND64:%.*]] = xor i64 [[NAND64_AND]], -1
	; IR: icmp ne i64 [[NAND64]], 0
	s_nand_b64 s[2:3], s[0:1], s[4:5]
	; IR: [[NOR32_OR:%.*]] = or i32 {{.*}}
	; IR-NEXT: [[NOR32:%.*]] = xor i32 [[NOR32_OR]], -1
	; IR: icmp ne i32 [[NOR32]], 0
	s_nor_b32 s2, s0, s1
	; IR: [[NOR64:%.*]] = xor i64 {{%.*}}, -1
	; IR: icmp ne i64 [[NOR64]], 0
	s_nor_b64 s[2:3], s[0:1], s[4:5]
	; IR: [[XNOR32_XOR:%.*]] = xor i32 {{.*}}
	; IR-NEXT: [[XNOR32:%.*]] = xor i32 [[XNOR32_XOR]], -1
	; IR: icmp ne i32 [[XNOR32]], 0
	s_xnor_b32 s2, s0, s1
	; IR: [[XNOR64_XOR:%.*]] = xor i64 {{%.*}}, {{%.*}}
	; IR-NEXT: [[XNOR64:%.*]] = xor i64 [[XNOR64_XOR]], -1
	; IR: icmp ne i64 [[XNOR64]], 0
	s_xnor_b64 s[2:3], s[0:1], s[4:5]
	; IR: [[LSHL32_AMOUNT:%.*]] = and i32 {{.*}}, 31
	; IR-NEXT: [[LSHL32:%.*]] = shl i32 {{.*}}, [[LSHL32_AMOUNT]]
	; IR: icmp ne i32 [[LSHL32]], 0
	s_lshl_b32 s2, s0, s1
	; IR: [[LSHL64_AMOUNT32:%.*]] = and i32 {{.*}}, 63
	; IR-NEXT: [[LSHL64_AMOUNT:%.*]] = zext i32 [[LSHL64_AMOUNT32]] to i64
	; IR-NEXT: [[LSHL64:%.*]] = shl i64 {{.*}}, [[LSHL64_AMOUNT]]
	; IR: icmp ne i64 [[LSHL64]], 0
	s_lshl_b64 s[2:3], s[0:1], s4
	; IR: [[LSHR32_AMOUNT:%.*]] = and i32 {{.*}}, 31
	; IR-NEXT: [[LSHR32:%.*]] = lshr i32 {{.*}}, [[LSHR32_AMOUNT]]
	; IR: icmp ne i32 [[LSHR32]], 0
	s_lshr_b32 s2, s0, s1
	; IR: [[LSHR64_AMOUNT32:%.*]] = and i32 {{.*}}, 63
	; IR-NEXT: [[LSHR64_AMOUNT:%.*]] = zext i32 [[LSHR64_AMOUNT32]] to i64
	; IR-NEXT: [[LSHR64:%.*]] = lshr i64 {{.*}}, [[LSHR64_AMOUNT]]
	; IR: icmp ne i64 [[LSHR64]], 0
	s_lshr_b64 s[2:3], s[0:1], s4
	; IR: [[ASHR32_AMOUNT:%.*]] = and i32 {{.*}}, 31
	; IR-NEXT: [[ASHR32:%.*]] = ashr i32 {{.*}}, [[ASHR32_AMOUNT]]
	; IR: icmp ne i32 [[ASHR32]], 0
	s_ashr_i32 s2, s0, s1
	; IR: [[ASHR64_AMOUNT32:%.*]] = and i32 {{.*}}, 63
	; IR-NEXT: [[ASHR64_AMOUNT:%.*]] = zext i32 [[ASHR64_AMOUNT32]] to i64
	; IR-NEXT: [[ASHR64:%.*]] = ashr i64 {{.*}}, [[ASHR64_AMOUNT]]
	; IR: icmp ne i64 [[ASHR64]], 0
	s_ashr_i64 s[2:3], s[0:1], s4
	; IR: [[BFEU_SHIFT:%.*]] = and i32 {{.*}}, 31
	; IR: [[BFEU_LENGTH:%.*]] = and i32 {{.*}}, 127
	; IR: [[BFEU_MASK:%.*]] = select i1 {{.*}}, i32 -1, i32 {{.*}}
	; IR: [[BFEU_EXTRACT:%.*]] = and i32 {{.*}}, [[BFEU_MASK]]
	; IR: [[BFEU:%.*]] = select i1 {{.*}}, i32 0, i32 [[BFEU_EXTRACT]]
	; IR: icmp ne i32 [[BFEU]], 0
	s_bfe_u32 s2, s0, s1
	; IR: [[BFEI_LENGTH:%.*]] = and i32 {{.*}}, 127
	; IR: [[BFEI_EXTRACT:%.*]] = ashr i32 {{.*}}
	; IR: [[BFEI_SAT:%.*]] = ashr i32 {{.*}}
	; IR: [[BFEI_NONEMPTY:%.*]] = select i1 {{.*}}, i32 [[BFEI_EXTRACT]], i32 [[BFEI_SAT]]
	; IR-NEXT: {{%.*}} = icmp eq i32 [[BFEI_LENGTH]], 0
	; IR-NEXT: [[BFEI:%.*]] = select i1 {{.*}}, i32 0, i32 [[BFEI_NONEMPTY]]
	; IR: icmp ne i32 [[BFEI]], 0
	s_bfe_i32 s2, s0, s1
	; IR: [[BFEI64_LENGTH32:%.*]] = and i32 {{.*}}, 127
	; IR: [[BFEI64_LENGTH:%.*]] = zext i32 [[BFEI64_LENGTH32]] to i64
	; IR: [[BFEI64_EXTRACT:%.*]] = ashr i64 {{.*}}
	; IR: [[BFEI64_SAT:%.*]] = ashr i64 {{.*}}
	; IR: [[BFEI64_NONEMPTY:%.*]] = select i1 {{.*}}, i64 [[BFEI64_EXTRACT]], i64 [[BFEI64_SAT]]
	; IR-NEXT: {{%.*}} = icmp eq i64 [[BFEI64_LENGTH]], 0
	; IR-NEXT: [[BFEI64:%.*]] = select i1 {{.*}}, i64 0, i64 [[BFEI64_NONEMPTY]]
	; IR: [[BFEI64_SCC:%.*]] = icmp ne i64 [[BFEI64]], 0
	s_bfe_i64 s[2:3], s[0:1], s4
	; IR: [[BFM32_WIDTH:%.*]] = and i32 {{.*}}, 31
	; IR: [[BFM32_ONE:%.*]] = shl i32 1, [[BFM32_WIDTH]]
	; IR-NEXT: [[BFM32_MASK:%.*]] = sub i32 [[BFM32_ONE]], 1
	; IR-NEXT: [[BFM32:%.*]] = shl i32 [[BFM32_MASK]], {{.*}}
	s_bfm_b32 s2, s0, s1
	; IR: [[BFM64_WIDTH32:%.*]] = and i32 {{.*}}, 63
	; IR: [[BFM64_WIDTH:%.*]] = zext i32 [[BFM64_WIDTH32]] to i64
	; IR: [[BFM64_ONE:%.*]] = shl i64 1, [[BFM64_WIDTH]]
	; IR-NEXT: [[BFM64_MASK:%.*]] = sub i64 [[BFM64_ONE]], 1
	; IR-NEXT: [[BFM64:%.*]] = shl i64 [[BFM64_MASK]], {{.*}}
	s_bfm_b64 s[2:3], s0, s1
	; IR: select i1 [[BFEI64_SCC]], i32 {{.*}}, i32 {{.*}}
	s_cselect_b32 s4, s0, s1
	s_mov_b32 s0, 0
	s_mov_b32 s1, 1
	; IR: %[[B32_SCALAR:.*]] = and i32 {{.*}}, -1
	; IR: %[[B32_SCC:.*]] = icmp ne i32 %[[B32_SCALAR]], 0
	s_and_b32 s2, s0, -1
	; IR: select i1 %[[B32_SCC]], i32 1, i32 0
	s_cselect_b32 s3, 1, 0
	; IR: [[PACK_LL_LO:%.*]] = and i32 {{.*}}, 65535
	; IR-NEXT: [[PACK_LL_HI16:%.*]] = and i32 {{.*}}, 65535
	; IR-NEXT: [[PACK_LL_HI:%.*]] = shl i32 [[PACK_LL_HI16]], 16
	; IR-NEXT: {{%.*}} = or i32 [[PACK_LL_LO]], [[PACK_LL_HI]]
	s_pack_ll_b32_b16 s2, s0, s1
	; IR: [[PACK_LH_LO:%.*]] = and i32 {{.*}}, 65535
	; IR-NEXT: [[PACK_LH_HI:%.*]] = and i32 {{.*}}, -65536
	; IR-NEXT: {{%.*}} = or i32 [[PACK_LH_LO]], [[PACK_LH_HI]]
	s_pack_lh_b32_b16 s2, s0, s1
	; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sop2_bitwise
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 6
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
    .name:           sop2_bitwise
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         sop2_bitwise.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
