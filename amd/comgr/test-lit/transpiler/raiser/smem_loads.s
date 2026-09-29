; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s \
; RUN:   -o %t.gfx1250.o
; RUN: %ld.lld -shared %t.gfx1250.o -o %t.gfx1250.hsaco

; RUN: %transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=smem_loads,smem_wide_loads,smem_wide_overlap \
; RUN:   --emit-ir=smem_register_offset,smem_wide_register_offset \
; RUN:   --emit-ir=smem_soffset_overlap,smem_scale_offset \
; RUN:   --emit-ir=smem_narrow_loads,smem_narrow_scale_offset \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: not %transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=smem_cache_policy,smem_buffer_load \
; RUN:   --emit-ir=smem_negative_offset 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	smem_loads
	.p2align	8
	.type	smem_loads,@function
; IR-LABEL: define amdgpu_kernel void @smem_loads(
smem_loads:
; The i64 -4 mask clears the two low address bits.
; IR: [[BASE0:%.+]] = and i64 {{%.+}}, -4
; IR: [[ADDRESS0:%.+]] = add i64 [[BASE0]], 0
; IR: [[POINTER0:%.+]] = inttoptr i64 [[ADDRESS0]] to ptr addrspace(1)
; IR: [[LOAD128:%.+]] = load <4 x i32>, ptr addrspace(1) [[POINTER0]], align 4
	s_load_b128 s[4:7], s[0:1], 0x3
; IR: [[LOAD128_BITS:%.+]] = bitcast <4 x i32> [[LOAD128]] to i128
; IR: [[LOAD128_WORD2_SHIFTED:%.+]] = lshr i128 [[LOAD128_BITS]], 64
; IR: [[LOAD128_WORD2:%.+]] = trunc i128 [[LOAD128_WORD2_SHIFTED]] to i32
; IR: [[LOAD128_WORD3_SHIFTED:%.+]] = lshr i128 [[LOAD128_BITS]], 96
; IR: [[LOAD128_WORD3:%.+]] = trunc i128 [[LOAD128_WORD3_SHIFTED]] to i32
; IR: [[LOAD128_WORD2_EXT:%.+]] = zext i32 [[LOAD128_WORD2]] to i64
; IR: [[LOAD128_WORD3_EXT:%.+]] = zext i32 [[LOAD128_WORD3]] to i64
; IR: [[LOAD128_WORD3_BITS:%.+]] = shl i64 [[LOAD128_WORD3_EXT]], 32
; IR: [[BASE1_BITS:%.+]] = or i64 [[LOAD128_WORD2_EXT]], [[LOAD128_WORD3_BITS]]

; IR: [[BASE1:%.+]] = and i64 [[BASE1_BITS]], -4
; IR: [[ADDRESS1:%.+]] = add i64 [[BASE1]], 4
; IR: [[POINTER1:%.+]] = inttoptr i64 [[ADDRESS1]] to ptr addrspace(1)
; IR: [[LOAD64:%.+]] = load i64, ptr addrspace(1) [[POINTER1]], align 4
	s_load_b64 s[2:3], s[6:7], 0x7
; IR: [[LOAD64_LO:%.+]] = trunc i64 [[LOAD64]] to i32
; IR: [[LOAD64_SHIFTED:%.+]] = lshr i64 [[LOAD64]], 32
; IR: [[LOAD64_HI:%.+]] = trunc i64 [[LOAD64_SHIFTED]] to i32
; IR: [[LOAD64_LO_EXT:%.+]] = zext i32 [[LOAD64_LO]] to i64
; IR: [[LOAD64_HI_EXT:%.+]] = zext i32 [[LOAD64_HI]] to i64
; IR: [[LOAD64_HI_BITS:%.+]] = shl i64 [[LOAD64_HI_EXT]], 32
; IR: [[BASE2_BITS:%.+]] = or i64 [[LOAD64_LO_EXT]], [[LOAD64_HI_BITS]]

; IR: [[BASE2:%.+]] = and i64 [[BASE2_BITS]], -4
; IR: [[ADDRESS2:%.+]] = add i64 [[BASE2]], 8
; IR: [[POINTER2:%.+]] = inttoptr i64 [[ADDRESS2]] to ptr addrspace(1)
; IR: [[LOAD32:%.+]] = load i32, ptr addrspace(1) [[POINTER2]], align 4
	s_load_b32 s4, s[2:3], 0xb
; IR: [[LOAD32_EXT:%.+]] = zext i32 [[LOAD32]] to i64
; IR: [[BASE3_BITS:%.+]] = or i64 [[LOAD32_EXT]], {{%.+}}

; IR: [[BASE3:%.+]] = and i64 [[BASE3_BITS]], -4
; IR: [[ADDRESS3:%.+]] = add i64 [[BASE3]], 12
; IR: [[POINTER3:%.+]] = inttoptr i64 [[ADDRESS3]] to ptr addrspace(1)
; IR: load i32, ptr addrspace(1) [[POINTER3]], align 4
	s_load_b32 s8, s[4:5], 0xf
; IR: ret void
	s_endpgm

	.globl	smem_wide_loads
	.p2align	8
	.type	smem_wide_loads,@function
; IR-LABEL: define amdgpu_kernel void @smem_wide_loads(
smem_wide_loads:
; IR: [[BASE96:%.+]] = and i64 {{%.+}}, -4
; IR: [[ADDRESS96:%.+]] = add i64 [[BASE96]], 0
; IR: [[POINTER96:%.+]] = inttoptr i64 [[ADDRESS96]] to ptr addrspace(1)
; IR: [[LOAD96:%.+]] = load <3 x i32>, ptr addrspace(1) [[POINTER96]], align 4
	s_load_b96 s[4:6], s[0:1], 0x0
; IR: [[LOAD96_BITS:%.+]] = bitcast <3 x i32> [[LOAD96]] to i96
; IR: trunc i96 [[LOAD96_BITS]] to i32
; IR: [[LOAD96_WORD2_SHIFTED:%.+]] = lshr i96 [[LOAD96_BITS]], 64
; IR: trunc i96 [[LOAD96_WORD2_SHIFTED]] to i32

; IR: [[ADDRESS256:%.+]] = add i64 {{%.+}}, 16
; IR: [[POINTER256:%.+]] = inttoptr i64 [[ADDRESS256]] to ptr addrspace(1)
; IR: [[LOAD256:%.+]] = load <8 x i32>, ptr addrspace(1) [[POINTER256]], align 4
	s_load_b256 s[8:15], s[0:1], 0x10
; IR: [[LOAD256_BITS:%.+]] = bitcast <8 x i32> [[LOAD256]] to i256
; IR: trunc i256 [[LOAD256_BITS]] to i32
; IR: [[LOAD256_WORD7_SHIFTED:%.+]] = lshr i256 [[LOAD256_BITS]], 224
; IR: trunc i256 [[LOAD256_WORD7_SHIFTED]] to i32

; IR: [[ADDRESS512:%.+]] = add i64 {{%.+}}, 64
; IR: [[POINTER512:%.+]] = inttoptr i64 [[ADDRESS512]] to ptr addrspace(1)
; IR: [[LOAD512:%.+]] = load <16 x i32>, ptr addrspace(1) [[POINTER512]], align 4
	s_load_b512 s[16:31], s[0:1], 0x40
; IR: [[LOAD512_BITS:%.+]] = bitcast <16 x i32> [[LOAD512]] to i512
; IR: trunc i512 [[LOAD512_BITS]] to i32
; IR: [[LOAD512_WORD15_SHIFTED:%.+]] = lshr i512 [[LOAD512_BITS]], 480
; IR: trunc i512 [[LOAD512_WORD15_SHIFTED]] to i32
; IR: ret void
	s_endpgm

; Test overlap with the source base pair.

	.globl	smem_wide_overlap
	.p2align	8
	.type	smem_wide_overlap,@function
; IR-LABEL: define amdgpu_kernel void @smem_wide_overlap(
smem_wide_overlap:
; IR: [[OVERLAP_BASE:%.+]] = and i64 {{%.+}}, -4
; IR: [[OVERLAP_ADDRESS:%.+]] = add i64 [[OVERLAP_BASE]], 4
; IR: [[OVERLAP_POINTER:%.+]] = inttoptr i64 [[OVERLAP_ADDRESS]] to ptr addrspace(1)
; IR: [[OVERLAP_LOAD:%.+]] = load <3 x i32>, ptr addrspace(1) [[OVERLAP_POINTER]], align 4
	s_load_b96 s[0:2], s[0:1], 0x4
; IR: bitcast <3 x i32> [[OVERLAP_LOAD]] to i96
; IR: ret void
	s_endpgm

; The SGPR and immediate offsets are intentionally unaligned. Each is rounded
; down to a dword before the address components are added.

	.globl	smem_register_offset
	.p2align	8
	.type	smem_register_offset,@function
; IR-LABEL: define amdgpu_kernel void @smem_register_offset(
smem_register_offset:
	s_mov_b32 s4, 0x13
; IR: [[RO_BASE:%.+]] = and i64 {{.+}}, -4
; IR: [[RO_ADDRESS:%.+]] = add i64 [[RO_BASE]], 32
; IR: [[RO_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR: [[RO_SOFFSET_DWORD:%.+]] = and i64 [[RO_SOFFSET]], -4
; IR: [[RO_SUM:%.+]] = add i64 [[RO_ADDRESS]], [[RO_SOFFSET_DWORD]]
; IR: [[RO_POINTER:%.+]] = inttoptr i64 [[RO_SUM]] to ptr addrspace(1)
; IR: load i32, ptr addrspace(1) [[RO_POINTER]], align 4
	s_load_b32 s2, s[0:1], s4 offset:0x23
; IR: ret void
	s_endpgm

	.globl	smem_wide_register_offset
	.p2align	8
	.type	smem_wide_register_offset,@function
; IR-LABEL: define amdgpu_kernel void @smem_wide_register_offset(
smem_wide_register_offset:
	s_mov_b32 s8, 0x13
; IR: [[WRO_BASE:%.+]] = and i64 {{.+}}, -4
; IR: [[WRO_ADDRESS:%.+]] = add i64 [[WRO_BASE]], 0
; IR: [[WRO_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR: [[WRO_SOFFSET_DWORD:%.+]] = and i64 [[WRO_SOFFSET]], -4
; IR: [[WRO_SUM:%.+]] = add i64 [[WRO_ADDRESS]], [[WRO_SOFFSET_DWORD]]
; IR: [[WRO_POINTER:%.+]] = inttoptr i64 [[WRO_SUM]] to ptr addrspace(1)
; IR: load <3 x i32>, ptr addrspace(1) [[WRO_POINTER]], align 4
	s_load_b96 s[4:6], s[0:1], s8
; IR: ret void
	s_endpgm

; Test overlap with the SGPR offset.

	.globl	smem_soffset_overlap
	.p2align	8
	.type	smem_soffset_overlap,@function
; IR-LABEL: define amdgpu_kernel void @smem_soffset_overlap(
smem_soffset_overlap:
	s_mov_b32 s6, 0x10
; IR: [[SO_BASE:%.+]] = and i64 {{.+}}, -4
; IR: [[SO_ADDRESS:%.+]] = add i64 [[SO_BASE]], 0
; IR: [[SO_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR: [[SO_SOFFSET_DWORD:%.+]] = and i64 [[SO_SOFFSET]], -4
; IR: [[SO_SUM:%.+]] = add i64 [[SO_ADDRESS]], [[SO_SOFFSET_DWORD]]
; IR: [[SO_POINTER:%.+]] = inttoptr i64 [[SO_SUM]] to ptr addrspace(1)
; IR: [[SO_LOAD:%.+]] = load <3 x i32>, ptr addrspace(1) [[SO_POINTER]], align 4
	s_load_b96 s[4:6], s[0:1], s6
; IR: bitcast <3 x i32> [[SO_LOAD]] to i96
; IR: ret void
	s_endpgm

; SCALE_OFFSET scales the SGPR element index by the load width before alignment.

	.globl	smem_scale_offset
	.p2align	8
	.type	smem_scale_offset,@function
; IR-LABEL: define amdgpu_kernel void @smem_scale_offset(
smem_scale_offset:
	s_mov_b32 s8, 0x13
; IR: [[SC96_BASE:%.+]] = and i64 {{.+}}, -4
; IR: [[SC96_ADDRESS:%.+]] = add i64 [[SC96_BASE]], 0
; IR: [[SC96_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR: [[SC96_SCALED:%.+]] = mul i64 [[SC96_SOFFSET]], 12
; IR: [[SC96_DWORD:%.+]] = and i64 [[SC96_SCALED]], -4
; IR: [[SC96_SUM:%.+]] = add i64 [[SC96_ADDRESS]], [[SC96_DWORD]]
; IR: [[SC96_POINTER:%.+]] = inttoptr i64 [[SC96_SUM]] to ptr addrspace(1)
; IR: load <3 x i32>, ptr addrspace(1) [[SC96_POINTER]], align 4
	s_load_b96 s[4:6], s[0:1], s8 scale_offset
; IR: [[SC512_BASE:%.+]] = and i64 {{.+}}, -4
; IR: [[SC512_ADDRESS:%.+]] = add i64 [[SC512_BASE]], 0
; IR: [[SC512_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR: [[SC512_SCALED:%.+]] = mul i64 [[SC512_SOFFSET]], 64
; IR: [[SC512_DWORD:%.+]] = and i64 [[SC512_SCALED]], -4
; IR: [[SC512_SUM:%.+]] = add i64 [[SC512_ADDRESS]], [[SC512_DWORD]]
; IR: [[SC512_POINTER:%.+]] = inttoptr i64 [[SC512_SUM]] to ptr addrspace(1)
; IR: load <16 x i32>, ptr addrspace(1) [[SC512_POINTER]], align 4
	s_load_b512 s[16:31], s[0:1], s8 scale_offset
; IR: ret void
	s_endpgm

; A byte load uses the full address; a 16-bit load ignores only bit 0. Both
; widen into a single dword, zero extending for u8/u16 and sign extending for
; i8/i16.

	.globl	smem_narrow_loads
	.p2align	8
	.type	smem_narrow_loads,@function
; IR-LABEL: define amdgpu_kernel void @smem_narrow_loads(
smem_narrow_loads:
; IR-NOT: and i64 {{.+}}, -1
; IR: [[U8_ADDRESS:%.+]] = add i64 {{%.+}}, 3
; IR: [[U8_POINTER:%.+]] = inttoptr i64 [[U8_ADDRESS]] to ptr addrspace(1)
; IR: [[U8_LOAD:%.+]] = load i8, ptr addrspace(1) [[U8_POINTER]], align 1
; IR: [[U8_VALUE:%.+]] = zext i8 [[U8_LOAD]] to i32
	s_load_u8 s2, s[0:1], 0x3
; IR: xor i32 [[U8_VALUE]], -1
	s_not_b32 s3, s2
; IR-NOT: and i64 {{.+}}, -1
; IR: [[I8_ADDRESS:%.+]] = add i64 {{%.+}}, 3
; IR: [[I8_POINTER:%.+]] = inttoptr i64 [[I8_ADDRESS]] to ptr addrspace(1)
; IR: [[I8_LOAD:%.+]] = load i8, ptr addrspace(1) [[I8_POINTER]], align 1
; IR: [[I8_VALUE:%.+]] = sext i8 [[I8_LOAD]] to i32
	s_load_i8 s4, s[0:1], 0x3
; IR: xor i32 [[I8_VALUE]], -1
	s_not_b32 s5, s4
; IR: [[U16_BASE:%.+]] = and i64 {{%.+}}, -2
; IR: [[U16_ADDRESS:%.+]] = add i64 [[U16_BASE]], 2
; IR: [[U16_POINTER:%.+]] = inttoptr i64 [[U16_ADDRESS]] to ptr addrspace(1)
; IR: [[U16_LOAD:%.+]] = load i16, ptr addrspace(1) [[U16_POINTER]], align 2
; IR: [[U16_VALUE:%.+]] = zext i16 [[U16_LOAD]] to i32
	s_load_u16 s6, s[0:1], 0x3
; IR: xor i32 [[U16_VALUE]], -1
	s_not_b32 s7, s6
; IR: [[I16_BASE:%.+]] = and i64 {{%.+}}, -2
; IR: [[I16_ADDRESS:%.+]] = add i64 [[I16_BASE]], 2
; IR: [[I16_POINTER:%.+]] = inttoptr i64 [[I16_ADDRESS]] to ptr addrspace(1)
; IR: [[I16_LOAD:%.+]] = load i16, ptr addrspace(1) [[I16_POINTER]], align 2
; IR: [[I16_VALUE:%.+]] = sext i16 [[I16_LOAD]] to i32
	s_load_i16 s8, s[0:1], 0x3
; IR: xor i32 [[I16_VALUE]], -1
	s_not_b32 s9, s8
; IR: ret void
	s_endpgm

; SCALE_OFFSET multiplies the SGPR element index by the data size, which is one
; byte for u8/i8 and two for u16/i16. The scale tracks the data size, not the
; signedness.

	.globl	smem_narrow_scale_offset
	.p2align	8
	.type	smem_narrow_scale_offset,@function
; IR-LABEL: define amdgpu_kernel void @smem_narrow_scale_offset(
smem_narrow_scale_offset:
	s_mov_b32 s8, 0x13
; A one-byte scale is the identity, so the SGPR offset is added unscaled.
; IR: [[SCU8_ADDRESS:%.+]] = add i64 {{%.+}}, 0
; IR: [[SCU8_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR-NOT: mul i64
; IR: [[SCU8_SUM:%.+]] = add i64 [[SCU8_ADDRESS]], [[SCU8_SOFFSET]]
; IR: [[SCU8_POINTER:%.+]] = inttoptr i64 [[SCU8_SUM]] to ptr addrspace(1)
; IR: load i8, ptr addrspace(1) [[SCU8_POINTER]], align 1
	s_load_u8 s2, s[0:1], s8 scale_offset
; IR: [[SCI8_ADDRESS:%.+]] = add i64 {{%.+}}, 0
; IR: [[SCI8_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR-NOT: mul i64
; IR: [[SCI8_SUM:%.+]] = add i64 [[SCI8_ADDRESS]], [[SCI8_SOFFSET]]
; IR: [[SCI8_POINTER:%.+]] = inttoptr i64 [[SCI8_SUM]] to ptr addrspace(1)
; IR: [[SCI8_LOAD:%.+]] = load i8, ptr addrspace(1) [[SCI8_POINTER]], align 1
; IR: sext i8 [[SCI8_LOAD]] to i32
	s_load_i8 s3, s[0:1], s8 scale_offset
; IR: [[SCU16_BASE:%.+]] = and i64 {{%.+}}, -2
; IR: [[SCU16_ADDRESS:%.+]] = add i64 [[SCU16_BASE]], 0
; IR: [[SCU16_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR: [[SCU16_SCALED:%.+]] = mul i64 [[SCU16_SOFFSET]], 2
; IR: [[SCU16_DWORD:%.+]] = and i64 [[SCU16_SCALED]], -2
; IR: [[SCU16_SUM:%.+]] = add i64 [[SCU16_ADDRESS]], [[SCU16_DWORD]]
; IR: [[SCU16_POINTER:%.+]] = inttoptr i64 [[SCU16_SUM]] to ptr addrspace(1)
; IR: load i16, ptr addrspace(1) [[SCU16_POINTER]], align 2
	s_load_u16 s4, s[0:1], s8 scale_offset
; IR: [[SCI16_BASE:%.+]] = and i64 {{%.+}}, -2
; IR: [[SCI16_ADDRESS:%.+]] = add i64 [[SCI16_BASE]], 0
; IR: [[SCI16_SOFFSET:%.+]] = zext i32 {{.+}} to i64
; IR: [[SCI16_SCALED:%.+]] = mul i64 [[SCI16_SOFFSET]], 2
; IR: [[SCI16_DWORD:%.+]] = and i64 [[SCI16_SCALED]], -2
; IR: [[SCI16_SUM:%.+]] = add i64 [[SCI16_ADDRESS]], [[SCI16_DWORD]]
; IR: [[SCI16_POINTER:%.+]] = inttoptr i64 [[SCI16_SUM]] to ptr addrspace(1)
; IR: [[SCI16_LOAD:%.+]] = load i16, ptr addrspace(1) [[SCI16_POINTER]], align 2
; IR: sext i16 [[SCI16_LOAD]] to i32
	s_load_i16 s5, s[0:1], s8 scale_offset
; IR: ret void
	s_endpgm

	.globl	smem_cache_policy
	.p2align	8
	.type	smem_cache_policy,@function
smem_cache_policy:
; REFUSE: scalar load cache-policy modifiers other than SCALE_OFFSET are not supported
	s_load_b32 s2, s[0:1], 0x0 scope:SCOPE_SYS
	s_endpgm

	.globl	smem_buffer_load
	.p2align	8
	.type	smem_buffer_load,@function
smem_buffer_load:
; REFUSE: unsupported scalar memory operation
	s_buffer_load_b32 s4, s[0:3], 0x0
	s_endpgm

	.globl	smem_negative_offset
	.p2align	8
	.type	smem_negative_offset,@function
smem_negative_offset:
; REFUSE: negative scalar load offsets are not supported
	s_load_b32 s2, s[0:1], -4
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_loads
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 9
	.end_amdhsa_kernel
	.amdhsa_kernel smem_wide_loads
		.amdhsa_kernarg_size 128
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 32
	.end_amdhsa_kernel
	.amdhsa_kernel smem_wide_overlap
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.amdhsa_kernel smem_register_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel smem_wide_register_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 9
	.end_amdhsa_kernel
	.amdhsa_kernel smem_soffset_overlap
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 7
	.end_amdhsa_kernel
	.amdhsa_kernel smem_scale_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 32
	.end_amdhsa_kernel
	.amdhsa_kernel smem_narrow_loads
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 10
	.end_amdhsa_kernel
	.amdhsa_kernel smem_narrow_scale_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 9
	.end_amdhsa_kernel
	.amdhsa_kernel smem_cache_policy
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.amdhsa_kernel smem_buffer_load
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel smem_negative_offset
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_loads
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .symbol:         smem_loads.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .max_flat_workgroup_size: 1024
    .name:           smem_wide_loads
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .symbol:         smem_wide_loads.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_wide_overlap
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         smem_wide_overlap.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_register_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         smem_register_offset.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_wide_register_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .symbol:         smem_wide_register_offset.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_soffset_overlap
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         smem_soffset_overlap.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_scale_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .symbol:         smem_scale_offset.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_narrow_loads
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         smem_narrow_loads.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_narrow_scale_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .symbol:         smem_narrow_scale_offset.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_cache_policy
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         smem_cache_policy.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_buffer_load
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         smem_buffer_load.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           smem_negative_offset
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         smem_negative_offset.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
