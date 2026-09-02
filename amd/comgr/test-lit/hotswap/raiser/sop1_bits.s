; REQUIRES: comgr-has-hotswap-transpile

; gfx1250 names these opcodes after what they do rather than after the bit they
; find, and it carries every one of them, so the whole family fits one fixture.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=search_kernel,search_nothing_kernel,sext_kernel,bitset_kernel \
; RUN:   --emit-ir=bitreplicate_kernel,abs_kernel,bcnt_kernel,scc_kernel \
; RUN:   | %FileCheck %s

; s_rfe_i64 has no lowering, and SOP1 refuses an opcode it does not lift rather
; than letting it through unlowered.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: s_rfe_i64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

; Every source is seeded with a literal, so the value each opcode searched sits
; in the IR next to the search, and an s_cvt_f32_u32 after each one reads the
; destination back so that the result reaches the IR as well.
	.globl	search_kernel
	.p2align	8
	.type	search_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @search_kernel(
search_kernel:
	s_mov_b32 s0, 0x10000
; CHECK: [[CTZ:%.+]] = call i32 @llvm.cttz.i32(i32 65536, i1 false)
; CHECK: [[CTZ_ANY:%.+]] = icmp ne i32 65536, 0
; CHECK: [[CTZ_POS:%.+]] = select i1 [[CTZ_ANY]], i32 [[CTZ]], i32 -1
	s_ctz_i32_b32 s1, s0
; CHECK: uitofp i32 [[CTZ_POS]] to float
	s_cvt_f32_u32 s30, s1
; s_clz counts from the other end of the same operand.
; CHECK: [[CLZ:%.+]] = call i32 @llvm.ctlz.i32(i32 65536, i1 false)
; CHECK: [[CLZ_ANY:%.+]] = icmp ne i32 65536, 0
; CHECK: [[CLZ_POS:%.+]] = select i1 [[CLZ_ANY]], i32 [[CLZ]], i32 -1
	s_clz_i32_u32 s2, s0
; CHECK: uitofp i32 [[CLZ_POS]] to float
	s_cvt_f32_u32 s30, s2
; s_cls stops at the first bit unlike the sign rather than at the first set
; bit, so a negative source is complemented before the leading zeros of what is
; left are counted.
	s_mov_b32 s3, 0xffff3333
; CHECK: [[SIGN:%.+]] = ashr i32 -52429, 31
; CHECK: [[CLS_IN:%.+]] = xor i32 -52429, [[SIGN]]
; CHECK: [[CLS:%.+]] = call i32 @llvm.ctlz.i32(i32 [[CLS_IN]], i1 false)
; CHECK: [[CLS_ANY:%.+]] = icmp ne i32 [[CLS_IN]], 0
; CHECK: [[CLS_POS:%.+]] = select i1 [[CLS_ANY]], i32 [[CLS]], i32 -1
	s_cls_i32 s4, s3
; CHECK: uitofp i32 [[CLS_POS]] to float
	s_cvt_f32_u32 s30, s4
; A non-negative source goes into the count as it stands.
	s_mov_b32 s5, 0xcccc
; CHECK: [[PSIGN:%.+]] = ashr i32 52428, 31
; CHECK: [[PCLS_IN:%.+]] = xor i32 52428, [[PSIGN]]
; CHECK: [[PCLS:%.+]] = call i32 @llvm.ctlz.i32(i32 [[PCLS_IN]], i1 false)
; CHECK: [[PCLS_ANY:%.+]] = icmp ne i32 [[PCLS_IN]], 0
; CHECK: [[PCLS_POS:%.+]] = select i1 [[PCLS_ANY]], i32 [[PCLS]], i32 -1
	s_cls_i32 s6, s5
; CHECK: uitofp i32 [[PCLS_POS]] to float
	s_cvt_f32_u32 s30, s6
; The 64-bit forms search a register pair and still write a single dword, so
; the count is narrowed before it reaches the destination.
	s_mov_b32 s8, 0
	s_mov_b32 s9, 0x10000
; CHECK: [[CTZ64_IN:%.+]] = or i64 {{.+}}, {{.+}}
; CHECK: [[CTZ64:%.+]] = call i64 @llvm.cttz.i64(i64 [[CTZ64_IN]], i1 false)
; CHECK: [[CTZ64_ANY:%.+]] = icmp ne i64 [[CTZ64_IN]], 0
; CHECK: [[CTZ64_LO:%.+]] = trunc i64 [[CTZ64]] to i32
; CHECK: [[CTZ64_POS:%.+]] = select i1 [[CTZ64_ANY]], i32 [[CTZ64_LO]], i32 -1
	s_ctz_i32_b64 s10, s[8:9]
; CHECK: uitofp i32 [[CTZ64_POS]] to float
	s_cvt_f32_u32 s30, s10
; CHECK: [[CLZ64_IN:%.+]] = or i64 {{.+}}, {{.+}}
; CHECK: [[CLZ64:%.+]] = call i64 @llvm.ctlz.i64(i64 [[CLZ64_IN]], i1 false)
; CHECK: [[CLZ64_ANY:%.+]] = icmp ne i64 [[CLZ64_IN]], 0
; CHECK: [[CLZ64_LO:%.+]] = trunc i64 [[CLZ64]] to i32
; CHECK: [[CLZ64_POS:%.+]] = select i1 [[CLZ64_ANY]], i32 [[CLZ64_LO]], i32 -1
	s_clz_i32_u64 s11, s[8:9]
; CHECK: uitofp i32 [[CLZ64_POS]] to float
	s_cvt_f32_u32 s30, s11
; The sign of a pair is its bit 63.
	s_mov_b32 s12, 0
	s_mov_b32 s13, 0xffff3333
; CHECK: [[PAIR:%.+]] = or i64 {{.+}}, {{.+}}
; CHECK: [[SIGN64:%.+]] = ashr i64 [[PAIR]], 63
; CHECK: [[CLS64_IN:%.+]] = xor i64 [[PAIR]], [[SIGN64]]
; CHECK: [[CLS64:%.+]] = call i64 @llvm.ctlz.i64(i64 [[CLS64_IN]], i1 false)
; CHECK: [[CLS64_ANY:%.+]] = icmp ne i64 [[CLS64_IN]], 0
; CHECK: [[CLS64_LO:%.+]] = trunc i64 [[CLS64]] to i32
; CHECK: [[CLS64_POS:%.+]] = select i1 [[CLS64_ANY]], i32 [[CLS64_LO]], i32 -1
	s_cls_i32_i64 s14, s[12:13]
; CHECK: uitofp i32 [[CLS64_POS]] to float
	s_cvt_f32_u32 s30, s14
	s_endpgm

; An input with no bit to find is the one the hardware defines and the count
; intrinsics do not: it answers -1 where they answer the operand width. Every
; search below takes that input, and for s_cls both of the two sources that
; have it.
	.globl	search_nothing_kernel
	.p2align	8
	.type	search_nothing_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @search_nothing_kernel(
search_nothing_kernel:
	s_mov_b32 s0, 0
	s_mov_b32 s1, 0
; CHECK: call i32 @llvm.cttz.i32(i32 0, i1 false)
; CHECK: [[CTZ_ANY:%.+]] = icmp ne i32 0, 0
; CHECK: [[CTZ_POS:%.+]] = select i1 [[CTZ_ANY]], i32 {{.+}}, i32 -1
	s_ctz_i32_b32 s2, s0
; CHECK: uitofp i32 [[CTZ_POS]] to float
	s_cvt_f32_u32 s30, s2
; CHECK: call i32 @llvm.ctlz.i32(i32 0, i1 false)
; CHECK: [[CLZ_ANY:%.+]] = icmp ne i32 0, 0
; CHECK: [[CLZ_POS:%.+]] = select i1 [[CLZ_ANY]], i32 {{.+}}, i32 -1
	s_clz_i32_u32 s3, s0
; CHECK: uitofp i32 [[CLZ_POS]] to float
	s_cvt_f32_u32 s30, s3
; CHECK: [[Z_IN:%.+]] = xor i32 0, {{.+}}
; CHECK: [[Z_ANY:%.+]] = icmp ne i32 [[Z_IN]], 0
; CHECK: [[Z_POS:%.+]] = select i1 [[Z_ANY]], i32 {{.+}}, i32 -1
	s_cls_i32 s4, s0
; CHECK: uitofp i32 [[Z_POS]] to float
	s_cvt_f32_u32 s30, s4
; All ones is as uniformly signed as all zeros, so it too has no bit to find.
	s_mov_b32 s5, -1
; CHECK: [[ONES_IN:%.+]] = xor i32 -1, {{.+}}
; CHECK: [[ONES_ANY:%.+]] = icmp ne i32 [[ONES_IN]], 0
; CHECK: [[ONES_POS:%.+]] = select i1 [[ONES_ANY]], i32 {{.+}}, i32 -1
	s_cls_i32 s6, s5
; CHECK: uitofp i32 [[ONES_POS]] to float
	s_cvt_f32_u32 s30, s6
; CHECK: [[CTZ64_IN:%.+]] = or i64 {{.+}}, {{.+}}
; CHECK: call i64 @llvm.cttz.i64(i64 [[CTZ64_IN]], i1 false)
; CHECK: [[CTZ64_ANY:%.+]] = icmp ne i64 [[CTZ64_IN]], 0
; CHECK: [[CTZ64_POS:%.+]] = select i1 [[CTZ64_ANY]], i32 {{.+}}, i32 -1
	s_ctz_i32_b64 s7, s[0:1]
; CHECK: uitofp i32 [[CTZ64_POS]] to float
	s_cvt_f32_u32 s30, s7
; CHECK: [[CLZ64_IN:%.+]] = or i64 {{.+}}, {{.+}}
; CHECK: call i64 @llvm.ctlz.i64(i64 [[CLZ64_IN]], i1 false)
; CHECK: [[CLZ64_ANY:%.+]] = icmp ne i64 [[CLZ64_IN]], 0
; CHECK: [[CLZ64_POS:%.+]] = select i1 [[CLZ64_ANY]], i32 {{.+}}, i32 -1
	s_clz_i32_u64 s8, s[0:1]
; CHECK: uitofp i32 [[CLZ64_POS]] to float
	s_cvt_f32_u32 s30, s8
; CHECK: [[CLS64_IN:%.+]] = xor i64 {{.+}}, {{.+}}
; CHECK: [[CLS64_ANY:%.+]] = icmp ne i64 [[CLS64_IN]], 0
; CHECK: [[CLS64_POS:%.+]] = select i1 [[CLS64_ANY]], i32 {{.+}}, i32 -1
	s_cls_i32_i64 s9, s[0:1]
; CHECK: uitofp i32 [[CLS64_POS]] to float
	s_cvt_f32_u32 s30, s9
	s_endpgm

	.globl	sext_kernel
	.p2align	8
	.type	sext_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @sext_kernel(
sext_kernel:
; The source has bit 7 and bit 15 set, so the two widths disagree on the sign
; and a check on one of them cannot pass for the other.
	s_mov_b32 s0, 0xff81
; CHECK: [[BYTE:%.+]] = trunc i32 65409 to i8
; CHECK: [[SEXT8:%.+]] = sext i8 [[BYTE]] to i32
	s_sext_i32_i8 s1, s0
; CHECK: uitofp i32 [[SEXT8]] to float
	s_cvt_f32_u32 s30, s1
; CHECK: [[HALF:%.+]] = trunc i32 65409 to i16
; CHECK: [[SEXT16:%.+]] = sext i16 [[HALF]] to i32
	s_sext_i32_i16 s2, s0
; CHECK: uitofp i32 [[SEXT16]] to float
	s_cvt_f32_u32 s30, s2
	s_endpgm

; s_bitset takes the bit index from the low bits of its source and leaves the
; rest of the destination alone, so the destination is an input as well. Each
; destination here is seeded with the value the untouched bits have to keep.
	.globl	bitset_kernel
	.p2align	8
	.type	bitset_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @bitset_kernel(
bitset_kernel:
	s_mov_b32 s0, -1
	s_mov_b32 s1, 5
; CHECK: [[I0:%.+]] = and i32 5, 31
; CHECK: [[B0:%.+]] = shl i32 1, [[I0]]
; CHECK: [[N0:%.+]] = xor i32 [[B0]], -1
; CHECK: [[CLEARED:%.+]] = and i32 -1, [[N0]]
	s_bitset0_b32 s0, s1
; CHECK: uitofp i32 [[CLEARED]] to float
	s_cvt_f32_u32 s30, s0
	s_mov_b32 s2, 0
; CHECK: [[I1:%.+]] = and i32 5, 31
; CHECK: [[B1:%.+]] = shl i32 1, [[I1]]
; CHECK: [[SET:%.+]] = or i32 0, [[B1]]
	s_bitset1_b32 s2, s1
; CHECK: uitofp i32 [[SET]] to float
	s_cvt_f32_u32 s30, s2
; A pair takes six index bits, so index 32 reaches the high half of the
; destination instead of wrapping back to its bit 0.
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0
	s_mov_b32 s3, 32
; CHECK: [[I2:%.+]] = and i32 32, 63
; CHECK: [[I2W:%.+]] = zext i32 [[I2]] to i64
; CHECK: [[B2:%.+]] = shl i64 1, [[I2W]]
; CHECK: [[SET64:%.+]] = or i64 {{.+}}, [[B2]]
; CHECK: [[SET64_LO:%.+]] = trunc i64 [[SET64]] to i32
; CHECK: [[SET64_SHIFTED:%.+]] = lshr i64 [[SET64]], 32
; CHECK: [[SET64_HI:%.+]] = trunc i64 [[SET64_SHIFTED]] to i32
	s_bitset1_b64 s[4:5], s3
; CHECK: uitofp i32 [[SET64_LO]] to float
	s_cvt_f32_u32 s30, s4
; CHECK: uitofp i32 [[SET64_HI]] to float
	s_cvt_f32_u32 s30, s5
	s_mov_b32 s6, -1
	s_mov_b32 s7, -1
; CHECK: [[I3:%.+]] = and i32 32, 63
; CHECK: [[I3W:%.+]] = zext i32 [[I3]] to i64
; CHECK: [[B3:%.+]] = shl i64 1, [[I3W]]
; CHECK: [[N3:%.+]] = xor i64 [[B3]], -1
; CHECK: [[CLEARED64:%.+]] = and i64 {{.+}}, [[N3]]
; CHECK: [[CLEARED64_LO:%.+]] = trunc i64 [[CLEARED64]] to i32
; CHECK: [[CLEARED64_SHIFTED:%.+]] = lshr i64 [[CLEARED64]], 32
; CHECK: [[CLEARED64_HI:%.+]] = trunc i64 [[CLEARED64_SHIFTED]] to i32
	s_bitset0_b64 s[6:7], s3
; CHECK: uitofp i32 [[CLEARED64_LO]] to float
	s_cvt_f32_u32 s30, s6
; CHECK: uitofp i32 [[CLEARED64_HI]] to float
	s_cvt_f32_u32 s30, s7
	s_endpgm

; Each source bit ends up in two neighbouring result bits. The source here has
; one bit at either end, so a step that spread the bits by the wrong distance
; would land them somewhere visibly different.
	.globl	bitreplicate_kernel
	.p2align	8
	.type	bitreplicate_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @bitreplicate_kernel(
bitreplicate_kernel:
	s_mov_b32 s0, 0x80000003
; CHECK: [[WIDE:%.+]] = zext i32 -2147483645 to i64
; CHECK: [[SH16:%.+]] = shl i64 [[WIDE]], 16
; CHECK: [[OR16:%.+]] = or i64 [[WIDE]], [[SH16]]
; CHECK: [[K16:%.+]] = and i64 [[OR16]], 281470681808895
; CHECK: [[SH8:%.+]] = shl i64 [[K16]], 8
; CHECK: [[OR8:%.+]] = or i64 [[K16]], [[SH8]]
; CHECK: [[K8:%.+]] = and i64 [[OR8]], 71777214294589695
; CHECK: [[SH4:%.+]] = shl i64 [[K8]], 4
; CHECK: [[OR4:%.+]] = or i64 [[K8]], [[SH4]]
; CHECK: [[K4:%.+]] = and i64 [[OR4]], 1085102592571150095
; CHECK: [[SH2:%.+]] = shl i64 [[K4]], 2
; CHECK: [[OR2:%.+]] = or i64 [[K4]], [[SH2]]
; CHECK: [[K2:%.+]] = and i64 [[OR2]], 3689348814741910323
; CHECK: [[SH1:%.+]] = shl i64 [[K2]], 1
; CHECK: [[OR1:%.+]] = or i64 [[K2]], [[SH1]]
; CHECK: [[EVEN:%.+]] = and i64 [[OR1]], 6148914691236517205
; CHECK: [[ODD:%.+]] = shl i64 [[EVEN]], 1
; CHECK: [[BOTH:%.+]] = or i64 [[EVEN]], [[ODD]]
; CHECK: [[LO:%.+]] = trunc i64 [[BOTH]] to i32
; CHECK: [[SHIFTED:%.+]] = lshr i64 [[BOTH]], 32
; CHECK: [[HI:%.+]] = trunc i64 [[SHIFTED]] to i32
	s_bitreplicate_b64_b32 s[2:3], s0
; CHECK: uitofp i32 [[LO]] to float
	s_cvt_f32_u32 s30, s2
; CHECK: uitofp i32 [[HI]] to float
	s_cvt_f32_u32 s30, s3
	s_endpgm

	.globl	abs_kernel
	.p2align	8
	.type	abs_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @abs_kernel(
abs_kernel:
; The most negative input has no positive counterpart and the hardware keeps
; it, so the intrinsic is the one that does not make that poison.
	s_mov_b32 s0, -5
; CHECK: [[NEG:%.+]] = call i32 @llvm.abs.i32(i32 -5, i1 false)
	s_abs_i32 s1, s0
; CHECK: uitofp i32 [[NEG]] to float
	s_cvt_f32_u32 s30, s1
	s_mov_b32 s2, 5
; CHECK: [[POS:%.+]] = call i32 @llvm.abs.i32(i32 5, i1 false)
; CHECK: [[FLAG:%.+]] = icmp ne i32 [[POS]], 0
	s_abs_i32 s3, s2
; CHECK: uitofp i32 [[POS]] to float
	s_cvt_f32_u32 s30, s3
; s_abs writes SCC, and the move that follows it selects on the bit it wrote.
	s_mov_b32 s4, 7
; CHECK: [[MOVED:%.+]] = select i1 [[FLAG]], i32 [[POS]], i32 7
	s_cmov_b32 s4, s3
; CHECK: uitofp i32 [[MOVED]] to float
	s_cvt_f32_u32 s30, s4
	s_endpgm

	.globl	bcnt_kernel
	.p2align	8
	.type	bcnt_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @bcnt_kernel(
bcnt_kernel:
; Half the bits of the source are set, so a count of ones and a count of zeros
; agree on the value and only the complement in front of one of them tells the
; two lowerings apart.
	s_mov_b32 s0, 0xcccccccc
; CHECK: [[FLIPPED:%.+]] = xor i32 -858993460, -1
; CHECK: [[ZEROS:%.+]] = call i32 @llvm.ctpop.i32(i32 [[FLIPPED]])
	s_bcnt0_i32_b32 s1, s0
; CHECK: uitofp i32 [[ZEROS]] to float
	s_cvt_f32_u32 s30, s1
; CHECK: [[ONES:%.+]] = call i32 @llvm.ctpop.i32(i32 -858993460)
	s_bcnt1_i32_b32 s2, s0
; CHECK: uitofp i32 [[ONES]] to float
	s_cvt_f32_u32 s30, s2
; A pair is counted whole and the count still comes out one dword wide.
	s_mov_b32 s4, 0xcccccccc
	s_mov_b32 s5, 0
; CHECK: [[FLIPPED64:%.+]] = xor i64 {{.+}}, -1
; CHECK: [[ZEROS64:%.+]] = call i64 @llvm.ctpop.i64(i64 [[FLIPPED64]])
; CHECK: [[ZEROS64_LO:%.+]] = trunc i64 [[ZEROS64]] to i32
	s_bcnt0_i32_b64 s6, s[4:5]
; CHECK: uitofp i32 [[ZEROS64_LO]] to float
	s_cvt_f32_u32 s30, s6
; CHECK: [[PAIR:%.+]] = or i64 {{.+}}, {{.+}}
; CHECK: [[ONES64:%.+]] = call i64 @llvm.ctpop.i64(i64 [[PAIR]])
; CHECK: [[ONES64_LO:%.+]] = trunc i64 [[ONES64]] to i32
; CHECK: [[FLAG:%.+]] = icmp ne i32 [[ONES64_LO]], 0
	s_bcnt1_i32_b64 s7, s[4:5]
; CHECK: uitofp i32 [[ONES64_LO]] to float
	s_cvt_f32_u32 s30, s7
; Every s_bcnt writes SCC, and the move that follows selects on the bit the
; last of them wrote.
	s_mov_b32 s8, 7
; CHECK: [[MOVED:%.+]] = select i1 [[FLAG]], i32 [[ONES64_LO]], i32 7
	s_cmov_b32 s8, s7
; CHECK: uitofp i32 [[MOVED]] to float
	s_cvt_f32_u32 s30, s8
	s_endpgm

	.globl	scc_kernel
	.p2align	8
	.type	scc_kernel,@function
; s_not_b32 writes SCC and s_cmov_b32 reads it. The thirteen opcodes here that
; leave SCC alone all run in between, so the select still taking the bit
; s_not_b32 produced is what says none of them touched it on the way.
; CHECK-LABEL: define amdgpu_kernel void @scc_kernel(
scc_kernel:
; The value s_cmov_b32 preserves when SCC is clear.
	s_mov_b32 s2, 7
; CHECK: [[NOT:%.+]] = xor i32 {{.+}}, -1
; CHECK: [[BIT:%.+]] = icmp ne i32 [[NOT]], 0
	s_not_b32 s0, s1
	s_ctz_i32_b32 s3, s0
	s_ctz_i32_b64 s3, s[0:1]
	s_clz_i32_u32 s3, s0
	s_clz_i32_u64 s3, s[0:1]
	s_cls_i32 s3, s0
	s_cls_i32_i64 s3, s[0:1]
	s_sext_i32_i8 s3, s0
	s_sext_i32_i16 s3, s0
	s_bitset0_b32 s3, s0
	s_bitset1_b32 s3, s0
	s_bitset0_b64 s[4:5], s0
	s_bitset1_b64 s[4:5], s0
	s_bitreplicate_b64_b32 s[6:7], s0
; CHECK: [[MOVED:%.+]] = select i1 [[BIT]], i32 {{.+}}, i32 7
	s_cmov_b32 s2, s3
; CHECK: uitofp i32 [[MOVED]] to float
	s_cvt_f32_u32 s30, s2
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
	s_rfe_i64 s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel search_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel search_nothing_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel sext_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel bitset_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel bitreplicate_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel abs_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel bcnt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel scc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 31
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           search_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         search_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           search_nothing_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         search_nothing_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sext_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         sext_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           bitset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         bitset_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           bitreplicate_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         bitreplicate_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           abs_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         abs_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           bcnt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         bcnt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           scc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .symbol:         scc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           rfe_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         rfe_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
