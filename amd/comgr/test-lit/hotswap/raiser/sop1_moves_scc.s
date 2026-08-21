; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=sop1_moves_kernel | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @sop1_moves_kernel(

; A register nothing reads again holds a dead store, so the kernels below read
; each destination back through a bit reversal to keep the value in the lifted
; IR, and the checks look at what the reversal is handed.

; A destination outside the general-purpose registers reaches its own slot.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=special_dst_kernel \
; RUN:   | %FileCheck %s --check-prefix=SPECIAL

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=literal_kernel \
; RUN:   | %FileCheck %s --check-prefix=LITERAL

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=cmov_undef_kernel \
; RUN:   | %FileCheck %s --check-prefix=CMOVUNDEF

; A SOP1 opcode the handler does not lift is refused, not mislowered.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: s_rfe_b64

; A register the raiser does not model is refused when the move writes it.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=bad_dst_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BADDST
; BADDST:      unsupported-instruction-form: s_mov_b32
; BADDST-SAME: register-decode: unsupported register 'XNACK_MASK_LO'

; And when the move reads it.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=bad_src_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BADSRC
; BADSRC:      unsupported-instruction-form: s_mov_b32
; BADSRC-SAME: register-decode: unsupported register 'XNACK_MASK_LO'

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	sop1_moves_kernel
	.p2align	8
	.type	sop1_moves_kernel,@function
sop1_moves_kernel:
; The value s_cmov_b32 has to preserve when SCC is clear.
	s_mov_b32 s6, 7

; An SGPR pair is two dword slots, so the 64-bit read joins them back up.
; CHECK: [[MOV64:%.+]] = or i64 {{.+}}, {{.+}}
	s_mov_b64 s[0:1], -1
; CHECK: [[BREV64:%.+]] = call i64 @llvm.bitreverse.i64(i64 [[MOV64]])
	s_brev_b64 s[2:3], s[0:1]

; CHECK: [[NOT32:%.+]] = xor i32 {{.+}}, -1
; The 32-bit spelling compares the 32-bit result.
; CHECK: [[SCC32:%.+]] = icmp ne i32 [[NOT32]], 0
	s_not_b32 s4, s2
; s_brev_b32 leaves SCC alone, so the select below still reads the bit
; s_not_b32 wrote.
; CHECK: [[BREV32:%.+]] = call i32 @llvm.bitreverse.i32(i32 [[NOT32]])
	s_brev_b32 s5, s4
; CHECK: select i1 [[SCC32]], i32 [[BREV32]], i32 7
	s_cmov_b32 s6, s5

; CHECK: [[NOT64:%.+]] = xor i64 {{.+}}, -1
; The 64-bit spelling compares the full 64-bit result.
; CHECK: [[SCC64:%.+]] = icmp ne i64 [[NOT64]], 0
	s_not_b64 s[8:9], s[2:3]
; s_mov_b64 leaves SCC alone as well, and the value it moves is what the
; select below preserves.
	s_mov_b64 s[10:11], 3
; CHECK: [[TAKE64:%.+]] = or i64 {{.+}}, {{.+}}
; CHECK: [[KEEPLO:%.+]] = zext i32 3 to i64
; CHECK: [[KEEPHI:%.+]] = zext i32 0 to i64
; CHECK: [[KEEPSHL:%.+]] = shl i64 [[KEEPHI]], 32
; CHECK: [[KEEP64:%.+]] = or i64 [[KEEPLO]], [[KEEPSHL]]
; CHECK: select i1 [[SCC64]], i64 [[TAKE64]], i64 [[KEEP64]]
	s_cmov_b64 s[10:11], s[8:9]

; CHECK: ret void
	s_endpgm

	.globl	special_dst_kernel
	.p2align	8
	.type	special_dst_kernel,@function
special_dst_kernel:
; SPECIAL-LABEL: define amdgpu_kernel void @special_dst_kernel(
; s0 is preloaded with the workgroup index, and reading M0 back hands the same
; value to the bit reversal.
; SPECIAL: [[WGX:%.+]] = call i32 @llvm.amdgcn.workgroup.id.x()
	s_mov_b32 m0, s0
	s_mov_b32 s2, m0
; SPECIAL: call i32 @llvm.bitreverse.i32(i32 [[WGX]])
	s_brev_b32 s3, s2

; EXEC is stored at the wave width, and reading it back splits it into the two
; dword halves an SGPR pair holds.
	s_mov_b64 exec, -1
	s_mov_b64 s[4:5], exec
; SPECIAL: [[EXECLO:%.+]] = trunc i64 -1 to i32
; SPECIAL: [[EXECSHR:%.+]] = lshr i64 -1, 32
; SPECIAL: [[EXECHI:%.+]] = trunc i64 [[EXECSHR]] to i32
	s_brev_b64 s[6:7], s[4:5]

; A VCC write is a per-lane bit, and reading VCC back reassembles the wave
; mask with a ballot.
	s_mov_b32 vcc_lo, s0
	s_mov_b32 s8, vcc_lo
; SPECIAL: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(
; SPECIAL: [[VCCLO:%.+]] = trunc i64 [[BALLOT]] to i32
; SPECIAL: call i32 @llvm.bitreverse.i32(i32 [[VCCLO]])
	s_brev_b32 s9, s8

; s_not_b64 reaches EXEC as well, and its SCC write is the 64-bit comparison.
	s_not_b64 exec, exec
; SPECIAL: [[NOTEXEC:%.+]] = xor i64 {{.+}}, -1
; SPECIAL: icmp ne i64 [[NOTEXEC]], 0
	s_mov_b64 s[10:11], exec
; SPECIAL: trunc i64 [[NOTEXEC]] to i32
	s_brev_b64 s[12:13], s[10:11]
; SPECIAL: ret void
	s_endpgm

	.globl	literal_kernel
	.p2align	8
	.type	literal_kernel,@function
literal_kernel:
; LITERAL-LABEL: define amdgpu_kernel void @literal_kernel(
; A 32-bit literal reaches the destination unchanged rather than truncated or
; re-signed: 0x12345678 is 305419896.
	s_mov_b32 s0, 0x12345678
; LITERAL: call i32 @llvm.bitreverse.i32(i32 305419896)
	s_brev_b32 s1, s0
; The 64-bit spelling of the same literal zero-extends into the high dword.
	s_mov_b64 s[2:3], 0x12345678
; LITERAL: [[LO:%.+]] = zext i32 305419896 to i64
; LITERAL: [[HI:%.+]] = zext i32 0 to i64
; LITERAL: [[SHL:%.+]] = shl i64 [[HI]], 32
; LITERAL: [[JOIN:%.+]] = or i64 [[LO]], [[SHL]]
; LITERAL: call i64 @llvm.bitreverse.i64(i64 [[JOIN]])
	s_brev_b64 s[4:5], s[2:3]
; LITERAL: ret void
	s_endpgm

	.globl	cmov_undef_kernel
	.p2align	8
	.type	cmov_undef_kernel,@function
cmov_undef_kernel:
; CMOVUNDEF-LABEL: define amdgpu_kernel void @cmov_undef_kernel(
; s6 is never written before the move, so the value s_cmov_b32 preserves when
; SCC is clear is the undefined content of that register.
	s_not_b32 s2, s2
; CMOVUNDEF: [[NOT32:%.+]] = xor i32 undef, -1
; CMOVUNDEF: [[SCC32:%.+]] = icmp ne i32 [[NOT32]], 0
	s_cmov_b32 s6, s2
; CMOVUNDEF: select i1 [[SCC32]], i32 [[NOT32]], i32 undef
	s_brev_b32 s7, s6

; The 64-bit spelling reads both undefined dwords of the pair back.
	s_not_b64 s[10:11], s[10:11]
; CMOVUNDEF: [[NOT64:%.+]] = xor i64 {{.+}}, -1
; CMOVUNDEF: [[SCC64:%.+]] = icmp ne i64 [[NOT64]], 0
	s_cmov_b64 s[12:13], s[10:11]
; CMOVUNDEF: [[KEEPLO:%.+]] = zext i32 undef to i64
; CMOVUNDEF: [[KEEPHI:%.+]] = zext i32 undef to i64
; CMOVUNDEF: [[KEEPSHL:%.+]] = shl i64 [[KEEPHI]], 32
; CMOVUNDEF: [[KEEP:%.+]] = or i64 [[KEEPLO]], [[KEEPSHL]]
; CMOVUNDEF: select i1 [[SCC64]], i64 {{.+}}, i64 [[KEEP]]
	s_brev_b64 s[14:15], s[12:13]
; CMOVUNDEF: ret void
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
	s_rfe_b64 s[0:1]
	s_endpgm

	.globl	bad_dst_kernel
	.p2align	8
	.type	bad_dst_kernel,@function
bad_dst_kernel:
	s_mov_b32 xnack_mask_lo, s0
	s_endpgm

	.globl	bad_src_kernel
	.p2align	8
	.type	bad_src_kernel,@function
bad_src_kernel:
	s_mov_b32 s0, xnack_mask_lo
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sop1_moves_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel special_dst_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel literal_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel cmov_undef_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel bad_dst_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel bad_src_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
    .name:           sop1_moves_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         sop1_moves_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           special_dst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .symbol:         special_dst_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           literal_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         literal_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cmov_undef_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         cmov_undef_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
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
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           bad_dst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         bad_dst_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           bad_src_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         bad_src_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
