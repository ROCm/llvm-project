; REQUIRES: comgr-has-hotswap-transpile

; The 32-bit wave-mask opcodes exist from gfx10 on and describe a 32-lane wave,
; so this fixture is a wave32 gfx1250 kernel.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=quad_kernel \
; RUN:   | %FileCheck %s --check-prefix=QUAD
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=saveexec_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAVE
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=wrexec_kernel \
; RUN:   | %FileCheck %s --check-prefix=WREXEC

; A 64-bit mask names lanes a 32-lane wave does not have, so neither combining
; one with EXEC nor writing one to EXEC is lifted.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=wide_combine_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WIDE-COMBINE
; WIDE-COMBINE: unsupported-instruction-form: s_and_saveexec_b64
; WIDE-COMBINE-SAME: combines a 64-bit mask with EXEC, but the source wave is 32 bits wide and EXEC holds 32 bits

; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=wide_exec_write_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WIDE-WRITE
; WIDE-WRITE: unsupported-instruction-form: s_mov_b64
; WIDE-WRITE-SAME: writes a 64-bit mask to EXEC, but the source wave is 32 bits wide and EXEC holds 32 bits

; s_rfe_i64 has no lowering, and SOP1 refuses an opcode it does not lift rather
; than letting it through unlowered.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: s_rfe_i64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	quad_kernel
	.p2align	8
	.type	quad_kernel,@function
; QUAD-LABEL: define amdgpu_kernel void @quad_kernel(
quad_kernel:
; A named value to start from, so the checks below pin what each opcode read.
; QUAD: [[SRC:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s1, s1
; s_quadmask folds each group of four bits onto the lowest bit of the group,
; keeps only those bits, and then halves the distance between them until they
; are adjacent: 0x11111111, then two bits per byte, four per halfword, eight in
; the low byte.
; QUAD: [[Q_SHIFTED:%.+]] = lshr i32 [[SRC]], 2
; QUAD: [[Q_PAIRS:%.+]] = or i32 [[SRC]], [[Q_SHIFTED]]
; QUAD: [[Q_PAIRS_SHIFTED:%.+]] = lshr i32 [[Q_PAIRS]], 1
; QUAD: [[Q_ANY:%.+]] = or i32 [[Q_PAIRS]], [[Q_PAIRS_SHIFTED]]
; QUAD: [[Q_LOW:%.+]] = and i32 [[Q_ANY]], 286331153
; QUAD: [[Q_BY3:%.+]] = lshr i32 [[Q_LOW]], 3
; QUAD: [[Q_JOIN2:%.+]] = or i32 [[Q_LOW]], [[Q_BY3]]
; QUAD: [[Q_KEEP2:%.+]] = and i32 [[Q_JOIN2]], 50529027
; QUAD: [[Q_BY6:%.+]] = lshr i32 [[Q_KEEP2]], 6
; QUAD: [[Q_JOIN4:%.+]] = or i32 [[Q_KEEP2]], [[Q_BY6]]
; QUAD: [[Q_KEEP4:%.+]] = and i32 [[Q_JOIN4]], 983055
; QUAD: [[Q_BY12:%.+]] = lshr i32 [[Q_KEEP4]], 12
; QUAD: [[Q_JOIN8:%.+]] = or i32 [[Q_KEEP4]], [[Q_BY12]]
; QUAD: [[QUADMASK:%.+]] = and i32 [[Q_JOIN8]], 255
; QUAD: icmp ne i32 [[QUADMASK]], 0
	s_quadmask_b32 s2, s1
; s_wqm stops at the same one-bit-per-group value and spreads it back over the
; whole group instead of packing it.
; QUAD: [[W_SHIFTED:%.+]] = lshr i32 [[QUADMASK]], 2
; QUAD: [[W_PAIRS:%.+]] = or i32 [[QUADMASK]], [[W_SHIFTED]]
; QUAD: [[W_PAIRS_SHIFTED:%.+]] = lshr i32 [[W_PAIRS]], 1
; QUAD: [[W_ANY:%.+]] = or i32 [[W_PAIRS]], [[W_PAIRS_SHIFTED]]
; QUAD: [[W_LOW:%.+]] = and i32 [[W_ANY]], 286331153
; QUAD: [[W_TWO:%.+]] = shl i32 [[W_LOW]], 1
; QUAD: [[W_HALF:%.+]] = or i32 [[W_LOW]], [[W_TWO]]
; QUAD: [[W_FOUR:%.+]] = shl i32 [[W_HALF]], 2
; QUAD: [[WQM:%.+]] = or i32 [[W_HALF]], [[W_FOUR]]
; QUAD: icmp ne i32 [[WQM]], 0
	s_wqm_b32 s3, s2
; QUAD: xor i32 [[WQM]], -1
	s_not_b32 s4, s3
	s_endpgm

	.globl	saveexec_kernel
	.p2align	8
	.type	saveexec_kernel,@function
; SAVE-LABEL: define amdgpu_kernel void @saveexec_kernel(
saveexec_kernel:
; SAVE: [[SRC:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s1, s1
; Move EXEC off the all-ones mask the kernel starts with, so a check naming it
; cannot pass on a constant that happens to be there.
; SAVE: [[EXEC0:%.+]] = xor i32 [[SRC]], -1
	s_not_b32 s0, s1
	s_mov_b32 exec_lo, s0
; Every opcode here leaves the EXEC it replaced in its scalar destination,
; which the s_not after it reads back.
; SAVE: [[EXEC1:%.+]] = and i32 [[SRC]], [[EXEC0]]
; SAVE: icmp ne i32 [[EXEC1]], 0
	s_and_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC0]], -1
	s_not_b32 s3, s2
; SAVE: [[EXEC2:%.+]] = or i32 [[SRC]], [[EXEC1]]
; SAVE: icmp ne i32 [[EXEC2]], 0
	s_or_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC1]], -1
	s_not_b32 s3, s2
; SAVE: [[EXEC3:%.+]] = xor i32 [[SRC]], [[EXEC2]]
; SAVE: icmp ne i32 [[EXEC3]], 0
	s_xor_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC2]], -1
	s_not_b32 s3, s2
; SAVE: [[NAND_AND:%.+]] = and i32 [[SRC]], [[EXEC3]]
; SAVE: [[EXEC4:%.+]] = xor i32 [[NAND_AND]], -1
; SAVE: icmp ne i32 [[EXEC4]], 0
	s_nand_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC3]], -1
	s_not_b32 s3, s2
; SAVE: [[NOR_OR:%.+]] = or i32 [[SRC]], [[EXEC4]]
; SAVE: [[EXEC5:%.+]] = xor i32 [[NOR_OR]], -1
; SAVE: icmp ne i32 [[EXEC5]], 0
	s_nor_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC4]], -1
	s_not_b32 s3, s2
; SAVE: [[XNOR_XOR:%.+]] = xor i32 [[SRC]], [[EXEC5]]
; SAVE: [[EXEC6:%.+]] = xor i32 [[XNOR_XOR]], -1
; SAVE: icmp ne i32 [[EXEC6]], 0
	s_xnor_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC5]], -1
	s_not_b32 s3, s2
; The not0 forms complement the scalar source and the not1 forms complement
; EXEC, which is what tells the two apart.
; SAVE: [[NOT_SRC1:%.+]] = xor i32 [[SRC]], -1
; SAVE: [[EXEC7:%.+]] = and i32 [[NOT_SRC1]], [[EXEC6]]
; SAVE: icmp ne i32 [[EXEC7]], 0
	s_and_not0_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC6]], -1
	s_not_b32 s3, s2
; SAVE: [[NOT_SRC2:%.+]] = xor i32 [[SRC]], -1
; SAVE: [[EXEC8:%.+]] = or i32 [[NOT_SRC2]], [[EXEC7]]
; SAVE: icmp ne i32 [[EXEC8]], 0
	s_or_not0_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC7]], -1
	s_not_b32 s3, s2
; SAVE: [[NOT_EXEC1:%.+]] = xor i32 [[EXEC8]], -1
; SAVE: [[EXEC9:%.+]] = and i32 [[SRC]], [[NOT_EXEC1]]
; SAVE: icmp ne i32 [[EXEC9]], 0
	s_and_not1_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC8]], -1
	s_not_b32 s3, s2
; SAVE: [[NOT_EXEC2:%.+]] = xor i32 [[EXEC9]], -1
; SAVE: [[EXEC10:%.+]] = or i32 [[SRC]], [[NOT_EXEC2]]
; SAVE: icmp ne i32 [[EXEC10]], 0
	s_or_not1_saveexec_b32 s2, s1
; SAVE: xor i32 [[EXEC9]], -1
	s_not_b32 s3, s2
	s_endpgm

	.globl	wrexec_kernel
	.p2align	8
	.type	wrexec_kernel,@function
; WREXEC-LABEL: define amdgpu_kernel void @wrexec_kernel(
wrexec_kernel:
; WREXEC: [[SRC:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s1, s1
; WREXEC: [[EXEC0:%.+]] = xor i32 [[SRC]], -1
	s_not_b32 s0, s1
	s_mov_b32 exec_lo, s0
; A wrexec opcode leaves the mask it computed in its scalar destination, not
; the EXEC it replaced.
; WREXEC: [[NOT_SRC:%.+]] = xor i32 [[SRC]], -1
; WREXEC: [[EXEC1:%.+]] = and i32 [[NOT_SRC]], [[EXEC0]]
; WREXEC: icmp ne i32 [[EXEC1]], 0
	s_and_not0_wrexec_b32 s2, s1
; WREXEC: xor i32 [[EXEC1]], -1
	s_not_b32 s3, s2
; WREXEC: [[NOT_EXEC:%.+]] = xor i32 [[EXEC1]], -1
; WREXEC: [[EXEC2:%.+]] = and i32 [[SRC]], [[NOT_EXEC]]
; WREXEC: icmp ne i32 [[EXEC2]], 0
	s_and_not1_wrexec_b32 s2, s1
; WREXEC: xor i32 [[EXEC2]], -1
	s_not_b32 s3, s2
	s_endpgm

	.globl	wide_combine_kernel
	.p2align	8
	.type	wide_combine_kernel,@function
wide_combine_kernel:
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_endpgm

	.globl	wide_exec_write_kernel
	.p2align	8
	.type	wide_exec_write_kernel,@function
wide_exec_write_kernel:
	s_mov_b64 exec, s[0:1]
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
	s_rfe_i64 s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel quad_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
	.end_amdhsa_kernel
	.amdhsa_kernel saveexec_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
	.end_amdhsa_kernel
	.amdhsa_kernel wrexec_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 8
	.end_amdhsa_kernel
	.amdhsa_kernel wide_combine_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.amdhsa_kernel wide_exec_write_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
    .name:           quad_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         quad_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           saveexec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         saveexec_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wrexec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         wrexec_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wide_combine_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         wide_combine_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wide_exec_write_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         wide_exec_write_kernel.kd
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
