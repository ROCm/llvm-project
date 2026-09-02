; REQUIRES: comgr-has-hotswap-transpile

; The 32-bit wave-mask opcodes exist from gfx10 on and describe a 32-lane wave,
; so this fixture is a wave32 gfx1250 kernel.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=quad_kernel,saveexec_kernel,wrexec_kernel | %FileCheck %s

; A 64-bit mask names lanes a 32-lane wave does not have, so neither combining
; one with EXEC nor writing one to EXEC is lifted. s_rfe_i64 has no lowering at
; all, and SOP1 refuses an opcode it does not lift rather than letting it
; through unlowered.
; RUN: not %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=wide_combine_kernel,wide_exec_write_kernel,rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	quad_kernel
	.p2align	8
	.type	quad_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @quad_kernel(
quad_kernel:
; A named value to start from, so the checks below pin what each opcode read.
; CHECK: [[SRC:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s1, s1
; s_quadmask folds each group of four bits onto the lowest bit of the group,
; keeps only those bits, and then halves the distance between them until they
; are adjacent: 0x11111111, then two bits per byte, four per halfword, eight in
; the low byte.
; CHECK: [[Q_SHIFTED:%.+]] = lshr i32 [[SRC]], 2
; CHECK: [[Q_PAIRS:%.+]] = or i32 [[SRC]], [[Q_SHIFTED]]
; CHECK: [[Q_PAIRS_SHIFTED:%.+]] = lshr i32 [[Q_PAIRS]], 1
; CHECK: [[Q_ANY:%.+]] = or i32 [[Q_PAIRS]], [[Q_PAIRS_SHIFTED]]
; CHECK: [[Q_LOW:%.+]] = and i32 [[Q_ANY]], 286331153
; CHECK: [[Q_BY3:%.+]] = lshr i32 [[Q_LOW]], 3
; CHECK: [[Q_JOIN2:%.+]] = or i32 [[Q_LOW]], [[Q_BY3]]
; CHECK: [[Q_KEEP2:%.+]] = and i32 [[Q_JOIN2]], 50529027
; CHECK: [[Q_BY6:%.+]] = lshr i32 [[Q_KEEP2]], 6
; CHECK: [[Q_JOIN4:%.+]] = or i32 [[Q_KEEP2]], [[Q_BY6]]
; CHECK: [[Q_KEEP4:%.+]] = and i32 [[Q_JOIN4]], 983055
; CHECK: [[Q_BY12:%.+]] = lshr i32 [[Q_KEEP4]], 12
; CHECK: [[Q_JOIN8:%.+]] = or i32 [[Q_KEEP4]], [[Q_BY12]]
; CHECK: [[QUADMASK:%.+]] = and i32 [[Q_JOIN8]], 255
; CHECK: icmp ne i32 [[QUADMASK]], 0
	s_quadmask_b32 s2, s1
; s_wqm stops at the same one-bit-per-group value and spreads it back over the
; whole group instead of packing it.
; CHECK: [[W_SHIFTED:%.+]] = lshr i32 [[QUADMASK]], 2
; CHECK: [[W_PAIRS:%.+]] = or i32 [[QUADMASK]], [[W_SHIFTED]]
; CHECK: [[W_PAIRS_SHIFTED:%.+]] = lshr i32 [[W_PAIRS]], 1
; CHECK: [[W_ANY:%.+]] = or i32 [[W_PAIRS]], [[W_PAIRS_SHIFTED]]
; CHECK: [[W_LOW:%.+]] = and i32 [[W_ANY]], 286331153
; CHECK: [[W_TWO:%.+]] = shl i32 [[W_LOW]], 1
; CHECK: [[W_HALF:%.+]] = or i32 [[W_LOW]], [[W_TWO]]
; CHECK: [[W_FOUR:%.+]] = shl i32 [[W_HALF]], 2
; CHECK: [[WQM:%.+]] = or i32 [[W_HALF]], [[W_FOUR]]
; CHECK: icmp ne i32 [[WQM]], 0
	s_wqm_b32 s3, s2
; CHECK: xor i32 [[WQM]], -1
	s_not_b32 s4, s3
	s_endpgm

	.globl	saveexec_kernel
	.p2align	8
	.type	saveexec_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @saveexec_kernel(
saveexec_kernel:
; CHECK: [[SRC:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s1, s1
; Move EXEC off the all-ones mask the kernel starts with, so a check naming it
; cannot pass on a constant that happens to be there.
; CHECK: [[EXEC0:%.+]] = xor i32 [[SRC]], -1
	s_not_b32 s0, s1
	s_mov_b32 exec_lo, s0
; Every opcode here leaves the EXEC it replaced in its scalar destination,
; which the s_not after it reads back.
; CHECK: [[EXEC1:%.+]] = and i32 [[SRC]], [[EXEC0]]
; CHECK: icmp ne i32 [[EXEC1]], 0
	s_and_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC0]], -1
	s_not_b32 s3, s2
; CHECK: [[EXEC2:%.+]] = or i32 [[SRC]], [[EXEC1]]
; CHECK: icmp ne i32 [[EXEC2]], 0
	s_or_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC1]], -1
	s_not_b32 s3, s2
; CHECK: [[EXEC3:%.+]] = xor i32 [[SRC]], [[EXEC2]]
; CHECK: icmp ne i32 [[EXEC3]], 0
	s_xor_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC2]], -1
	s_not_b32 s3, s2
; CHECK: [[NAND_AND:%.+]] = and i32 [[SRC]], [[EXEC3]]
; CHECK: [[EXEC4:%.+]] = xor i32 [[NAND_AND]], -1
; CHECK: icmp ne i32 [[EXEC4]], 0
	s_nand_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC3]], -1
	s_not_b32 s3, s2
; CHECK: [[NOR_OR:%.+]] = or i32 [[SRC]], [[EXEC4]]
; CHECK: [[EXEC5:%.+]] = xor i32 [[NOR_OR]], -1
; CHECK: icmp ne i32 [[EXEC5]], 0
	s_nor_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC4]], -1
	s_not_b32 s3, s2
; CHECK: [[XNOR_XOR:%.+]] = xor i32 [[SRC]], [[EXEC5]]
; CHECK: [[EXEC6:%.+]] = xor i32 [[XNOR_XOR]], -1
; CHECK: icmp ne i32 [[EXEC6]], 0
	s_xnor_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC5]], -1
	s_not_b32 s3, s2
; The not0 forms complement the scalar source and the not1 forms complement
; EXEC, which is what tells the two apart.
; CHECK: [[NOT_SRC1:%.+]] = xor i32 [[SRC]], -1
; CHECK: [[EXEC7:%.+]] = and i32 [[NOT_SRC1]], [[EXEC6]]
; CHECK: icmp ne i32 [[EXEC7]], 0
	s_and_not0_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC6]], -1
	s_not_b32 s3, s2
; CHECK: [[NOT_SRC2:%.+]] = xor i32 [[SRC]], -1
; CHECK: [[EXEC8:%.+]] = or i32 [[NOT_SRC2]], [[EXEC7]]
; CHECK: icmp ne i32 [[EXEC8]], 0
	s_or_not0_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC7]], -1
	s_not_b32 s3, s2
; CHECK: [[NOT_EXEC1:%.+]] = xor i32 [[EXEC8]], -1
; CHECK: [[EXEC9:%.+]] = and i32 [[SRC]], [[NOT_EXEC1]]
; CHECK: icmp ne i32 [[EXEC9]], 0
	s_and_not1_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC8]], -1
	s_not_b32 s3, s2
; CHECK: [[NOT_EXEC2:%.+]] = xor i32 [[EXEC9]], -1
; CHECK: [[EXEC10:%.+]] = or i32 [[SRC]], [[NOT_EXEC2]]
; CHECK: icmp ne i32 [[EXEC10]], 0
	s_or_not1_saveexec_b32 s2, s1
; CHECK: xor i32 [[EXEC9]], -1
	s_not_b32 s3, s2
	s_endpgm

	.globl	wrexec_kernel
	.p2align	8
	.type	wrexec_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @wrexec_kernel(
wrexec_kernel:
; CHECK: [[SRC:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s1, s1
; CHECK: [[EXEC0:%.+]] = xor i32 [[SRC]], -1
	s_not_b32 s0, s1
	s_mov_b32 exec_lo, s0
; A wrexec opcode leaves the mask it computed in its scalar destination, not
; the EXEC it replaced.
; CHECK: [[NOT_SRC:%.+]] = xor i32 [[SRC]], -1
; CHECK: [[EXEC1:%.+]] = and i32 [[NOT_SRC]], [[EXEC0]]
; CHECK: icmp ne i32 [[EXEC1]], 0
	s_and_not0_wrexec_b32 s2, s1
; CHECK: xor i32 [[EXEC1]], -1
	s_not_b32 s3, s2
; CHECK: [[NOT_EXEC:%.+]] = xor i32 [[EXEC1]], -1
; CHECK: [[EXEC2:%.+]] = and i32 [[SRC]], [[NOT_EXEC]]
; CHECK: icmp ne i32 [[EXEC2]], 0
	s_and_not1_wrexec_b32 s2, s1
; CHECK: xor i32 [[EXEC2]], -1
	s_not_b32 s3, s2
	s_endpgm

	.globl	wide_combine_kernel
	.p2align	8
	.type	wide_combine_kernel,@function
wide_combine_kernel:
; REFUSE: unsupported-instruction-form: s_and_saveexec_b64
; REFUSE-SAME: combines a 64-bit mask with EXEC, but the source wave is 32 bits wide and EXEC holds 32 bits
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_endpgm

	.globl	wide_exec_write_kernel
	.p2align	8
	.type	wide_exec_write_kernel,@function
wide_exec_write_kernel:
; REFUSE: unsupported-instruction-form: s_mov_b64
; REFUSE-SAME: writes a 64-bit mask to EXEC, but the source wave is 32 bits wide and EXEC holds 32 bits
	s_mov_b64 exec, s[0:1]
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
; REFUSE: unsupported-instruction-form: s_rfe_i64
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
