; REQUIRES: comgr-has-hotswap-transpile

; The 64-bit wave-mask opcodes describe a 64-lane wave, so this fixture is a
; wave64 gfx942 kernel. A 64-bit scalar register is a pair of 32-bit halves, so
; a value written to one and read back arrives split and rejoined; EXEC is held
; whole and can be named directly.
; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=quad_kernel,saveexec_kernel,wrexec_kernel \
; RUN:   | %FileCheck %s

; s_rfe_b64 has no lowering, and SOP1 refuses an opcode it does not lift rather
; than letting it through unlowered.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: s_rfe_b64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	quad_kernel
	.p2align	8
	.type	quad_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @quad_kernel(
quad_kernel:
; A named value to start from, so the checks below pin what each opcode read.
; CHECK: [[SEED:%.+]] = xor i64 {{.+}}, -1
; CHECK: [[SEED_LO:%.+]] = trunc i64 [[SEED]] to i32
; CHECK: [[SEED_SHIFTED:%.+]] = lshr i64 [[SEED]], 32
; CHECK: [[SEED_HI:%.+]] = trunc i64 [[SEED_SHIFTED]] to i32
	s_not_b64 s[0:1], s[0:1]
; CHECK: [[SRC_LO:%.+]] = zext i32 [[SEED_LO]] to i64
; CHECK: [[SRC_HI:%.+]] = zext i32 [[SEED_HI]] to i64
; CHECK: [[SRC_HI_SHIFTED:%.+]] = shl i64 [[SRC_HI]], 32
; CHECK: [[SRC:%.+]] = or i64 [[SRC_LO]], [[SRC_HI_SHIFTED]]
; s_quadmask folds each group of four bits onto the lowest bit of the group,
; keeps only those bits, and then halves the distance between them until they
; are adjacent: 0x1111111111111111, then two bits per byte, four per halfword,
; eight per dword, sixteen in the low word.
; CHECK: [[Q_SHIFTED:%.+]] = lshr i64 [[SRC]], 2
; CHECK: [[Q_PAIRS:%.+]] = or i64 [[SRC]], [[Q_SHIFTED]]
; CHECK: [[Q_PAIRS_SHIFTED:%.+]] = lshr i64 [[Q_PAIRS]], 1
; CHECK: [[Q_ANY:%.+]] = or i64 [[Q_PAIRS]], [[Q_PAIRS_SHIFTED]]
; CHECK: [[Q_LOW:%.+]] = and i64 [[Q_ANY]], 1229782938247303441
; CHECK: [[Q_BY3:%.+]] = lshr i64 [[Q_LOW]], 3
; CHECK: [[Q_JOIN2:%.+]] = or i64 [[Q_LOW]], [[Q_BY3]]
; CHECK: [[Q_KEEP2:%.+]] = and i64 [[Q_JOIN2]], 217020518514230019
; CHECK: [[Q_BY6:%.+]] = lshr i64 [[Q_KEEP2]], 6
; CHECK: [[Q_JOIN4:%.+]] = or i64 [[Q_KEEP2]], [[Q_BY6]]
; CHECK: [[Q_KEEP4:%.+]] = and i64 [[Q_JOIN4]], 4222189076152335
; CHECK: [[Q_BY12:%.+]] = lshr i64 [[Q_KEEP4]], 12
; CHECK: [[Q_JOIN8:%.+]] = or i64 [[Q_KEEP4]], [[Q_BY12]]
; CHECK: [[Q_KEEP8:%.+]] = and i64 [[Q_JOIN8]], 1095216660735
; CHECK: [[Q_BY24:%.+]] = lshr i64 [[Q_KEEP8]], 24
; CHECK: [[Q_JOIN16:%.+]] = or i64 [[Q_KEEP8]], [[Q_BY24]]
; CHECK: [[QUADMASK:%.+]] = and i64 [[Q_JOIN16]], 65535
; CHECK: [[QUAD_LO:%.+]] = trunc i64 [[QUADMASK]] to i32
; CHECK: [[QUAD_SHIFTED:%.+]] = lshr i64 [[QUADMASK]], 32
; CHECK: [[QUAD_HI:%.+]] = trunc i64 [[QUAD_SHIFTED]] to i32
; CHECK: icmp ne i64 [[QUADMASK]], 0
	s_quadmask_b64 s[2:3], s[0:1]
; s_wqm stops at the same one-bit-per-group value and spreads it back over the
; whole group instead of packing it.
; CHECK: [[WSRC_LO:%.+]] = zext i32 [[QUAD_LO]] to i64
; CHECK: [[WSRC_HI:%.+]] = zext i32 [[QUAD_HI]] to i64
; CHECK: [[WSRC_HI_SHIFTED:%.+]] = shl i64 [[WSRC_HI]], 32
; CHECK: [[WSRC:%.+]] = or i64 [[WSRC_LO]], [[WSRC_HI_SHIFTED]]
; CHECK: [[W_SHIFTED:%.+]] = lshr i64 [[WSRC]], 2
; CHECK: [[W_PAIRS:%.+]] = or i64 [[WSRC]], [[W_SHIFTED]]
; CHECK: [[W_PAIRS_SHIFTED:%.+]] = lshr i64 [[W_PAIRS]], 1
; CHECK: [[W_ANY:%.+]] = or i64 [[W_PAIRS]], [[W_PAIRS_SHIFTED]]
; CHECK: [[W_LOW:%.+]] = and i64 [[W_ANY]], 1229782938247303441
; CHECK: [[W_TWO:%.+]] = shl i64 [[W_LOW]], 1
; CHECK: [[W_HALF:%.+]] = or i64 [[W_LOW]], [[W_TWO]]
; CHECK: [[W_FOUR:%.+]] = shl i64 [[W_HALF]], 2
; CHECK: [[WQM:%.+]] = or i64 [[W_HALF]], [[W_FOUR]]
; CHECK: [[WQM_LO:%.+]] = trunc i64 [[WQM]] to i32
; CHECK: [[WQM_SHIFTED:%.+]] = lshr i64 [[WQM]], 32
; CHECK: [[WQM_HI:%.+]] = trunc i64 [[WQM_SHIFTED]] to i32
; CHECK: icmp ne i64 [[WQM]], 0
	s_wqm_b64 s[4:5], s[2:3]
; CHECK: [[RSRC_LO:%.+]] = zext i32 [[WQM_LO]] to i64
; CHECK: [[RSRC_HI:%.+]] = zext i32 [[WQM_HI]] to i64
; CHECK: [[RSRC_HI_SHIFTED:%.+]] = shl i64 [[RSRC_HI]], 32
; CHECK: [[RSRC:%.+]] = or i64 [[RSRC_LO]], [[RSRC_HI_SHIFTED]]
; CHECK: xor i64 [[RSRC]], -1
	s_not_b64 s[6:7], s[4:5]
	s_endpgm

	.globl	saveexec_kernel
	.p2align	8
	.type	saveexec_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @saveexec_kernel(
saveexec_kernel:
; Move EXEC off the all-ones mask the kernel starts with, so a check naming it
; cannot pass on a constant that happens to be there.
; CHECK: [[EXEC0:%.+]] = xor i64 {{.+}}, -1
	s_not_b64 exec, s[0:1]
; Every opcode here leaves the EXEC it replaced in its scalar destination,
; which the trunc splitting it into register halves names and the population
; count after it reads back.
; CHECK: [[EXEC1:%.+]] = and i64 {{.+}}, [[EXEC0]]
; CHECK: icmp ne i64 [[EXEC1]], 0
; CHECK: trunc i64 [[EXEC0]] to i32
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[EXEC2:%.+]] = or i64 {{.+}}, [[EXEC1]]
; CHECK: icmp ne i64 [[EXEC2]], 0
; CHECK: trunc i64 [[EXEC1]] to i32
	s_or_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[EXEC3:%.+]] = xor i64 {{.+}}, [[EXEC2]]
; CHECK: icmp ne i64 [[EXEC3]], 0
; CHECK: trunc i64 [[EXEC2]] to i32
	s_xor_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[NAND_AND:%.+]] = and i64 {{.+}}, [[EXEC3]]
; CHECK: [[EXEC4:%.+]] = xor i64 [[NAND_AND]], -1
; CHECK: icmp ne i64 [[EXEC4]], 0
; CHECK: trunc i64 [[EXEC3]] to i32
	s_nand_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[NOR_OR:%.+]] = or i64 {{.+}}, [[EXEC4]]
; CHECK: [[EXEC5:%.+]] = xor i64 [[NOR_OR]], -1
; CHECK: icmp ne i64 [[EXEC5]], 0
; CHECK: trunc i64 [[EXEC4]] to i32
	s_nor_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[XNOR_XOR:%.+]] = xor i64 {{.+}}, [[EXEC5]]
; CHECK: [[EXEC6:%.+]] = xor i64 [[XNOR_XOR]], -1
; CHECK: icmp ne i64 [[EXEC6]], 0
; CHECK: trunc i64 [[EXEC5]] to i32
	s_xnor_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; The n1 forms complement the scalar source and leave EXEC as it is, and the n2
; forms complement EXEC, which is what tells the two apart.
; CHECK: [[NOT_SRC1:%.+]] = xor i64 {{.+}}, -1
; CHECK: [[EXEC7:%.+]] = and i64 [[NOT_SRC1]], [[EXEC6]]
; CHECK: icmp ne i64 [[EXEC7]], 0
; CHECK: trunc i64 [[EXEC6]] to i32
	s_andn1_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[NOT_SRC2:%.+]] = xor i64 {{.+}}, -1
; CHECK: [[EXEC8:%.+]] = or i64 [[NOT_SRC2]], [[EXEC7]]
; CHECK: icmp ne i64 [[EXEC8]], 0
; CHECK: trunc i64 [[EXEC7]] to i32
	s_orn1_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[NOT_EXEC1:%.+]] = xor i64 [[EXEC8]], -1
; CHECK: [[EXEC9:%.+]] = and i64 {{.+}}, [[NOT_EXEC1]]
; CHECK: icmp ne i64 [[EXEC9]], 0
; CHECK: trunc i64 [[EXEC8]] to i32
	s_andn2_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[NOT_EXEC2:%.+]] = xor i64 [[EXEC9]], -1
; CHECK: [[EXEC10:%.+]] = or i64 {{.+}}, [[NOT_EXEC2]]
; CHECK: icmp ne i64 [[EXEC10]], 0
; CHECK: trunc i64 [[EXEC9]] to i32
	s_orn2_saveexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
	s_endpgm

	.globl	wrexec_kernel
	.p2align	8
	.type	wrexec_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @wrexec_kernel(
wrexec_kernel:
; Move EXEC off the all-ones mask the kernel starts with, so a check naming it
; cannot pass on a constant that happens to be there.
; CHECK: [[EXEC0:%.+]] = xor i64 {{.+}}, -1
	s_not_b64 exec, s[0:1]
; A wrexec opcode leaves the mask it computed in its scalar destination, not
; the EXEC it replaced.
; CHECK: [[NOT_SRC:%.+]] = xor i64 {{.+}}, -1
; CHECK: [[EXEC1:%.+]] = and i64 [[NOT_SRC]], [[EXEC0]]
; CHECK: icmp ne i64 [[EXEC1]], 0
; CHECK: trunc i64 [[EXEC1]] to i32
	s_andn1_wrexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
; CHECK: [[NOT_EXEC:%.+]] = xor i64 [[EXEC1]], -1
; CHECK: [[EXEC2:%.+]] = and i64 {{.+}}, [[NOT_EXEC]]
; CHECK: icmp ne i64 [[EXEC2]], 0
; CHECK: trunc i64 [[EXEC2]] to i32
	s_andn2_wrexec_b64 s[2:3], s[0:1]
	s_bcnt1_i32_b64 s10, s[2:3]
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
	s_rfe_b64 s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel quad_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_accum_offset 4
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel saveexec_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_accum_offset 4
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel wrexec_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_accum_offset 4
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_accum_offset 4
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
    .name:           quad_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         quad_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           saveexec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         saveexec_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wrexec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         wrexec_kernel.kd
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
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
