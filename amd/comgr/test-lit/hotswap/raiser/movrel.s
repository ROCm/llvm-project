; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=movrels_kernel,movreld_kernel,movrelsd_2_kernel,m0_reload_kernel,last_sgpr_kernel \
; RUN:   | %FileCheck %s

; A raise that refuses reports one reason and returns no module, so each
; refused shape needs its own run.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=dynamic_m0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=DYNAMIC
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=odd_index_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ODD
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=past_last_sgpr_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PAST-LAST-SGPR
; RUN: not %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=past_last_sgpr_pair_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PAST-LAST-PAIR
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=not_sgpr_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=NOT-SGPR
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=unhandled_sop1_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; Every register these kernels touch is seeded with its own index, so the value
; a relative access carries names the register it resolved to. The
; s_cvt_f32_u32 after each move is a probe: it reads the register the move read
; or wrote and puts that value in the emitted IR where a CHECK can see it.
	.globl	movrels_kernel
	.p2align	8
	.type	movrels_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @movrels_kernel(
movrels_kernel:
	s_mov_b32 s7, 7
	s_mov_b32 s8, 8
	s_mov_b32 s9, 9
	s_mov_b32 s16, 16
	s_mov_b32 s17, 17
	s_mov_b32 s18, 18
	s_mov_b32 s19, 19
	s_mov_b32 m0, 10
; s_movrels displaces its source index, so s7 with M0 = 10 reads s17.
	s_movrels_b32 s5, s7
; CHECK: uitofp i32 17 to float
	s_cvt_f32_u32 s1, s5
; The pair based at s8 reads the pair based at s18. A 64-bit read pairs the two
; halves and the write splits them again, so the destination halves are the
; ends of that chain rather than plain constants.
; CHECK: [[S18:%.+]] = zext i32 18 to i64
; CHECK: [[S19:%.+]] = zext i32 19 to i64
; CHECK: [[HI:%.+]] = shl i64 [[S19]], 32
; CHECK: [[PAIR:%.+]] = or i64 [[S18]], [[HI]]
; CHECK: [[LOW:%.+]] = trunc i64 [[PAIR]] to i32
; CHECK: [[SHIFTED:%.+]] = lshr i64 [[PAIR]], 32
; CHECK: [[HIGH:%.+]] = trunc i64 [[SHIFTED]] to i32
	s_movrels_b64 s[2:3], s[8:9]
; CHECK: uitofp i32 [[LOW]] to float
	s_cvt_f32_u32 s1, s2
; CHECK: uitofp i32 [[HIGH]] to float
	s_cvt_f32_u32 s1, s3
	s_endpgm

	.globl	movreld_kernel
	.p2align	8
	.type	movreld_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @movreld_kernel(
movreld_kernel:
	s_mov_b32 s7, 7
	s_mov_b32 s8, 8
	s_mov_b32 s9, 9
	s_mov_b32 s14, 14
	s_mov_b32 s15, 15
	s_mov_b32 s16, 16
	s_mov_b32 m0, 10
; s_movreld displaces its destination index, so s5 with M0 = 10 writes s15
; while its source stays at s7.
	s_movreld_b32 s5, s7
; CHECK: uitofp i32 7 to float
	s_cvt_f32_u32 s1, s15
; The registers either side of the resolved destination keep their own values.
; CHECK: uitofp i32 14 to float
	s_cvt_f32_u32 s1, s14
; CHECK: uitofp i32 16 to float
	s_cvt_f32_u32 s1, s16
; The pair based at s2 is written through the pair based at s12. A 64-bit read
; pairs the two source halves and the write splits them again, so what lands in
; the destination halves is the end of that chain rather than a plain constant.
; CHECK: [[S8:%.+]] = zext i32 8 to i64
; CHECK: [[S9:%.+]] = zext i32 9 to i64
; CHECK: [[HI:%.+]] = shl i64 [[S9]], 32
; CHECK: [[PAIR:%.+]] = or i64 [[S8]], [[HI]]
; CHECK: [[LOW:%.+]] = trunc i64 [[PAIR]] to i32
; CHECK: [[SHIFTED:%.+]] = lshr i64 [[PAIR]], 32
; CHECK: [[HIGH:%.+]] = trunc i64 [[SHIFTED]] to i32
	s_movreld_b64 s[2:3], s[8:9]
; CHECK: uitofp i32 [[LOW]] to float
	s_cvt_f32_u32 s1, s12
; CHECK: uitofp i32 [[HIGH]] to float
	s_cvt_f32_u32 s1, s13
	s_endpgm

	.globl	movrelsd_2_kernel
	.p2align	8
	.type	movrelsd_2_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @movrelsd_2_kernel(
movrelsd_2_kernel:
	s_mov_b32 s7, 7
	s_mov_b32 s8, 8
	s_mov_b32 s10, 10
	s_mov_b32 s15, 15
	s_mov_b32 s17, 17
; s_movrelsd_2 displaces its source by M0[9:0] and its destination by
; M0[25:16], so s7 reads s17 and s5 writes s8. Swapping the two fields would
; read s10 and write s15 instead.
	s_mov_b32 m0, 0x3000a
	s_movrelsd_2_b32 s5, s7
; CHECK: uitofp i32 17 to float
	s_cvt_f32_u32 s1, s8
; CHECK: uitofp i32 15 to float
	s_cvt_f32_u32 s1, s15
	s_endpgm

; A later write to M0 displaces the accesses that follow it.
	.globl	m0_reload_kernel
	.p2align	8
	.type	m0_reload_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @m0_reload_kernel(
m0_reload_kernel:
	s_mov_b32 s17, 17
	s_mov_b32 s27, 27
	s_mov_b32 m0, 10
	s_movrels_b32 s5, s7
; CHECK: uitofp i32 17 to float
	s_cvt_f32_u32 s1, s5
	s_mov_b32 m0, 20
	s_movrels_b32 s6, s7
; CHECK: uitofp i32 27 to float
	s_cvt_f32_u32 s1, s6
	s_endpgm

; The gfx1250 scalar file holds 108 registers, so s107 is the last index a
; relative access may resolve to, and the last pair is s[106:107].
	.globl	last_sgpr_kernel
	.p2align	8
	.type	last_sgpr_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @last_sgpr_kernel(
last_sgpr_kernel:
	s_mov_b32 m0, 102
	s_movrels_b32 s5, s5
	s_mov_b32 m0, 100
	s_movrels_b64 s[2:3], s[6:7]
; CHECK: ret void
	s_endpgm

; One index past the last register is refused.
	.globl	past_last_sgpr_kernel
	.p2align	8
	.type	past_last_sgpr_kernel,@function
past_last_sgpr_kernel:
	s_mov_b32 m0, 103
	s_movrels_b32 s5, s5
; PAST-LAST-SGPR: unsupported-instruction-form: s_movrels_b32{{.+}}movrel: resolved SGPR index 108 is out of range
	s_endpgm

; A pair needs both its halves inside the file, so the last pair a 64-bit
; access may resolve to starts two registers before the end.
	.globl	past_last_sgpr_pair_kernel
	.p2align	8
	.type	past_last_sgpr_pair_kernel,@function
past_last_sgpr_pair_kernel:
	s_mov_b32 m0, 102
	s_movrels_b64 s[2:3], s[6:7]
; PAST-LAST-PAIR: unsupported-instruction-form: s_movrels_b64{{.+}}movrel: resolved SGPR index 108 is out of range
	s_endpgm

; An M0 that is not a raise-time constant leaves the relative index unresolved.
	.globl	dynamic_m0_kernel
	.p2align	8
	.type	dynamic_m0_kernel,@function
dynamic_m0_kernel:
	s_mov_b32 m0, s4
	s_movrels_b32 s5, s7
; DYNAMIC: unsupported-instruction-form: s_movrels_b32{{.+}}movrel: M0 does not hold a constant here
	s_endpgm

; A 64-bit relative access must land on an even index.
	.globl	odd_index_kernel
	.p2align	8
	.type	odd_index_kernel,@function
odd_index_kernel:
	s_mov_b32 m0, 1
	s_movrels_b64 s[6:7], s[8:9]
; ODD: unsupported-instruction-form: s_movrels_b64{{.+}}movrel: 64-bit access resolves to odd SGPR index 9
	s_endpgm

; Only an SGPR carries an index the displacement can be added to; the scalar
; registers outside the numbered file assemble in the same operand slot.
	.globl	not_sgpr_kernel
	.p2align	8
	.type	not_sgpr_kernel,@function
not_sgpr_kernel:
	s_mov_b32 m0, 0
	s_movrels_b32 s5, vcc_lo
; NOT-SGPR: unsupported-instruction-form: s_movrels_b32{{.+}}movrel: relative operand is not an SGPR
	s_endpgm

; handleSOP1 still refuses every SOP1 opcode it does not lift.
	.globl	unhandled_sop1_kernel
	.p2align	8
	.type	unhandled_sop1_kernel,@function
unhandled_sop1_kernel:
	s_rfe_i64 s[0:1]
; UNHANDLED: unsupported-instruction-form: s_rfe_i64
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel movrels_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 32
	.end_amdhsa_kernel
	.amdhsa_kernel movreld_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 32
	.end_amdhsa_kernel
	.amdhsa_kernel movrelsd_2_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 32
	.end_amdhsa_kernel
	.amdhsa_kernel m0_reload_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 32
	.end_amdhsa_kernel
	.amdhsa_kernel last_sgpr_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.amdhsa_kernel past_last_sgpr_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.amdhsa_kernel past_last_sgpr_pair_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.amdhsa_kernel dynamic_m0_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.amdhsa_kernel odd_index_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.amdhsa_kernel not_sgpr_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.amdhsa_kernel unhandled_sop1_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 16
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
    .name:           movrels_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .symbol:         movrels_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           movreld_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .symbol:         movreld_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           movrelsd_2_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .symbol:         movrelsd_2_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           m0_reload_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .symbol:         m0_reload_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           last_sgpr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         last_sgpr_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           past_last_sgpr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         past_last_sgpr_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           past_last_sgpr_pair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         past_last_sgpr_pair_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dynamic_m0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         dynamic_m0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           odd_index_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         odd_index_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           not_sgpr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         not_sgpr_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unhandled_sop1_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         unhandled_sop1_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
