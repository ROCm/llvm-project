; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	skip_kernel
	.p2align	8
	.type	skip_kernel,@function

; The shape a guarded kernel compiles to: skip the body when no lane is active,
; and rejoin at the end. EXECZ tests the whole source EXEC mask, which the
; raiser models at the source wave width, so the compare is against a value of
; that width whatever the target runs. It is also wave-uniform, which is what
; keeps it a branch rather than per-lane predication.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=skip_kernel | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @skip_kernel(
; CHECK: br label %bb_0x0
; CHECK: bb_0x0:
; CHECK: [[EXECZ:%.+]] = icmp eq i64 -1, 0
; CHECK-NEXT: br i1 [[EXECZ]], label %bb_0x8, label %bb_0x4
; CHECK: bb_0x4:
; CHECK-NEXT: br label %bb_0x8
; CHECK: bb_0x8:
; CHECK-NEXT: ret void

; Raising onto a wave32 target does not change which mask the branch reads: EXEC
; width is a property of the ISA the code object was compiled for, so the source
; mask stays 64-bit and the test stays exact.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=skip_kernel \
; RUN:   --target-isa=gfx1250 | %FileCheck %s --check-prefix=TOGFX1250
; TOGFX1250-LABEL: define amdgpu_kernel void @skip_kernel(
; TOGFX1250: [[EXECZ:%.+]] = icmp eq i64 -1, 0
; TOGFX1250-NEXT: br i1 [[EXECZ]]

; The decoder maps each branch onto its canonical op, which is what routes it to
; the arm that recovers an edge rather than refusing.
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=skip_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE
skip_kernel:
; DECODE: S_CBRANCH_EXECZ{{.+}}s_cbranch_execz
	s_cbranch_execz .Lskip_end
	s_mov_b32 s0, 1
.Lskip_end:
	s_endpgm

	.globl	loop_kernel
	.p2align	8
	.type	loop_kernel,@function

; A backward branch to the kernel's first instruction gives that block a
; predecessor. LLVM forbids an entry block from having one, so the register file
; lives in a block of its own that branches into the body rather than in the
; body's first block.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=loop_kernel \
; RUN:   | %FileCheck %s --check-prefix=LOOP
; LOOP-LABEL: define amdgpu_kernel void @loop_kernel(
; LOOP: entry:
; LOOP: br label %bb_0x0
; LOOP: bb_0x0:
; LOOP-SAME: preds = %bb_0x0, %entry
loop_kernel:
	s_mov_b32 s0, 1
	s_cbranch_scc1 loop_kernel
	s_endpgm

	.globl	early_return_kernel
	.p2align	8
	.type	early_return_kernel,@function

; An s_endpgm partway through a kernel ends its block, not the kernel: the
; decode carries on while a recovered branch target still lies ahead of it. The
; bytes an unconditional branch skips over are not decoded into any block --
; nothing reaches them.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=early_return_kernel \
; RUN:   | %FileCheck %s --check-prefix=EARLY
; EARLY-LABEL: define amdgpu_kernel void @early_return_kernel(
; EARLY: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 false)
; EARLY-NEXT: [[VCCZ:%.+]] = icmp eq i64 [[BALLOT]], 0
; EARLY-NEXT: br i1 [[VCCZ]], label %bb_0x8, label %bb_0x4
; EARLY: bb_0x4:
; EARLY-NEXT: ret void
; EARLY: bb_0x8:
; EARLY-NEXT: br label %bb_0x10
; EARLY: bb_0x10:
; EARLY-NEXT: ret void
early_return_kernel:
	s_cbranch_vccz .Lafter
	s_endpgm
.Lafter:
	s_branch .Ltail
	s_nop 0
.Ltail:
	s_endpgm

	.globl	conditions_kernel
	.p2align	8
	.type	conditions_kernel,@function

; Every SOPP conditional branch reads one of three condition sources, and the
; negated form of each is the same test inverted. SCC is a plain bit; VCC is
; modeled per lane and comes back through the wave projection as the mask the
; source observes.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=conditions_kernel \
; RUN:   | %FileCheck %s --check-prefix=COND
; COND-LABEL: define amdgpu_kernel void @conditions_kernel(
; COND: [[SCCZ:%.+]] = xor i1 false, true
; COND-NEXT: br i1 [[SCCZ]]
; COND: [[VCCZ:%.+]] = icmp eq i64 %{{.+}}, 0
; COND-NEXT: [[VCCNZ:%.+]] = xor i1 [[VCCZ]], true
; COND-NEXT: br i1 [[VCCNZ]]
; COND: [[EXECZ:%.+]] = icmp eq i64 -1, 0
; COND-NEXT: [[EXECNZ:%.+]] = xor i1 [[EXECZ]], true
; COND-NEXT: br i1 [[EXECNZ]]
conditions_kernel:
	s_cbranch_scc0 .Lcond_end
	s_cbranch_vccnz .Lcond_end
	s_cbranch_execnz .Lcond_end
.Lcond_end:
	s_endpgm

	.globl	underflow_kernel
	.p2align	8
	.type	underflow_kernel,@function

; Nothing constrains the displacement a code object carries, so a branch near
; the start of the text section can address before it. The offset that computes
; is rejected like any other target outside the extent, rather than reaching a
; block lookup with nothing to find.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=underflow_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNDERFLOW
; UNDERFLOW: kernel-boundary-violation: s_branch [SOPP]
; UNDERFLOW-SAME: is outside the kernel extent
underflow_kernel:
	s_branch 0x8000

	.globl	escape_kernel
	.p2align	8
	.type	escape_kernel,@function

; A branch out of the kernel's own symbol is refused: following it would raise
; whichever symbol the target lands in as if it were part of this kernel. The
; explicit size is what bounds the extent tightly enough for the branch to leave
; it.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=escape_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ESCAPE
; ESCAPE: kernel-boundary-violation: s_branch [SOPP]
; ESCAPE-SAME: is outside the kernel extent
escape_kernel:
	s_branch dangle_kernel
	.size	escape_kernel, .-escape_kernel

	.globl	dangle_kernel
	.p2align	8
	.type	dangle_kernel,@function

; A conditional branch as the last instruction of the extent leaves the
; not-taken path running off the end of the kernel, which means the code is
; truncated or the extent is misbounded.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=dangle_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=DANGLE
; DANGLE: unterminated-kernel-extent: s_cbranch_execz [SOPP]
dangle_kernel:
	s_cbranch_execz dangle_kernel
	.size	dangle_kernel, .-dangle_kernel

	.globl	midinst_kernel
	.p2align	8
	.type	midinst_kernel,@function

; A branch target inside the extent still has to name an instruction. This one
; addresses the literal half of the following 8-byte s_mov, so no instruction
; ever raises into the block it names.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=midinst_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=MIDINST
; MIDINST: kernel-boundary-violation
; MIDINST-SAME: branch target 0x{{[0-9A-F]+}} does not begin an instruction
midinst_kernel:
	s_cbranch_execz 1
	s_mov_b32 s0, 0x12345678
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel skip_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel loop_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel early_return_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel conditions_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel underflow_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel escape_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel dangle_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel midinst_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
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
    .name:           skip_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         skip_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           loop_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         loop_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           early_return_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         early_return_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           conditions_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         conditions_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           underflow_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         underflow_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           escape_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         escape_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dangle_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         dangle_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           midinst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         midinst_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
