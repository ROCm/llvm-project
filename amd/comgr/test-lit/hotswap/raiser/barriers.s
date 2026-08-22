; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=workgroup_barrier_kernel \
; RUN:   | %FileCheck %s --check-prefix=WORKGROUP
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=workgroup_barrier_kernel \
; RUN:   | %FileCheck %s --check-prefix=ONTO-GFX942
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=signal_named_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SIGNAL-NAMED
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=signal_cluster_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SIGNAL-CLUSTER
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=signal_m0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SIGNAL-M0
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=signal_isfirst_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ISFIRST-IMM
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=signal_isfirst_m0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ISFIRST-M0
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=get_barrier_state_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STATE-IMM
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=get_barrier_state_m0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STATE-M0
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=barrier_init_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=INIT-IMM
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=barrier_init_m0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=INIT-M0
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=barrier_join_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=JOIN-IMM
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=barrier_join_m0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=JOIN-M0
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=wakeup_barrier_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WAKEUP-IMM
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=wakeup_barrier_m0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WAKEUP-M0
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=wait_named_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WAIT-NAMED
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=wait_cluster_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WAIT-CLUSTER
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=barrier_leave_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=LEAVE
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED-SOP1
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=trap_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED-SOPP

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	workgroup_barrier_kernel
	.p2align	8
	.type	workgroup_barrier_kernel,@function
workgroup_barrier_kernel:
; The arrival raises to the whole barrier and so does the release, so the
; source pair raises to a pair. Neither is dropped: each says on its own that
; the workgroup meets here.
; WORKGROUP-LABEL: define amdgpu_kernel void @workgroup_barrier_kernel(
; WORKGROUP-NEXT: entry:
; WORKGROUP-NEXT: call void @llvm.amdgcn.s.barrier()
; WORKGROUP-NEXT: call void @llvm.amdgcn.s.barrier()
; WORKGROUP-NEXT: ret void
; WORKGROUP-NEXT: }
; Raising onto the GPU that never had the split barrier reaches the same
; barrier, which is the point of raising it to the one every target has.
; ONTO-GFX942-LABEL: define amdgpu_kernel void @workgroup_barrier_kernel(
; ONTO-GFX942-NEXT: entry:
; ONTO-GFX942-NEXT: call void @llvm.amdgcn.s.barrier()
; ONTO-GFX942-NEXT: call void @llvm.amdgcn.s.barrier()
; ONTO-GFX942-NEXT: ret void
	s_barrier_signal -1
	s_barrier_wait -1
	s_endpgm

	.globl	signal_named_kernel
	.p2align	8
	.type	signal_named_kernel,@function
signal_named_kernel:
; SIGNAL-NAMED: unsupported-instruction-form: s_barrier_signal [SOP1]
; SIGNAL-NAMED-SAME: names barrier 1 rather than the workgroup barrier
	s_barrier_signal 1
	s_endpgm

	.globl	signal_cluster_kernel
	.p2align	8
	.type	signal_cluster_kernel,@function
signal_cluster_kernel:
; SIGNAL-CLUSTER: unsupported-instruction-form: s_barrier_signal [SOP1]
; SIGNAL-CLUSTER-SAME: names barrier -3 rather than the workgroup barrier
	s_barrier_signal -3
	s_endpgm

	.globl	signal_m0_kernel
	.p2align	8
	.type	signal_m0_kernel,@function
signal_m0_kernel:
; SIGNAL-M0: unsupported-instruction-form: s_barrier_signal [SOP1]
; SIGNAL-M0-SAME: takes its barrier id from m0
	s_barrier_signal m0
	s_endpgm

	.globl	signal_isfirst_kernel
	.p2align	8
	.type	signal_isfirst_kernel,@function
signal_isfirst_kernel:
; ISFIRST-IMM: unsupported-instruction-form: s_barrier_signal_isfirst [SOP1]
; ISFIRST-IMM-SAME: reports whether this wave arrived at the barrier first
	s_barrier_signal_isfirst -1
	s_endpgm

	.globl	signal_isfirst_m0_kernel
	.p2align	8
	.type	signal_isfirst_m0_kernel,@function
signal_isfirst_m0_kernel:
; ISFIRST-M0: unsupported-instruction-form: s_barrier_signal_isfirst [SOP1]
; ISFIRST-M0-SAME: reports whether this wave arrived at the barrier first
	s_barrier_signal_isfirst m0
	s_endpgm

	.globl	get_barrier_state_kernel
	.p2align	8
	.type	get_barrier_state_kernel,@function
get_barrier_state_kernel:
; STATE-IMM: unsupported-instruction-form: s_get_barrier_state [SOP1]
; STATE-IMM-SAME: reads the arrival and membership counts
	s_get_barrier_state s0, 1
	s_endpgm

	.globl	get_barrier_state_m0_kernel
	.p2align	8
	.type	get_barrier_state_m0_kernel,@function
get_barrier_state_m0_kernel:
; STATE-M0: unsupported-instruction-form: s_get_barrier_state [SOP1]
; STATE-M0-SAME: reads the arrival and membership counts
	s_get_barrier_state s0, m0
	s_endpgm

	.globl	barrier_init_kernel
	.p2align	8
	.type	barrier_init_kernel,@function
barrier_init_kernel:
; INIT-IMM: unsupported-instruction-form: s_barrier_init [SOP1]
; INIT-IMM-SAME: sizes the membership of a named barrier
	s_barrier_init 1
	s_endpgm

	.globl	barrier_init_m0_kernel
	.p2align	8
	.type	barrier_init_m0_kernel,@function
barrier_init_m0_kernel:
; INIT-M0: unsupported-instruction-form: s_barrier_init [SOP1]
; INIT-M0-SAME: sizes the membership of a named barrier
	s_barrier_init m0
	s_endpgm

	.globl	barrier_join_kernel
	.p2align	8
	.type	barrier_join_kernel,@function
barrier_join_kernel:
; JOIN-IMM: unsupported-instruction-form: s_barrier_join [SOP1]
; JOIN-IMM-SAME: joins this wave to a named barrier
	s_barrier_join 1
	s_endpgm

	.globl	barrier_join_m0_kernel
	.p2align	8
	.type	barrier_join_m0_kernel,@function
barrier_join_m0_kernel:
; JOIN-M0: unsupported-instruction-form: s_barrier_join [SOP1]
; JOIN-M0-SAME: joins this wave to a named barrier
	s_barrier_join m0
	s_endpgm

	.globl	wakeup_barrier_kernel
	.p2align	8
	.type	wakeup_barrier_kernel,@function
wakeup_barrier_kernel:
; WAKEUP-IMM: unsupported-instruction-form: s_wakeup_barrier [SOP1]
; WAKEUP-IMM-SAME: wakes the waves waiting on a named barrier
	s_wakeup_barrier 1
	s_endpgm

	.globl	wakeup_barrier_m0_kernel
	.p2align	8
	.type	wakeup_barrier_m0_kernel,@function
wakeup_barrier_m0_kernel:
; WAKEUP-M0: unsupported-instruction-form: s_wakeup_barrier [SOP1]
; WAKEUP-M0-SAME: wakes the waves waiting on a named barrier
	s_wakeup_barrier m0
	s_endpgm

	.globl	wait_named_kernel
	.p2align	8
	.type	wait_named_kernel,@function
wait_named_kernel:
; The immediate does not pick the named barrier out by number, so the refusal
; names the barrier the way the source does: the one joined last.
; WAIT-NAMED: unsupported-instruction-form: s_barrier_wait [SOPP]
; WAIT-NAMED-SAME: waits on the named barrier this wave joined last
	s_barrier_wait 1
	s_endpgm

	.globl	wait_cluster_kernel
	.p2align	8
	.type	wait_cluster_kernel,@function
wait_cluster_kernel:
; WAIT-CLUSTER: unsupported-instruction-form: s_barrier_wait [SOPP]
; WAIT-CLUSTER-SAME: names barrier -3 rather than the workgroup barrier
	s_barrier_wait -3
	s_endpgm

	.globl	barrier_leave_kernel
	.p2align	8
	.type	barrier_leave_kernel,@function
barrier_leave_kernel:
; LEAVE: unsupported-instruction-form: s_barrier_leave [SOPP]
; LEAVE-SAME: leaves a named barrier
	s_barrier_leave
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
; UNHANDLED-SOP1: unsupported-instruction-form: s_rfe_i64 [SOP1]
	s_rfe_i64 s[0:1]
	s_endpgm

	.globl	trap_kernel
	.p2align	8
	.type	trap_kernel,@function
trap_kernel:
; UNHANDLED-SOPP: unsupported-instruction-form: s_trap [SOPP]
	s_trap 1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel workgroup_barrier_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel signal_named_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel signal_cluster_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel signal_m0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel signal_isfirst_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel signal_isfirst_m0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel get_barrier_state_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel get_barrier_state_m0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel barrier_init_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel barrier_init_m0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel barrier_join_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel barrier_join_m0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel wakeup_barrier_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel wakeup_barrier_m0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel wait_named_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel wait_cluster_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel barrier_leave_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel trap_kernel
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
    .name:           workgroup_barrier_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         workgroup_barrier_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           signal_named_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         signal_named_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           signal_cluster_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         signal_cluster_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           signal_m0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         signal_m0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           signal_isfirst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         signal_isfirst_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           signal_isfirst_m0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         signal_isfirst_m0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           get_barrier_state_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         get_barrier_state_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           get_barrier_state_m0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         get_barrier_state_m0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           barrier_init_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         barrier_init_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           barrier_init_m0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         barrier_init_m0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           barrier_join_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         barrier_join_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           barrier_join_m0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         barrier_join_m0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wakeup_barrier_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         wakeup_barrier_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wakeup_barrier_m0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         wakeup_barrier_m0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wait_named_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         wait_named_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wait_cluster_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         wait_cluster_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           barrier_leave_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         barrier_leave_kernel.kd
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
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           trap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         trap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
