; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=workgroup_barrier_kernel \
; RUN:   | %FileCheck %s
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=workgroup_barrier_kernel \
; RUN:   | %FileCheck %s --check-prefix=ONTO-GFX942

; The barriers that name a barrier other than the workgroup one, or ask about
; barrier state, are refused: the raised kernel has neither the source wave's
; barrier resources nor its cluster. So are the SOP1 and SOPP opcodes the
; handlers do not lift at all.
; RUN: not %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=signal_named_kernel,signal_cluster_kernel,signal_m0_kernel \
; RUN:   --emit-ir=signal_isfirst_kernel,signal_isfirst_m0_kernel \
; RUN:   --emit-ir=get_barrier_state_kernel,get_barrier_state_m0_kernel \
; RUN:   --emit-ir=barrier_init_kernel,barrier_init_m0_kernel \
; RUN:   --emit-ir=barrier_join_kernel,barrier_join_m0_kernel \
; RUN:   --emit-ir=wakeup_barrier_kernel,wakeup_barrier_m0_kernel \
; RUN:   --emit-ir=wait_named_kernel,wait_cluster_kernel,barrier_leave_kernel \
; RUN:   --emit-ir=rfe_kernel,trap_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	workgroup_barrier_kernel
	.p2align	8
	.type	workgroup_barrier_kernel,@function
workgroup_barrier_kernel:
; The arrival raises to the whole barrier and so does the release, so the
; source pair raises to a pair. Neither is dropped: each says on its own that
; the workgroup meets here.
; CHECK-LABEL: define amdgpu_kernel void @workgroup_barrier_kernel(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.amdgcn.s.barrier()
; CHECK-NEXT: call void @llvm.amdgcn.s.barrier()
; CHECK-NEXT: ret void
; CHECK-NEXT: }
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
; REFUSE: unsupported-instruction-form: s_barrier_signal [SOP1]
; REFUSE-SAME: names barrier 1 rather than the workgroup barrier
	s_barrier_signal 1
	s_endpgm

	.globl	signal_cluster_kernel
	.p2align	8
	.type	signal_cluster_kernel,@function
signal_cluster_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_signal [SOP1]
; REFUSE-SAME: names barrier -3 rather than the workgroup barrier
	s_barrier_signal -3
	s_endpgm

	.globl	signal_m0_kernel
	.p2align	8
	.type	signal_m0_kernel,@function
signal_m0_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_signal [SOP1]
; REFUSE-SAME: takes its barrier id from m0
	s_barrier_signal m0
	s_endpgm

	.globl	signal_isfirst_kernel
	.p2align	8
	.type	signal_isfirst_kernel,@function
signal_isfirst_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_signal_isfirst [SOP1]
; REFUSE-SAME: reports whether this wave arrived at the barrier first
	s_barrier_signal_isfirst -1
	s_endpgm

	.globl	signal_isfirst_m0_kernel
	.p2align	8
	.type	signal_isfirst_m0_kernel,@function
signal_isfirst_m0_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_signal_isfirst [SOP1]
; REFUSE-SAME: reports whether this wave arrived at the barrier first
	s_barrier_signal_isfirst m0
	s_endpgm

	.globl	get_barrier_state_kernel
	.p2align	8
	.type	get_barrier_state_kernel,@function
get_barrier_state_kernel:
; REFUSE: unsupported-instruction-form: s_get_barrier_state [SOP1]
; REFUSE-SAME: reads the arrival and membership counts
	s_get_barrier_state s0, 1
	s_endpgm

	.globl	get_barrier_state_m0_kernel
	.p2align	8
	.type	get_barrier_state_m0_kernel,@function
get_barrier_state_m0_kernel:
; REFUSE: unsupported-instruction-form: s_get_barrier_state [SOP1]
; REFUSE-SAME: reads the arrival and membership counts
	s_get_barrier_state s0, m0
	s_endpgm

	.globl	barrier_init_kernel
	.p2align	8
	.type	barrier_init_kernel,@function
barrier_init_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_init [SOP1]
; REFUSE-SAME: sizes the membership of a named barrier
	s_barrier_init 1
	s_endpgm

	.globl	barrier_init_m0_kernel
	.p2align	8
	.type	barrier_init_m0_kernel,@function
barrier_init_m0_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_init [SOP1]
; REFUSE-SAME: sizes the membership of a named barrier
	s_barrier_init m0
	s_endpgm

	.globl	barrier_join_kernel
	.p2align	8
	.type	barrier_join_kernel,@function
barrier_join_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_join [SOP1]
; REFUSE-SAME: joins this wave to a named barrier
	s_barrier_join 1
	s_endpgm

	.globl	barrier_join_m0_kernel
	.p2align	8
	.type	barrier_join_m0_kernel,@function
barrier_join_m0_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_join [SOP1]
; REFUSE-SAME: joins this wave to a named barrier
	s_barrier_join m0
	s_endpgm

	.globl	wakeup_barrier_kernel
	.p2align	8
	.type	wakeup_barrier_kernel,@function
wakeup_barrier_kernel:
; REFUSE: unsupported-instruction-form: s_wakeup_barrier [SOP1]
; REFUSE-SAME: wakes the waves waiting on a named barrier
	s_wakeup_barrier 1
	s_endpgm

	.globl	wakeup_barrier_m0_kernel
	.p2align	8
	.type	wakeup_barrier_m0_kernel,@function
wakeup_barrier_m0_kernel:
; REFUSE: unsupported-instruction-form: s_wakeup_barrier [SOP1]
; REFUSE-SAME: wakes the waves waiting on a named barrier
	s_wakeup_barrier m0
	s_endpgm

	.globl	wait_named_kernel
	.p2align	8
	.type	wait_named_kernel,@function
wait_named_kernel:
; The immediate does not pick the named barrier out by number, so the refusal
; names the barrier the way the source does: the one joined last.
; REFUSE: unsupported-instruction-form: s_barrier_wait [SOPP]
; REFUSE-SAME: waits on the named barrier this wave joined last
	s_barrier_wait 1
	s_endpgm

	.globl	wait_cluster_kernel
	.p2align	8
	.type	wait_cluster_kernel,@function
wait_cluster_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_wait [SOPP]
; REFUSE-SAME: names barrier -3 rather than the workgroup barrier
	s_barrier_wait -3
	s_endpgm

	.globl	barrier_leave_kernel
	.p2align	8
	.type	barrier_leave_kernel,@function
barrier_leave_kernel:
; REFUSE: unsupported-instruction-form: s_barrier_leave [SOPP]
; REFUSE-SAME: leaves a named barrier
	s_barrier_leave
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
; REFUSE: unsupported-instruction-form: s_rfe_i64 [SOP1]
	s_rfe_i64 s[0:1]
	s_endpgm

	.globl	trap_kernel
	.p2align	8
	.type	trap_kernel,@function
trap_kernel:
; REFUSE: unsupported-instruction-form: s_trap [SOPP]
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
