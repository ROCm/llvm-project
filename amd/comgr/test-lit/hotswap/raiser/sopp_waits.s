; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=wait_kernel \
; RUN:   | %FileCheck %s --check-prefix=WAIT
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=inert_kernel \
; RUN:   | %FileCheck %s --check-prefix=INERT
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=trap_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=TRAP
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=sethalt_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SETHALT
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=endpgm_saved_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ENDPGM-SAVED
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=branch_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BRANCH
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_scc0_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CBRANCH-SCC0
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=cbranch_execz_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CBRANCH-EXECZ
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=barrier_wait_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BARRIER-WAIT
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=barrier_leave_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BARRIER-LEAVE
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=sendmsg_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SENDMSG

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	wait_kernel
	.p2align	8
	.type	wait_kernel,@function
wait_kernel:
; Each counter lowers to one fence and to nothing else; the scope and the
; ordering are what let one fence stand in for every counter.
; WAIT-LABEL: define amdgpu_kernel void @wait_kernel(
; WAIT-NEXT: entry:
; WAIT-COUNT-11: fence syncscope("agent") seq_cst
; WAIT-NOT: fence
; WAIT: ret void
; WAIT-NEXT: }
	s_wait_alu depctr_va_vdst(0)
	s_wait_idle
	s_wait_loadcnt 0
	s_wait_storecnt 0
	s_wait_xcnt 0
	s_wait_dscnt 0
	s_wait_kmcnt 0
	s_wait_loadcnt_dscnt 0
	s_wait_storecnt_dscnt 0
	s_wait_asynccnt 0
	s_wait_tensorcnt 0
	s_endpgm

	.globl	inert_kernel
	.p2align	8
	.type	inert_kernel,@function
inert_kernel:
; The inert opcodes are lifted and emit nothing, so the body is the terminator.
; INERT-LABEL: define amdgpu_kernel void @inert_kernel(
; INERT-NEXT: entry:
; INERT-NEXT: ret void
; INERT-NEXT: }
	s_nop 0
	s_sleep 1
	s_monitor_sleep 1
	s_clause 1
	s_delay_alu instid0(VALU_DEP_1)
	s_wakeup
	s_setprio 1
	s_setprio_inc_wg 1
	s_incperflevel 1
	s_decperflevel 1
	s_ttracedata
	s_ttracedata_imm 1
	s_icache_inv
	s_code_end
	s_endpgm

; Every other SOPP opcode is refused by name. Traps, halts and the alternate
; terminators change what the kernel does; branches, barriers and messages are
; control flow and communication the raise does not carry.

	.globl	trap_kernel
	.p2align	8
	.type	trap_kernel,@function
trap_kernel:
; TRAP: unsupported-instruction-form: s_trap [SOPP]
	s_trap 1
	s_endpgm

	.globl	sethalt_kernel
	.p2align	8
	.type	sethalt_kernel,@function
sethalt_kernel:
; SETHALT: unsupported-instruction-form: s_sethalt [SOPP]
	s_sethalt 1
	s_endpgm

	.globl	endpgm_saved_kernel
	.p2align	8
	.type	endpgm_saved_kernel,@function
endpgm_saved_kernel:
; ENDPGM-SAVED: unsupported-instruction-form: s_endpgm_saved [SOPP]
	s_endpgm_saved

	.globl	branch_kernel
	.p2align	8
	.type	branch_kernel,@function
branch_kernel:
; BRANCH: unsupported-instruction-form: s_branch [SOPP]
	s_branch 0
	s_endpgm

	.globl	cbranch_scc0_kernel
	.p2align	8
	.type	cbranch_scc0_kernel,@function
cbranch_scc0_kernel:
; CBRANCH-SCC0: unsupported-instruction-form: s_cbranch_scc0 [SOPP]
	s_cbranch_scc0 0
	s_endpgm

	.globl	cbranch_execz_kernel
	.p2align	8
	.type	cbranch_execz_kernel,@function
cbranch_execz_kernel:
; CBRANCH-EXECZ: unsupported-instruction-form: s_cbranch_execz [SOPP]
	s_cbranch_execz 0
	s_endpgm

	.globl	barrier_wait_kernel
	.p2align	8
	.type	barrier_wait_kernel,@function
barrier_wait_kernel:
; BARRIER-WAIT: unsupported-instruction-form: s_barrier_wait [SOPP]
	s_barrier_wait 1
	s_endpgm

	.globl	barrier_leave_kernel
	.p2align	8
	.type	barrier_leave_kernel,@function
barrier_leave_kernel:
; BARRIER-LEAVE: unsupported-instruction-form: s_barrier_leave [SOPP]
	s_barrier_leave
	s_endpgm

	.globl	sendmsg_kernel
	.p2align	8
	.type	sendmsg_kernel,@function
sendmsg_kernel:
; SENDMSG: unsupported-instruction-form: s_sendmsg [SOPP]
	s_sendmsg 1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wait_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel inert_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel trap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sethalt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel endpgm_saved_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel branch_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_scc0_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel cbranch_execz_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel barrier_wait_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel barrier_leave_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sendmsg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
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
    .name:           wait_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         wait_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           inert_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         inert_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           trap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         trap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sethalt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sethalt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           endpgm_saved_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         endpgm_saved_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           branch_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         branch_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_scc0_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         cbranch_scc0_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cbranch_execz_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         cbranch_execz_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           barrier_wait_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         barrier_wait_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           barrier_leave_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         barrier_leave_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sendmsg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sendmsg_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
