; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=wait_kernel,inert_kernel \
; RUN:   | %FileCheck %s
; RUN: not %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=trap_kernel,endpgm_saved_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	wait_kernel
	.p2align	8
	.type	wait_kernel,@function
wait_kernel:
; Each counter lowers to one fence and to nothing else; the scope and the
; ordering are what let one fence stand in for every counter.
; CHECK-LABEL: define amdgpu_kernel void @wait_kernel(
; CHECK-NEXT: entry:
; CHECK-COUNT-11: fence syncscope("agent") seq_cst
; CHECK-NOT: fence
; CHECK: ret void
; CHECK-NEXT: }
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
; CHECK-LABEL: define amdgpu_kernel void @inert_kernel(
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
; CHECK-NEXT: }
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

; The two opcodes that dispose of the wave are refused by name, each saying
; what it would have done with it.

	.globl	trap_kernel
	.p2align	8
	.type	trap_kernel,@function
trap_kernel:
; REFUSE: unsupported-instruction-form: s_trap [SOPP]
; REFUSE-SAME: enters trap handler 1, which the raised kernel does not have
	s_trap 1
	s_endpgm

	.globl	endpgm_saved_kernel
	.p2align	8
	.type	endpgm_saved_kernel,@function
endpgm_saved_kernel:
; REFUSE: unsupported-instruction-form: s_endpgm_saved [SOPP]
; REFUSE-SAME: ends the wave for a context save nothing here resumes
	s_endpgm_saved

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
	.amdhsa_kernel endpgm_saved_kernel
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
    .name:           endpgm_saved_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         endpgm_saved_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
