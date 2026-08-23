; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx942 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	waits_kernel
	.p2align	8
	.type	waits_kernel,@function

; The source wait raises to a wait for every counter the target tracks, because
; counter identities do not correspond across ISA families. Targeting gfx9 that
; is the combined form. The source's per-counter values do not survive either: a
; count is a position in the source's issue order, so only zero translates. The
; scheduling hints emit nothing: the backend issues its own for the target it
; lowers to.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=waits_kernel | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @waits_kernel(
; CHECK: call void @llvm.amdgcn.s.waitcnt(i32 0)
; CHECK-NEXT: ret void

; gfx1250 dropped the combined s_waitcnt, so the same source wait has to raise
; to the split counters when that is the compilation target. The source's
; expcnt(2) does not survive as a wait: gfx1250 has no export instructions, so
; nothing can be pending in that counter there.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=waits_kernel \
; RUN:   --target-isa=gfx1250 | %FileCheck %s --check-prefix=SPLIT
; SPLIT-LABEL: define amdgpu_kernel void @waits_kernel(
; SPLIT: call void @llvm.amdgcn.s.wait.loadcnt(i16 0)
; SPLIT-NEXT: call void @llvm.amdgcn.s.wait.storecnt(i16 0)
; SPLIT-NEXT: call void @llvm.amdgcn.s.wait.dscnt(i16 0)
; SPLIT-NEXT: call void @llvm.amdgcn.s.wait.kmcnt(i16 0)
; SPLIT-NEXT: ret void

; The decoder maps each instruction onto its canonical op rather than leaving it
; unknown, which is what routes it to the arm that handles it at all.
; RUN: %hotswap_transpile_cli %t.hsaco --dump-decoded=waits_kernel \
; RUN:   | %FileCheck %s --check-prefix=DECODE
waits_kernel:
; DECODE: S_WAITCNT{{.+}}s_waitcnt
	s_waitcnt vmcnt(1) expcnt(2) lgkmcnt(3)
; DECODE: S_NOP{{.+}}s_nop
	s_nop 0
; DECODE: S_SLEEP{{.+}}s_sleep
	s_sleep 1
; DECODE: S_SETPRIO{{.+}}s_setprio
	s_setprio 0
	s_endpgm

	.globl	trap_kernel
	.p2align	8
	.type	trap_kernel,@function

; s_trap enters the trap handler and has no IR equivalent, so it refuses rather
; than raising to nothing.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=trap_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=TRAP
; TRAP: unsupported-instruction-form: s_trap [SOPP]
trap_kernel:
	s_trap 1
	s_endpgm

	.globl	sleep_mode_kernel
	.p2align	8
	.type	sleep_mode_kernel,@function

; A bounded sleep raises to nothing, but an immediate reaching past the duration
; field selects a mode the raiser does not model, so it refuses.
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=sleep_mode_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SLEEPMODE
; SLEEPMODE: unsupported-instruction-form: s_sleep [SOPP]
; SLEEPMODE-SAME: selects more than a sleep duration
sleep_mode_kernel:
	s_sleep 0x8000
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel waits_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel trap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.amdhsa_kernel sleep_mode_kernel
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
    .name:           waits_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         waits_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
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
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sleep_mode_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sleep_mode_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
