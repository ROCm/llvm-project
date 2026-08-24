; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	waits_kernel
	.p2align	8
	.type	waits_kernel,@function

; Raised for gfx1250 itself: the memory wait takes the split counters, and the
; async and tensor waits emit nothing even here, because the backend issues that
; work and pairs its own waits with it. No EXPcnt wait: gfx1250 has no export
; instructions, so nothing can be pending in the counter and the intrinsic has
; no pattern carrier on the target.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=waits_kernel | %FileCheck %s
; CHECK-LABEL: define amdgpu_kernel void @waits_kernel(
; CHECK: call void @llvm.amdgcn.s.wait.loadcnt(i16 0)
; CHECK-NEXT: call void @llvm.amdgcn.s.wait.storecnt(i16 0)
; CHECK-NEXT: call void @llvm.amdgcn.s.wait.dscnt(i16 0)
; CHECK-NEXT: call void @llvm.amdgcn.s.wait.kmcnt(i16 0)
; CHECK-NEXT: ret void

; Raised for gfx942, the memory wait collapses onto the combined form, and the
; async and tensor waits stay absent for the further reason that this target has
; no unit whose work those counters could be tracking.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=waits_kernel \
; RUN:   --target-isa=gfx942 | %FileCheck %s --check-prefix=TO942
; TO942-LABEL: define amdgpu_kernel void @waits_kernel(
; TO942: call void @llvm.amdgcn.s.waitcnt(i32 0)
; TO942-NEXT: ret void
waits_kernel:
	s_wait_kmcnt 0
	s_clause 1
	s_delay_alu instid0(VALU_DEP_1)
	s_wait_asynccnt 0
	s_wait_tensorcnt 0
	s_endpgm

	.globl	xcnt_kernel
	.p2align	8
	.type	xcnt_kernel,@function

; XCNT counts memory operations awaiting address translation. Where the wait
; belongs depends on the register assignment, which raising discards, so it
; raises to nothing on every target and the backend re-derives it for the
; assignment it makes.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=xcnt_kernel | %FileCheck %s \
; RUN:   --check-prefix=XCNT
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=xcnt_kernel \
; RUN:   --target-isa=gfx942 | %FileCheck %s --check-prefix=XCNT
; XCNT-LABEL: define amdgpu_kernel void @xcnt_kernel(
; XCNT-NOT: llvm.amdgcn.s.wait
; XCNT: ret void
xcnt_kernel:
	s_wait_xcnt 0
	s_endpgm

	.globl	idle_kernel
	.p2align	8
	.type	idle_kernel,@function

; s_wait_idle covers every counter, so it takes the target's memory wait-all and
; leaves the backend-owned counters to the backend, on both targets.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=idle_kernel | %FileCheck %s \
; RUN:   --check-prefix=IDLE
; IDLE-LABEL: define amdgpu_kernel void @idle_kernel(
; IDLE: call void @llvm.amdgcn.s.wait.loadcnt(i16 0)
; IDLE-NEXT: call void @llvm.amdgcn.s.wait.storecnt(i16 0)
; IDLE-NEXT: call void @llvm.amdgcn.s.wait.dscnt(i16 0)
; IDLE-NEXT: call void @llvm.amdgcn.s.wait.kmcnt(i16 0)
; IDLE-NEXT: ret void

; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=idle_kernel \
; RUN:   --target-isa=gfx942 | %FileCheck %s --check-prefix=IDLE942
; IDLE942-LABEL: define amdgpu_kernel void @idle_kernel(
; IDLE942: call void @llvm.amdgcn.s.waitcnt(i32 0)
; IDLE942-NEXT: ret void
idle_kernel:
	s_wait_idle
	s_endpgm



	.globl	waitalu_kernel
	.p2align	8
	.type	waitalu_kernel,@function

; The ALU counters count register hazards, so like XCNT the wait belongs
; wherever the register assignment puts it, and raises to nothing.
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=waitalu_kernel | %FileCheck %s \
; RUN:   --check-prefix=WAITALU
; RUN: %hotswap_transpile_cli %t.hsaco --emit-ir=waitalu_kernel \
; RUN:   --target-isa=gfx942 | %FileCheck %s --check-prefix=WAITALU
; WAITALU-LABEL: define amdgpu_kernel void @waitalu_kernel(
; WAITALU-NOT: llvm.amdgcn.s.wait
; WAITALU: ret void
waitalu_kernel:
	s_wait_alu depctr_va_vdst(0)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel waits_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel xcnt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel idle_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel waitalu_kernel
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
    .name:           waits_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         waits_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           xcnt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         xcnt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           idle_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         idle_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           waitalu_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         waitalu_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
