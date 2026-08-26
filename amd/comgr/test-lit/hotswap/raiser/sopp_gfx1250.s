; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=waits_kernel,xcnt_kernel,idle_kernel,setprio_kernel,waitalu_kernel \
; RUN:   --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   | %FileCheck %s --check-prefix=GFX125
; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=waits_kernel,xcnt_kernel,idle_kernel,waitalu_kernel \
; RUN:   --isa=gfx1250 --target-isa=gfx942 \
; RUN:   | %FileCheck %s --check-prefix=GFX942
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=setprio_kernel \
; RUN:   --isa=gfx1250 --target-isa=gfx942 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PRIO-CROSS
; RUN: not %hotswap_transpile_cli %t.hsaco --emit-ir=monitor_sleep_kernel \
; RUN:   --isa=gfx1250 --target-isa=gfx1250 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SLEEP

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	waits_kernel
	.p2align	8
	.type	waits_kernel,@function

; GFX125-LABEL: define amdgpu_kernel void @waits_kernel(
; GFX942-LABEL: define amdgpu_kernel void @waits_kernel(
waits_kernel:
; GFX125: call void @llvm.amdgcn.s.wait.loadcnt(i16 0)
; GFX125-NEXT: call void @llvm.amdgcn.s.wait.storecnt(i16 0)
; GFX125-NEXT: call void @llvm.amdgcn.s.wait.dscnt(i16 0)
; GFX125-NEXT: call void @llvm.amdgcn.s.wait.kmcnt(i16 0)
; GFX942: call void @llvm.amdgcn.s.waitcnt(i32 0)
	s_wait_kmcnt 0
	s_clause 1
	s_delay_alu instid0(VALU_DEP_1)
	s_wait_asynccnt 0
	s_wait_tensorcnt 0
	s_incperflevel 0
	s_decperflevel 0
	s_ttracedata
	s_ttracedata_imm 0
	s_icache_inv
	s_code_end
; GFX125-NEXT: ret void
; GFX942-NEXT: ret void
	s_endpgm

	.globl	xcnt_kernel
	.p2align	8
	.type	xcnt_kernel,@function

; GFX125-LABEL: define amdgpu_kernel void @xcnt_kernel(
; GFX942-LABEL: define amdgpu_kernel void @xcnt_kernel(
xcnt_kernel:
; GFX125-NOT: llvm.amdgcn.s.wait
; GFX942-NOT: llvm.amdgcn.s.wait
	s_wait_xcnt 0
; GFX125: ret void
; GFX942: ret void
	s_endpgm

	.globl	idle_kernel
	.p2align	8
	.type	idle_kernel,@function

; GFX125-LABEL: define amdgpu_kernel void @idle_kernel(
; GFX942-LABEL: define amdgpu_kernel void @idle_kernel(
idle_kernel:
; GFX125: call void @llvm.amdgcn.s.wait.loadcnt(i16 0)
; GFX125-NEXT: call void @llvm.amdgcn.s.wait.storecnt(i16 0)
; GFX125-NEXT: call void @llvm.amdgcn.s.wait.dscnt(i16 0)
; GFX125-NEXT: call void @llvm.amdgcn.s.wait.kmcnt(i16 0)
; GFX942: call void @llvm.amdgcn.s.waitcnt(i32 0)
	s_wait_idle
; GFX125-NEXT: ret void
; GFX942-NEXT: ret void
	s_endpgm

	.globl	setprio_kernel
	.p2align	8
	.type	setprio_kernel,@function

; GFX125-LABEL: define amdgpu_kernel void @setprio_kernel(
setprio_kernel:
; GFX125: call void @llvm.amdgcn.s.setprio(i16 2)
; PRIO-CROSS: unsupported-wave-priority: s_setprio [SOPP]
	s_setprio 2
; GFX125-NEXT: call void @llvm.amdgcn.s.setprio.inc.wg(i16 1)
	s_setprio_inc_wg 1
; GFX125-NEXT: ret void
	s_endpgm

	.globl	waitalu_kernel
	.p2align	8
	.type	waitalu_kernel,@function

; GFX125-LABEL: define amdgpu_kernel void @waitalu_kernel(
; GFX942-LABEL: define amdgpu_kernel void @waitalu_kernel(
waitalu_kernel:
; GFX125-NOT: llvm.amdgcn.s.wait
; GFX942-NOT: llvm.amdgcn.s.wait
	s_wait_alu depctr_va_vdst(0)
; GFX125: ret void
; GFX942: ret void
	s_endpgm

	.globl	monitor_sleep_kernel
	.p2align	8
	.type	monitor_sleep_kernel,@function

monitor_sleep_kernel:
; SLEEP: unsupported-opcode: s_monitor_sleep [SOPP]
	s_monitor_sleep 0
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
	.amdhsa_kernel setprio_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel waitalu_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel monitor_sleep_kernel
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
    .name:           setprio_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         setprio_kernel.kd
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
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           monitor_sleep_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         monitor_sleep_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
