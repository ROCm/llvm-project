; REQUIRES: comgr-has-hotswap-transpile

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=waits_kernel,sleep_kernel,monitor_sleep_kernel,wakeup_kernel \
; RUN:   --target-isa=gfx942 \
; RUN:   | %FileCheck %s --check-prefixes=GFX9,SLEEP,MONITOR-SLEEP,WAKEUP
; RUN: not %hotswap_transpile_cli %t.hsaco \
; RUN:   --emit-ir=setprio_kernel,setprio_inc_wg_kernel \
; RUN:   --target-isa=gfx942 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PRIO-CROSS
; RUN: %hotswap_transpile_cli %t.hsaco --isa=gfx1250 \
; RUN:   --dump-decoded=vgpr_msb_kernel \
; RUN:   | %FileCheck %s --check-prefix=MSB-DECODE
; RUN: %hotswap_transpile_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:   --emit-ir=vgpr_msb_kernel | %FileCheck %s --check-prefix=MSB-IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	waits_kernel
	.p2align	8
	.type	waits_kernel,@function

; GFX9-LABEL: define amdgpu_kernel void @waits_kernel(
waits_kernel:
; GFX9: fence syncscope("agent") seq_cst
	s_wait_loadcnt 1
; GFX9-NEXT: fence syncscope("agent") seq_cst
	s_wait_storecnt 2
; GFX9-NEXT: fence syncscope("agent") seq_cst
	s_wait_dscnt 3
; GFX9-NEXT: fence syncscope("agent") seq_cst
	s_wait_kmcnt 4
; GFX9-NEXT: fence syncscope("agent") seq_cst
	s_wait_loadcnt_dscnt 5
; GFX9-NEXT: fence syncscope("agent") seq_cst
	s_wait_storecnt_dscnt 6
; GFX9-NEXT: fence syncscope("agent") seq_cst
	s_wait_idle
; GFX9-NOT: fence
	s_wait_xcnt 0
	s_wait_alu depctr_va_vdst(0)
	s_nop 0
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
; GFX9-NEXT: ret void
	s_endpgm

	.globl	setprio_kernel
	.p2align	8
	.type	setprio_kernel,@function

setprio_kernel:
; PRIO-CROSS: unsupported-wave-priority: s_setprio [SOPP]
	s_setprio 2
	s_endpgm

	.globl	setprio_inc_wg_kernel
	.p2align	8
	.type	setprio_inc_wg_kernel,@function

setprio_inc_wg_kernel:
; PRIO-CROSS: unsupported-wave-priority: s_setprio_inc_wg [SOPP]
	s_setprio_inc_wg 1
	s_endpgm

	.globl	sleep_kernel
	.p2align	8
	.type	sleep_kernel,@function

; SLEEP-LABEL: define amdgpu_kernel void @sleep_kernel(
sleep_kernel:
	s_sleep 0
; SLEEP: ret void
; SLEEP-NEXT: }
	s_endpgm

	.globl	monitor_sleep_kernel
	.p2align	8
	.type	monitor_sleep_kernel,@function

; MONITOR-SLEEP-LABEL: define amdgpu_kernel void @monitor_sleep_kernel(
monitor_sleep_kernel:
	s_monitor_sleep 0
; MONITOR-SLEEP: ret void
; MONITOR-SLEEP-NEXT: }
	s_endpgm

	.globl	wakeup_kernel
	.p2align	8
	.type	wakeup_kernel,@function

; WAKEUP-LABEL: define amdgpu_kernel void @wakeup_kernel(
wakeup_kernel:
	s_wakeup
; WAKEUP: ret void
; WAKEUP-NEXT: }
	s_endpgm

	.globl	vgpr_msb_kernel
	.p2align	8
	.type	vgpr_msb_kernel,@function

; MSB-DECODE: S_SET_VGPR_MSB  s_set_vgpr_msb 64
; MSB-DECODE: S_SET_VGPR_MSB  s_set_vgpr_msb 0x4004
; MSB-IR-LABEL: define amdgpu_kernel void @vgpr_msb_kernel(
vgpr_msb_kernel:
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	s_set_vgpr_msb 0x40
	v_mov_b32_e32 v1, v0
	s_set_vgpr_msb 0x4004
; MSB-IR: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 4
	global_store_dword v[2:3], v1, off
; MSB-IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel waits_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel setprio_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel setprio_inc_wg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sleep_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel monitor_sleep_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel wakeup_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel vgpr_msb_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 258
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
    .name:           setprio_inc_wg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         setprio_inc_wg_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sleep_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sleep_kernel.kd
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
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           wakeup_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         wakeup_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           vgpr_msb_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         vgpr_msb_kernel.kd
    .vgpr_count:     258
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
