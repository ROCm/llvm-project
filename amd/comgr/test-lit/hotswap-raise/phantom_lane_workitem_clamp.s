; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=phantom_lane_workitem_clamp_kernel \
; RUN:   | %FileCheck %s

; phantom-lane workitem.id clamp of inactive lanes to 0.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	phantom_lane_workitem_clamp_kernel
	.p2align	8
	.type	phantom_lane_workitem_clamp_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @phantom_lane_workitem_clamp_kernel(
phantom_lane_workitem_clamp_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
; CHECK: %[[LANE:[A-Za-z0-9._]+]] = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %{{[A-Za-z0-9._]+}})
; CHECK: %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: %tid_is_real_lane = icmp ult i32 %[[LANE]], 32
; CHECK: %tid_phantom_clamp = select i1 %tid_is_real_lane, i32 %tid, i32 0
	v_lshlrev_b32 v1, 2, v0
; CHECK: %vlshl = shl i32 %tid_phantom_clamp, 2
; CHECK: %[[VOFF:[A-Za-z0-9._]+]] = sext i32 %vlshl to i64
; CHECK: %[[VADDR:[A-Za-z0-9._]+]] = add i64 %{{[A-Za-z0-9._]+}}, %[[VOFF]]
; The per-lane store address is frozen (cross-widening undef-address
; neutralisation, rocm-systems#157) before the pointer is materialised.
; CHECK: %mem_addr_frozen = freeze i64 %[[VADDR]]
; CHECK: %[[PTR:[0-9]+]] = inttoptr i64 %mem_addr_frozen to ptr addrspace(1)
; CHECK: store i32 %tid_phantom_clamp, ptr addrspace(1) %[[PTR]]
	global_store_b32 v1, v0, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel phantom_lane_workitem_clamp_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           phantom_lane_workitem_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         phantom_lane_workitem_clamp_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
