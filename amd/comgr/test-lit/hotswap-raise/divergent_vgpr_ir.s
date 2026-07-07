; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --emit-ir=divergent_vgpr_kernel 2>/dev/null | %FileCheck %s

; Lift divergent v_cmpx/exec-masked VGPR writes to per-lane predicated control flow.
; CHECK-LABEL: define amdgpu_kernel void @divergent_vgpr_kernel(
; CHECK:       %[[LANE_LO:[^ ]+]] = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK-NEXT:  %[[LANE_ID:[^ ]+]] = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %[[LANE_LO]])
; CHECK:       %[[LANE_IDX:[^ ]+]] = zext i32 %[[LANE_ID]] to i64
; CHECK-NEXT:  %[[LANE_MOD:[^ ]+]] = and i64 %[[LANE_IDX]], 63
; CHECK-NEXT:  %[[AT_LANE:[^ ]+]] = lshr i64 -1, %[[LANE_MOD]]
; CHECK-NEXT:  %[[EXEC_BIT:[^ ]+]] = and i64 %[[AT_LANE]], 1
; CHECK-NEXT:  %[[ACTIVE:[^ ]+]] = icmp ne i64 %[[EXEC_BIT]], 0
; CHECK-NEXT:  br i1 %[[ACTIVE]], label %[[DO:[^ ,]+]], label %[[SKIP:[^ ,]+]]
; CHECK:       %{{[^ ]+}} = icmp ult i32 %tid, 16
; CHECK:       %cmpx_exec = and i64 -1, %{{[^ ]+}}
; CHECK:       %[[AT_LANE2:[^ ]+]] = lshr i64 %cmpx_exec, %{{[^ ]+}}
; CHECK-NEXT:  %[[EXEC_BIT2:[^ ]+]] = and i64 %[[AT_LANE2]], 1
; CHECK-NEXT:  %[[ACTIVE2:[^ ]+]] = icmp ne i64 %[[EXEC_BIT2]], 0
; CHECK-NEXT:  br i1 %[[ACTIVE2]], label %[[DO2:[^ ,]+]], label %[[SKIP2:[^ ,]+]]
; CHECK:       {{%[^ ]+}} = phi i32 [ 170, %[[DO2]] ], [ 204, %[[SKIP]] ]
; CHECK:       %and64 = and i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK:       %[[WAVE_MASK_AND64:[^ ]+]] = and i1 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK-NEXT:  %[[WAVE_MASK_EXEC:[^ ]+]] = call i64 @llvm.amdgcn.ballot.i64(i1 %[[WAVE_MASK_AND64]])
; CHECK:       %[[AT_LANE3:[^ ]+]] = lshr i64 %[[WAVE_MASK_EXEC]], %{{[^ ]+}}
; CHECK:       br i1 %{{[^ ]+}}, label %[[DO3:[^ ,]+]], label %[[SKIP3:[^ ,]+]]
; CHECK:       %[[VAL_PHI:[^ ]+]] = phi i32 [ 187, %[[DO3]] ], [ %{{[^ ]+}}, %[[SKIP2]] ]
; CHECK:       phi i32 [ %vlshl, %{{[^ ,]+}} ], [ %tid, %{{[^ ,]+}} ]
; CHECK:       %[[FINAL_AT_LANE:[^ ]+]] = lshr i64 -1, %{{[^ ]+}}
; CHECK-NEXT:  %[[FINAL_BIT:[^ ]+]] = and i64 %[[FINAL_AT_LANE]], 1
; CHECK-NEXT:  %[[FINAL_ACTIVE:[^ ]+]] = icmp ne i64 %[[FINAL_BIT]], 0
; CHECK-NEXT:  br i1 %[[FINAL_ACTIVE]], label %[[FINAL_DO:[^ ,]+]], label %{{[^ ,]+}}
; CHECK:       [[FINAL_DO]]:
; CHECK-NEXT:    store i32 %[[VAL_PHI]], ptr addrspace(1) %{{[^ ]+}}, align 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	divergent_vgpr_kernel
	.p2align	8
	.type	divergent_vgpr_kernel,@function
divergent_vgpr_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0xcc
	v_cmpx_lt_u32_e64 exec, v0, 16
	v_mov_b32 v1, 0xAA
	s_mov_b64 exec, -1
	v_cmpx_ge_u32_e64 exec, v0, 16
	v_cmp_lt_u32_e64 s[4:5], v0, 32
	s_and_b64 exec, exec, s[4:5]
	v_mov_b32 v1, 0xBB
	s_mov_b64 exec, -1
	
	s_nop 0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel divergent_vgpr_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 4
		.amdhsa_float_denorm_mode_32 3
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
    .max_flat_workgroup_size: 1024
    .name:           divergent_vgpr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         divergent_vgpr_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
