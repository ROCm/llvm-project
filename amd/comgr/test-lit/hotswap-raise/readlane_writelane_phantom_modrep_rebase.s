; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=readlane_writelane_phantom_modrep_rebase_kernel \
; RUN:   | %FileCheck %s --check-prefix=REWRITE
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --disable-writelane-rewrite \
; RUN:     --emit-ir=readlane_writelane_phantom_modrep_rebase_kernel \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Under the phantom-lane ModuloReplicationProjection regime (selected by
; .max_flat_workgroup_size 32 < the gfx942 wave size 64), v_readlane_b32 /
; v_writelane_b32 must be source-wave-rebased. The default-on
; rewriteCrossLaneDivergent pass (raiser.cpp) does this: readlane becomes a
; source-wave-scoped ds.bpermute, writelane a lane-predicated select.
; --disable-writelane-rewrite pins the pre-rewrite native form, which is a
; silent wave32->wave64 miscompile under MODREP. The in-handler rebase
; (handle-valu-cross-lane.cpp) only fires for ThreadLoopProjection, so MODREP
; correctness rides entirely on the post-raise pass.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	readlane_writelane_phantom_modrep_rebase_kernel
	.p2align	8
	.type	readlane_writelane_phantom_modrep_rebase_kernel,@function
readlane_writelane_phantom_modrep_rebase_kernel:
; The source-wave lane id is synthesized once at entry via the two-step mbcnt.
; REWRITE-LABEL: define amdgpu_kernel void @readlane_writelane_phantom_modrep_rebase_kernel(
; REWRITE: [[LANELO:%.+]] = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; REWRITE: [[LANEID:%.+]] = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 [[LANELO]])
; UNCHANGED-LABEL: define amdgpu_kernel void @readlane_writelane_phantom_modrep_rebase_kernel(
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x14
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v0, s0, s4, v0
	v_mov_b32_e32 v1, 0
; readlane rebases to a source-wave-scoped ds.bpermute.
; REWRITE: [[RLBASE:%.+]] = and i32 [[LANEID]], -32
; REWRITE: [[RLLANE:%.+]] = or i32 [[RLBASE]], 5
; REWRITE: [[RLSEL:%.+]] = shl i32 [[RLLANE]], 2
; REWRITE: call i32 @llvm.amdgcn.ds.bpermute(i32 [[RLSEL]], i32 {{.+}})
; UNCHANGED: call i32 @llvm.amdgcn.readlane.i32(i32 {{.+}}, i32 5)
	v_readlane_b32 s0, v1, 5
; writelane rebases to a lane-predicated select.
; REWRITE: [[WLMOD:%.+]] = and i32 [[LANEID]], 31
; REWRITE: [[WLMASK:%.+]] = icmp eq i32 [[WLMOD]], 7
; REWRITE: select i1 [[WLMASK]], i32 {{.+}}, i32 {{.+}}
; UNCHANGED: call i32 @llvm.amdgcn.writelane.i32(i32 {{.+}}, i32 7, i32 {{.+}})
	v_writelane_b32 v1, s0, 7
; REWRITE-NOT: call i32 @llvm.amdgcn.readlane
; REWRITE-NOT: call i32 @llvm.amdgcn.writelane
; UNCHANGED-NOT: @llvm.amdgcn.ds.bpermute
	v_xor_b32_e32 v1, s0, v1
	global_store_b32 v0, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel readlane_writelane_phantom_modrep_rebase_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 32
    .name:           readlane_writelane_phantom_modrep_rebase_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         readlane_writelane_phantom_modrep_rebase_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
