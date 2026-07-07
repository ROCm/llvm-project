; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=c4_lane_dep_cmpx_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Class 4 "lane-position-dependent EXEC writes" under modulo-replication
; (hotswap/docs/wave-size-translation.md §6). The v_cmpx's LHS flows from
; v_mbcnt_lo, which makes the compare semantically "enable lanes below a
; source-wave-local lane-id threshold". WaveNative has a principled projection
; for this family; this fixture pins the opt-out MODREP refusal.
;
; This is the structural counter-example to lit_tests/cross_wave_warn,
; whose v_cmpx uses a lane-position-INDEPENDENT bounds expression
; (an 8-vs-lane-value compare with a constant bound, where the
; lane_id is not routed through a mbcnt-derived path). That warn-
; only fixture remains valid under the new gate because its
; compare is provably lane-position-independent.
;
; CLASSIFIER CONTRACT. The decoded-provenance classifier flags this
; kernel because the v_cmpx source register is defined by v_mbcnt_lo.
; This is the unsafe path: an EXEC predicate rooted in absolute
; lane-id arithmetic. Kernels that use v_mbcnt_* only for unrelated
; ds_bpermute selectors and write EXEC from bounds masks should not
; trip this fixture's refusal.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-lane-predicated-exec
; STDERR-SAME: v_cmpx

; STDERR: CmpxFromLaneId
; STDERR-SAME: Class 4
; STDERR: outcome: (c) refuse

; STDERR: raise_cli: kernel 'c4_lane_dep_cmpx_kernel' failed to raise:
; STDERR-SAME: v_cmpx

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c4_lane_dep_cmpx_kernel
	.p2align	8
	.type	c4_lane_dep_cmpx_kernel,@function
c4_lane_dep_cmpx_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s2, s[0:1], 0x14
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_and_b32 s4, ttmp6, 15
	s_add_co_i32 s3, s3, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s3, ttmp9, s3
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x0
	s_add_co_i32 s4, s4, s3
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s3, ttmp9, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u32 v0, s3, s2, v0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[2:3], v[0:1], 2, s[0:1]
	v_mov_b32_e32 v0, 1
	;;#ASMSTART
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_cmpx_lt_u32_e64 v10, 16
	global_store_b32 v[2:3], v0, off
	s_wait_storecnt 0
	s_mov_b32 exec_lo, -1
	
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c4_lane_dep_cmpx_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 11
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           c4_lane_dep_cmpx_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c4_lane_dep_cmpx_kernel.kd
    .vgpr_count:     11
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
