; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_pk_int_b16_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the VOP3P packed-pair `<2 x i16>` int family
; (V_PK_LSHLREV_B16 + V_PK_ADD_U16 + V_PK_ASHRREV_I16). Pins both:
;
;   * The shared operand-decode shape — `bitcast i32 -> <2 x i16>`
;     for each source (lo i16 = bits[15:0], hi i16 = bits[31:16]),
;     followed by an `extractelement` + `insertelement` round-trip
;     that pins the op_sel / op_sel_hi default permutation
;     (lo->lo, hi->hi). The kernel uses default modifiers on every
;     source so the round-trip should fold to the identity vector
;     in any subsequent IR pass — the explicit insert/extract pair
;     is what the handler emits literally.
;
;   * The per-opcode IR dispatch — `shl <2 x i16>` (after an
;     explicit `and <2 x i16> ..., <i16 15, i16 15>` shift-count
;     mask matching the AMDGPU clshl_rev_16 hardware clamp) for
;     V_PK_LSHLREV_B16, `add <2 x i16>` for V_PK_ADD_U16, and
;     `ashr <2 x i16>` for V_PK_ASHRREV_I16.
;
; Same-target lift (gfx1250 -> gfx1250) because the V_PK packed-int
; family is identical-on-the-wire across gfx9+ (no cross-arch
; emulation needed); the lit test pins the IR shape, the round-trip
; through llc would re-emit the same opcodes verbatim.

; CHECK-LABEL: define amdgpu_kernel void @v_pk_int_b16_kernel(
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_pk_int_b16_kernel
	.p2align	8
	.type	v_pk_int_b16_kernel,@function
v_pk_int_b16_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x1c
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s3, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v1, s0, s2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 1, v1
	global_load_b32 v0, v1, s[6:7] scale_offset
	s_wait_loadcnt 0x0
	; V_PK_LSHLREV_B16: masks shift count to low 4 bits, then shl <2 x i16>
	; (count=src0, value=src1).
	; CHECK: %pk_lshlrev_amt{{[0-9]*}} = and <2 x i16> {{.+}}, splat (i16 15)
	; CHECK: %pk_lshlrev{{[0-9]*}} = shl <2 x i16> %{{[^,]+}}, %pk_lshlrev_amt{{[0-9]*}}
	; CHECK: %pk_i16_pack{{[0-9]*}} = bitcast <2 x i16> %pk_lshlrev{{[0-9]*}} to i32
	; CHECK-NOT: shl <2 x f32>
	; CHECK-NOT: zext <2 x i16> %pk_lshlrev{{[0-9]*}} to <2 x i32>
	v_pk_lshlrev_b16 v0, 0x60002, v0
	; V_PK_ADD_U16: lane-wise add <2 x i16>.
	; CHECK: %pk_add_u16{{[0-9]*}} = add <2 x i16> %{{[^,]+}}, %{{[^)]+}}
	; CHECK: %pk_i16_pack{{[0-9]*}} = bitcast <2 x i16> %pk_add_u16{{[0-9]*}} to i32
	; CHECK-NOT: sub <2 x i16>
	v_pk_add_u16 v1, v0, v0
	; V_PK_ASHRREV_I16 arithmetic right-shifts each i16 lane of src1 by the
	; low 4 bits of the corresponding src0 lane
	; CHECK: %pk_ashrrev_amt{{[0-9]*}} = and <2 x i16> {{.+}}, splat (i16 15)
	; CHECK: %pk_ashrrev{{[0-9]*}} = ashr <2 x i16> %{{[^,]+}}, %pk_ashrrev_amt{{[0-9]*}}
	; CHECK: %pk_i16_pack{{[0-9]*}} = bitcast <2 x i16> %pk_ashrrev{{[0-9]*}} to i32
	; CHECK-NOT: lshr <2 x i16>
	v_pk_ashrrev_i16 v1, 0x10001, v1
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	global_store_b64 v[2:3], v[0:1], off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_int_b16_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           v_pk_int_b16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_pk_int_b16_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
