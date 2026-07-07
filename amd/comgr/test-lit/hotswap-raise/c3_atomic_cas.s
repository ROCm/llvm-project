; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=c3_atomic_cas_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR

; Refuse non-commutative global_atomic_cmpswap_b32 (Class 3 cross-wave-replica-race) via the wave-size obstruction path.
; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-replica-race
; STDERR-SAME: cmpswap
; STDERR: NonCommutativeAtomic
; STDERR-SAME: Class 3
; STDERR: outcome: (c) refuse
; STDERR: raise_cli: kernel 'c3_atomic_cas_kernel' failed to raise:
; STDERR-SAME: cmpswap

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c3_atomic_cas_kernel
	.p2align	8
	.type	c3_atomic_cas_kernel,@function
c3_atomic_cas_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
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
	s_and_b32 s0, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s1, ttmp9, s1
	s_mul_i32 s1, s1, s0
	s_mov_b32 s0, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s0
	v_add3_u32 v0, v0, s1, 1
	global_atomic_cmpswap_b32 v2, v[0:1], s[2:3] scope:SCOPE_DEV
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c3_atomic_cas_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
    .name:           c3_atomic_cas_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c3_atomic_cas_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
