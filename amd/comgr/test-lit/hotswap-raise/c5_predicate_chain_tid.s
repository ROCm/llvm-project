; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --disable-wave-native \
; RUN:     --emit-ir=c5_predicate_chain_tid_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c5_predicate_chain_tid_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR_WN

; Under --disable-wave-native the MODREP narrow-O1 classifier refuses this
; Class 5 workitem-id predicate chain, then the analysis-triggered
; ThreadLoopProjection retry rescues it (TLP iterates per source wave, so the
; predicate is evaluated source-wave-scoped); the kernel raises instead of
; aborting. Under the wave-native default the refusal is suppressed outright.
; STDERR: retrying kernel 'c5_predicate_chain_tid_kernel' under ThreadLoopProjection
; STDERR-SAME: after C5 predicate-chain refusal
; STDERR: thread-loop fallback trigger:
; STDERR-SAME: icmp ult
; STDERR-SAME: compile-time constant 16
; STDERR-SAME: W_s-1=31
; STDERR: selected ThreadLoopProjection
; STDERR: define amdgpu_kernel void @c5_predicate_chain_tid_kernel(
; IR_WN-LABEL: define amdgpu_kernel void @c5_predicate_chain_tid_kernel(
; IR_WN: call i32 @llvm.amdgcn.workitem.id.x()

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c5_predicate_chain_tid_kernel
	.p2align	8
	.type	c5_predicate_chain_tid_kernel,@function
c5_predicate_chain_tid_kernel:          ; @c5_predicate_chain_tid_kernel
; %bb.0:
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
	v_mad_u32 v1, s0, s4, v0
	v_cmp_lt_u32_e64 s0, v0, 16
	
	v_cndmask_b32_e64 v0, -1, v0, s0
	
	global_store_b32 v1, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c5_predicate_chain_tid_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           c5_predicate_chain_tid_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c5_predicate_chain_tid_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
