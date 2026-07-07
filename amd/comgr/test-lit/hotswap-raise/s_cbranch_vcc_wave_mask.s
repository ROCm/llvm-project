; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 \
; RUN:     --emit-ir=s_cbranch_vcc_wave_mask_kernel \
; RUN:   | %FileCheck %s --check-prefix=VCC
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 \
; RUN:     --emit-ir=s_cbranch_vccnz_wave_mask_kernel \
; RUN:   | %FileCheck %s --check-prefix=VCCNZ
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 \
; RUN:     --emit-ir=sop2_wave_mask_scc_kernel \
; RUN:   | %FileCheck %s --check-prefix=SCC
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 \
; RUN:     --emit-ir=sop2_wave_mask_exec_scc_kernel \
; RUN:   | %FileCheck %s --check-prefix=EXEC
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 \
; RUN:     --emit-ir=sop2_wave_mask_scalar_scc_kernel \
; RUN:   | %FileCheck %s --check-prefix=SCALAR \
; RUN:     --implicit-check-not=_scc_ballot

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

; s_cbranch_vcc/vccnz wave-mask ballot branch lowering (vs per-lane SCC path).
; VCC-LABEL: define amdgpu_kernel void @s_cbranch_vcc_wave_mask_kernel(
	.globl	s_cbranch_vcc_wave_mask_kernel
	.p2align	8
	.type	s_cbranch_vcc_wave_mask_kernel,@function
s_cbranch_vcc_wave_mask_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
; VCC: [[BALLOT:%vcc_ballot[0-9]*]] = call i64 @llvm.amdgcn.ballot.i64(
; VCC: [[ZERO:%vcc_is_zero[0-9]*]] = icmp eq i64 [[BALLOT]], 0
; VCC: br i1 [[ZERO]]
	s_cbranch_vccz .L_vcc_zero
	v_mov_b32_e32 v1, 1
	s_branch .L_vcc_store
.L_vcc_zero:
	v_mov_b32_e32 v1, 0
.L_vcc_store:
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

; s_cbranch_vccnz: nonzero form of the full-wave VCC test.
; VCCNZ-LABEL: define amdgpu_kernel void @s_cbranch_vccnz_wave_mask_kernel(
	.globl	s_cbranch_vccnz_wave_mask_kernel
	.p2align	8
	.type	s_cbranch_vccnz_wave_mask_kernel,@function
s_cbranch_vccnz_wave_mask_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; VCCNZ: [[BALLOT:%vcc_ballot[0-9]*]] = call i64 @llvm.amdgcn.ballot.i64(
; VCCNZ: [[ZERO:%vcc_is_zero[0-9]*]] = icmp eq i64 [[BALLOT]], 0
; VCCNZ: [[NZ:%vcc_nz[0-9]*]] = xor i1 [[ZERO]], true
; VCCNZ: br i1 [[NZ]]
	s_cbranch_vccnz .L_vcc_nonzero
	v_mov_b32_e32 v1, 0
	s_branch .L_vccnz_store
.L_vcc_nonzero:
	v_mov_b32_e32 v1, 1
.L_vccnz_store:
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

; SOP2 and/or/xor/andn2 on lane masks: SCC from whether any lane in the mask is set.
; SCC-LABEL: define amdgpu_kernel void @sop2_wave_mask_scc_kernel(
	.globl	sop2_wave_mask_scc_kernel
	.p2align	8
	.type	sop2_wave_mask_scc_kernel,@function
sop2_wave_mask_scc_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCC: [[AND:%wave_mask_and[0-9]*]] = and i1
; SCC: [[BALLOT:%wave_mask_and_scc_ballot[0-9]*]] = call i64 @llvm.amdgcn.ballot.i64(
; SCC-SAME: i1 [[AND]])
; SCC: [[NONZERO:%wave_mask_and_scc_nonzero[0-9]*]] = icmp ne i64 [[BALLOT]],
; SCC-SAME: 0
; SCC: br i1 [[NONZERO]]
	s_and_b32 s2, s2, s3
	s_cbranch_scc1 .L_scc_and_nonzero
	v_mov_b32_e32 v1, 0
	s_branch .L_scc_or
.L_scc_and_nonzero:
	v_mov_b32_e32 v1, 1
.L_scc_or:
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCC: [[OR:%wave_mask_or[0-9]*]] = or i1
; SCC: [[OR_BALLOT:%wave_mask_or_scc_ballot[0-9]*]] = call i64 @llvm.amdgcn.ballot.i64(
; SCC-SAME: i1 [[OR]])
; SCC: [[OR_NONZERO:%wave_mask_or_scc_nonzero[0-9]*]] = icmp ne i64 [[OR_BALLOT]],
; SCC-SAME: 0
; SCC: br i1 [[OR_NONZERO]]
	s_or_b32 s2, s2, s3
	s_cbranch_scc1 .L_scc_or_nonzero
	v_mov_b32_e32 v1, 2
	s_branch .L_scc_xor
.L_scc_or_nonzero:
	v_mov_b32_e32 v1, 3
.L_scc_xor:
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCC: [[XOR:%wave_mask_xor[0-9]*]] = xor i1
; SCC: [[XOR_BALLOT:%wave_mask_xor_scc_ballot[0-9]*]] = call i64 @llvm.amdgcn.ballot.i64(
; SCC-SAME: i1 [[XOR]])
; SCC: [[XOR_NONZERO:%wave_mask_xor_scc_nonzero[0-9]*]] = icmp ne i64 [[XOR_BALLOT]],
; SCC-SAME: 0
; SCC: br i1 [[XOR_NONZERO]]
	s_xor_b32 s2, s2, s3
	s_cbranch_scc1 .L_scc_xor_nonzero
	v_mov_b32_e32 v1, 4
	s_branch .L_scc_and64
.L_scc_xor_nonzero:
	v_mov_b32_e32 v1, 5
.L_scc_andn2:
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCC-DAG: %wave_mask_andn2{{[0-9]*}} = and i1
; SCC-DAG: %wave_mask_andn2_scc_ballot{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(
; SCC-DAG: %wave_mask_andn2_scc_nonzero{{[0-9]*}} = icmp ne i64 %wave_mask_andn2_scc_ballot{{[0-9]*}}, 0
; SCC-DAG: br i1 %wave_mask_andn2_scc_nonzero{{[0-9]*}}
	s_andn2_b32 s2, s2, s3
	s_cbranch_scc1 .L_scc_andn2_nonzero
	v_mov_b32_e32 v1, 6
	s_branch .L_scc_and64
.L_scc_andn2_nonzero:
	v_mov_b32_e32 v1, 7
.L_scc_and64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCC-DAG: %wave_mask_and64{{[0-9]*}} = and i1
; SCC-DAG: %wave_mask_and64_scc_ballot{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(
; SCC-DAG: %wave_mask_and64_scc_nonzero{{[0-9]*}} = icmp ne i64 %wave_mask_and64_scc_ballot{{[0-9]*}}, 0
; SCC-DAG: br i1 %wave_mask_and64_scc_nonzero{{[0-9]*}}
	s_and_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scc_and64_nonzero
	v_mov_b32_e32 v1, 8
	s_branch .L_scc_store
.L_scc_and64_nonzero:
	v_mov_b32_e32 v1, 9
.L_scc_or64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCC-DAG: %wave_mask_or64{{[0-9]*}} = or i1
; SCC-DAG: %wave_mask_or64_scc_ballot{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(
; SCC-DAG: %wave_mask_or64_scc_nonzero{{[0-9]*}} = icmp ne i64 %wave_mask_or64_scc_ballot{{[0-9]*}}, 0
; SCC-DAG: br i1 %wave_mask_or64_scc_nonzero{{[0-9]*}}
	s_or_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scc_or64_nonzero
	v_mov_b32_e32 v1, 10
	s_branch .L_scc_xor64
.L_scc_or64_nonzero:
	v_mov_b32_e32 v1, 11
.L_scc_xor64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCC-DAG: %wave_mask_xor64{{[0-9]*}} = xor i1
; SCC-DAG: %wave_mask_xor64_scc_ballot{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(
; SCC-DAG: %wave_mask_xor64_scc_nonzero{{[0-9]*}} = icmp ne i64 %wave_mask_xor64_scc_ballot{{[0-9]*}}, 0
; SCC-DAG: br i1 %wave_mask_xor64_scc_nonzero{{[0-9]*}}
	s_xor_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scc_xor64_nonzero
	v_mov_b32_e32 v1, 12
	s_branch .L_scc_andn2_64
.L_scc_xor64_nonzero:
	v_mov_b32_e32 v1, 13
.L_scc_andn2_64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCC-DAG: %wave_mask_andn2_64{{[0-9]*}} = and i1
; SCC-DAG: %wave_mask_andn2_64_scc_ballot{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(
; SCC-DAG: %wave_mask_andn2_64_scc_nonzero{{[0-9]*}} = icmp ne i64 %wave_mask_andn2_64_scc_ballot{{[0-9]*}}, 0
; SCC-DAG: br i1 %wave_mask_andn2_64_scc_nonzero{{[0-9]*}}
	s_andn2_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scc_andn2_64_nonzero
	v_mov_b32_e32 v1, 14
	s_branch .L_scc_store
.L_scc_andn2_64_nonzero:
	v_mov_b32_e32 v1, 15
.L_scc_store:
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

; SOP2 writes to EXEC propagate the wave mask and derive SCC from the ballot.
; EXEC-LABEL: define amdgpu_kernel void @sop2_wave_mask_exec_scc_kernel(
	.globl	sop2_wave_mask_exec_scc_kernel
	.p2align	8
	.type	sop2_wave_mask_exec_scc_kernel,@function
sop2_wave_mask_exec_scc_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; EXEC: [[AND:%wave_mask_and64[0-9]*]] = and i1
; EXEC: %wave_mask_exec{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(i1 [[AND]])
; EXEC: [[BALLOT:%wave_mask_and64_scc_ballot[0-9]*]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[AND]])
; EXEC: [[NONZERO:%wave_mask_and64_scc_nonzero[0-9]*]] = icmp ne i64 [[BALLOT]], 0
; EXEC: br i1 [[NONZERO]]
	s_and_b64 exec, exec, vcc
	s_cbranch_scc1 .L_exec_scc_nonzero
	v_mov_b32_e32 v1, 0
	s_branch .L_exec_scc_store
.L_exec_scc_nonzero:
	v_mov_b32_e32 v1, 1
.L_exec_scc_store:
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

; Complementing SOP2 ops derive SCC from the full scalar result while still tracking the lane mask.
; SCALAR-LABEL: define amdgpu_kernel void @sop2_wave_mask_scalar_scc_kernel(
	.globl	sop2_wave_mask_scalar_scc_kernel
	.p2align	8
	.type	sop2_wave_mask_scalar_scc_kernel,@function
sop2_wave_mask_scalar_scc_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCALAR-DAG: %wave_mask_orn2 = or i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i32 %orn2, 0
	s_orn2_b32 s2, s2, s3
	s_cbranch_scc1 .L_scalar_orn2_64
	v_mov_b32_e32 v1, 0
.L_scalar_orn2_64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCALAR-DAG: %wave_mask_orn2_64 = or i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i64 %orn2_64, 0
	s_orn2_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scalar_nand
	v_mov_b32_e32 v1, 1
.L_scalar_nand:
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCALAR-DAG: %wave_mask_nand = xor i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i32 %nand, 0
	s_nand_b32 s2, s2, s3
	s_cbranch_scc1 .L_scalar_nand64
	v_mov_b32_e32 v1, 2
.L_scalar_nand64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCALAR-DAG: %wave_mask_nand64 = xor i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i64 %nand64, 0
	s_nand_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scalar_nor
	v_mov_b32_e32 v1, 3
.L_scalar_nor:
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCALAR-DAG: %wave_mask_nor = xor i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i32 %nor, 0
	s_nor_b32 s2, s2, s3
	s_cbranch_scc1 .L_scalar_nor64
	v_mov_b32_e32 v1, 4
.L_scalar_nor64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCALAR-DAG: %wave_mask_nor64 = xor i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i64 %nor64, 0
	s_nor_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scalar_xnor
	v_mov_b32_e32 v1, 5
.L_scalar_xnor:
	v_cmp_lt_u32_e64 s2, v0, 16
	v_cmp_ne_u32_e64 s3, v0, 31
; SCALAR-DAG: %wave_mask_xnor = xor i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i32 %xnor, 0
	s_xnor_b32 s2, s2, s3
	s_cbranch_scc1 .L_scalar_xnor64
	v_mov_b32_e32 v1, 6
.L_scalar_xnor64:
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
; SCALAR-DAG: %wave_mask_xnor64 = xor i1
; SCALAR-DAG: %{{[0-9]+}} = icmp ne i64 %xnor64, 0
	s_xnor_b64 s[2:3], exec, vcc
	s_cbranch_scc1 .L_scalar_store
	v_mov_b32_e32 v1, 7
.L_scalar_store:
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_cbranch_vcc_wave_mask_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel s_cbranch_vccnz_wave_mask_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel sop2_wave_mask_scc_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel sop2_wave_mask_exec_scc_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel sop2_wave_mask_scalar_scc_kernel
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: s_cbranch_vcc_wave_mask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: s_cbranch_vcc_wave_mask_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: s_cbranch_vccnz_wave_mask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: s_cbranch_vccnz_wave_mask_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: sop2_wave_mask_scc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 4
    .symbol: sop2_wave_mask_scc_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: sop2_wave_mask_exec_scc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: sop2_wave_mask_exec_scc_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: sop2_wave_mask_scalar_scc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 4
    .symbol: sop2_wave_mask_scalar_scc_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
