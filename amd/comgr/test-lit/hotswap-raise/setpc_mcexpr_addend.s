; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=setpc_mcexpr_addend_kernel | %FileCheck %s
;
; s_set_pc_i64 Pattern A (statically resolvable intra-kernel branch) with the
; chain addend written as a label-difference, which the disassembler surfaces
; as an MCExpr literal rather than a plain Imm. Pins that imm32 folds the
; MCExpr addend through the PC chain
;   s_get_pc_i64 -> s_add_co_u32 -> s_add_co_ci_u32
; instead of breaking the chain and forcing the site to Unresolvable.
;
; Layout:
;     0x10: s_get_pc_i64 s[10:11]
;     0x14: .Lpost  s_add_co_u32 s10, s10, lit(0x1C)
;     0x1C:         s_add_co_ci_u32 s11, s11, lit(0x0)
;     0x24:         s_set_pc_i64 s[10:11]
;     0x30: .Ltarget v_mov_b32 v1, 0xDEAD0001
;   target = .Lpost(0x14) + (.Ltarget - .Lpost)(0x1C) = 0x30 -> bb_0x30.
; Both addends exercise the MCExpr path: low lit(0x1C) and high lit(0x0).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setpc_mcexpr_addend_kernel
	.p2align	8
	.type	setpc_mcexpr_addend_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @setpc_mcexpr_addend_kernel(
setpc_mcexpr_addend_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	s_get_pc_i64 s[10:11]
.Lpost:
	s_add_co_u32 s10, s10, (.Ltarget - .Lpost)
	s_add_co_ci_u32 s11, s11, ((.Ltarget - .Lpost) >> 32)
; CHECK: br label %bb_0x30
; CHECK-NOT: indirectbr ptr
; CHECK-NOT: unreachable
	s_set_pc_i64 s[10:11]
	v_mov_b32 v1, 0xCAFE0001
; CHECK: bb_0x30:
.Ltarget:
	v_mov_b32 v1, 0xDEAD0001

	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setpc_mcexpr_addend_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           setpc_mcexpr_addend_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         setpc_mcexpr_addend_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
