; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=nofunc_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR

; A statically resolvable s_set_pc_i64 target that lands outside the kernel's
; own extent AND outside every function symbol (here, in a padding gap between
; the sized kernel and the next symbol) is a boundary violation: there is no
; callee extent to follow, so the lift is refused rather than decoding stray
; bytes.
; STDERR: outside the selected kernel extent and any known function symbol
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	nofunc_kernel
	.p2align	8
	.type	nofunc_kernel,@function
nofunc_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	s_get_pc_i64 s[10:11]
.Lpost:
	s_add_co_u32 s10, s10, (.Lfar - .Lpost)
	s_add_co_ci_u32 s11, s11, ((.Lfar - .Lpost) >> 32)
	s_set_pc_i64 s[10:11]
	v_mov_b32 v1, 0xCAFE0001
	s_endpgm
.Lkernend:
	; Bound the kernel symbol so the padding below is outside its extent.
	.size	nofunc_kernel, .Lkernend-nofunc_kernel
	; Padding gap with NO function symbol; .Lfar points into it.
	.p2align	6
	.fill 16, 4, 3214868480
.Lfar:
	v_mov_b32 v1, 0xBEEF0001
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel nofunc_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 22
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
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           nofunc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .symbol:         nofunc_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
