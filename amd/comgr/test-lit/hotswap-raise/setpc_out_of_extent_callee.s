; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=caller_kernel | %FileCheck %s

; A kernel tail-calls an outlined device helper that lives in a DIFFERENT
; function symbol, outside the kernel's own byte extent. The raiser resolves
; the callee's extent from the code object's function-symbol table and decodes
; it too, so the whole call/return CFG lifts as one function. Here the helper
; sits at a LOWER text offset than the kernel, so its block would otherwise
; become the LLVM entry despite having a predecessor (the caller's branch);
; the lift must instead route through a dedicated predecessor-free entry block.
;
; The old raiser refused any target below the kernel offset outright, so a
; successful lift that reaches the helper body proves the new following path.
; CHECK-LABEL: define amdgpu_kernel void @caller_kernel(
; The dedicated entry block branches to the kernel's real start block.
; CHECK: entry:
; CHECK: br label %[[START:bb_0x.+]]
; CHECK: [[START]]:
; The helper store (v1 = 0xBEEF0001 -> i32 -1091633151) is merged into the lift.
; CHECK: store i32 -1091633151,
; CHECK-NOT: indirectbr
; CHECK-NOT: blockaddress(
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	; Outlined helper placed first, so it is at a lower text offset than the
	; caller kernel below. It is a plain function symbol, not a kernel.
	.p2align	8
	.type	outlined_helper,@function
outlined_helper:
	v_mov_b32 v1, 0xBEEF0001
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

	.globl	caller_kernel
	.p2align	8
	.type	caller_kernel,@function
caller_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	s_get_pc_i64 s[10:11]
.Lpost:
	s_add_co_u32 s10, s10, (outlined_helper - .Lpost)
	s_add_co_ci_u32 s11, s11, ((outlined_helper - .Lpost) >> 32)
	s_set_pc_i64 s[10:11]
	v_mov_b32 v1, 0xCAFE0001
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel caller_kernel
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
    .name:           caller_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .symbol:         caller_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
