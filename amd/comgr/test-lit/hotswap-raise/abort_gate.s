; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --emit-ir=unknown_exec_writer_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR

; Refuse unmodeled EXEC-writer (s_flbit_i32_b32 into exec_lo) via the pre-translation abort gate.
; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: 's_flbit_i32_b32'
; STDERR-SAME: writes EXEC
; STDERR-SAME: routesExecThroughStoreExec
; STDERR: raise_cli: kernel 'unknown_exec_writer_kernel' failed to raise:
; STDERR-SAME: s_flbit_i32_b32
; STDERR-SAME: SPE-unmodeled-EXEC-writer

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	unknown_exec_writer_kernel
	.p2align	8
	.type	unknown_exec_writer_kernel,@function
unknown_exec_writer_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_flbit_i32_b32 exec_lo, 0x12345678
	s_mov_b64 exec, -1
	
	s_nop 0
	v_lshlrev_b32_e32 v1, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v1, v0, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel unknown_exec_writer_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
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
    .name:           unknown_exec_writer_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         unknown_exec_writer_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
