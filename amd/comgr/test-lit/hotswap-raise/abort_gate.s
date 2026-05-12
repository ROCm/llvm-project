; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --emit-ir=unknown_exec_writer_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; The raiser must refuse to lower a kernel that contains an
; EXEC-writing instruction whose SemOp is not declared SPE-safe
; (`routesExecThroughStoreExec` in `sem_op_attrs.cpp`). This prevents
; us from silently emitting IR for an opcode whose handler has not
; been audited against the per-lane predication assumption.
;
; The fixture pins `s_flbit_i32_b32` as the offending instruction:
; its `S_FLBIT_I32_B32` SemOp is deliberately absent from the
; attribute table because no handler has been written for that shape
; of EXEC write. Routing the dst to `exec_lo` makes the EXEC-writer
; detector flag it.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic names the instruction and explains why,
;      matching on stable substrings. We do not pin to the full
;      sentence to keep the test resilient to future rewordings.
;
; History: the diagnostic previously mentioned "SPE-modelled
; allow-list"; that wording was replaced by the attribute-based check
; ("routesExecThroughStoreExec") when P1.3 moved the allow-list into
; `sem_op_attrs.cpp`. The attribute name is the new stable substring.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: 's_flbit_i32_b32'
; STDERR-SAME: writes EXEC
; STDERR-SAME: routesExecThroughStoreExec

; The raise_cli wrapper reports the failure once more so the kerneldex
; coverage format is preserved; the mnemonic must match.
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
	;;#ASMSTART
	s_flbit_i32_b32 exec_lo, 0x12345678
	s_mov_b64 exec, -1
	
	;;#ASMEND
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
