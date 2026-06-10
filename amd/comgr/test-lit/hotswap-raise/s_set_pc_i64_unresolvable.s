; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=setpc_unresolvable_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for s_set_pc_i64 — Unresolvable path. Pins the
; contractual loud-failure behaviour of the SOP1 handler in
; transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SET_PC_I64) { ... case Unresolvable: }`.
; The static analysis (transpiler/setpc_analysis.cpp) classifies any
; s_set_pc_i64 site whose source SGPR pair was not produced by a
; recognized PC chain (Pattern A) and is not consumed via a recorded
; chain-terminator (Pattern B) as Unresolvable, with a refusal
; reason. The handler converts that into a
; RaiseFailure::unsupportedInstructionForm — the user rules forbid silent
; fallbacks, so a "branch to unreachable" stub would be a regression.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (`s_set_pc_i64`) and the encoding format (`SOP1`).
;      raise_cli's failure line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>]`.
;
; History: this test was added together with the s_set_pc_i64
; handler implementation (see semop.hpp's S_SET_PC_I64 doc, and
; setpc_analysis.cpp). The mnemonic / format substrings are stable;
; rewordings of the human-readable refusal reason live in `detail`
; (RaiseFailure::detail) and are NOT printed by raise_cli, so this
; test does not depend on that wording.

; STDERR: raise_cli: kernel 'setpc_unresolvable_kernel' failed to raise:
; STDERR-SAME: s_set_pc_i64
; STDERR-SAME: [SOP1]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setpc_unresolvable_kernel
	.p2align	8
	.type	setpc_unresolvable_kernel,@function
setpc_unresolvable_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_mov_b32 s10, 0
	s_mov_b32 s11, 0
	s_set_pc_i64 s[10:11]
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b32 v0, v0, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setpc_unresolvable_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
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
    .name:           setpc_unresolvable_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         setpc_unresolvable_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
