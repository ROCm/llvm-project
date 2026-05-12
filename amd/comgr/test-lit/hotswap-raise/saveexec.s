; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --emit-ir=saveexec_kernel 2>/dev/null | %FileCheck %s
;
; Audit: `s_and_saveexec_b64` routes its EXEC write through
; `storeExec` (handle_sop1.cpp). This SemOp family is the one
; EXEC-writer that is simultaneously an SGPR-pair producer — the
; handler has to (a) save the OLD EXEC into the destination SGPR
; pair, and (b) write `old_exec AND src` to EXEC. Post-mem2reg we
; cannot see the SGPR-pair save directly (the saved value may fold
; away if the kernel never reads it), but we can see the new EXEC
; SSA value (`%new_exec`) being consumed by the SPE active-bit
; computation that wraps the subsequent side-effectful store.
;
; If the S_AND_SAVEEXEC_B64 handler regressed and no longer routed
; through storeExec, the `lshr i64 <exec>, %spe_lane_mod` before the
; store would still read `-1` (the initial EXEC) instead of
; `%new_exec`, and the single-lane-gated store would silently
; execute on every lane.

; CHECK-LABEL: define amdgpu_kernel void @saveexec_kernel(

; v_cmp_lt_u32_e64 produces the narrow mask in s[4:5]. Pre-mem2reg
; that's a store to the relevant sgpr alloca; post-mem2reg it's the
; SSA name for the comparison's sign-extended i64 representation.
; CHECK:       %vcmp = icmp ult i32 %tid, 16

; s_and_saveexec_b64 writes `old_exec AND src` to EXEC. The handler
; names the new EXEC SSA value `%new_exec` — this is the audit
; signal.
; CHECK:       %new_exec = and i64 -1, %{{[^ ]+}}

; The SPE lane-active computation that follows (wrapping the
; global_store_dword) keys off %new_exec, NOT -1. This proves the
; SSA graph of EXEC reflects the saveexec write.
; CHECK:       %[[AT_LANE:[^ ]+]] = lshr i64 %new_exec, %{{[^ ]+}}
; CHECK-NEXT:  %[[BIT:[^ ]+]] = and i64 %[[AT_LANE]], 1
; CHECK-NEXT:  %[[ACTIVE:[^ ]+]] = icmp ne i64 %[[BIT]], 0
; CHECK-NEXT:  br i1 %[[ACTIVE]], label %[[DO:[^ ,]+]], label %{{[^ ,]+}}

; The store under the narrowed EXEC is the observable side-effect
; that motivates the whole audit.
; CHECK:       [[DO]]:
; CHECK-NEXT:    store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	saveexec_kernel
	.p2align	8
	.type	saveexec_kernel,@function
saveexec_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v1, 0xcc
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[2:3]
	;;#ASMSTART
	v_cmp_lt_u32_e64 s[4:5], v0, 16
	s_and_saveexec_b64 s[6:7], s[4:5]
	global_store_dword v[2:3], v1, off
	s_waitcnt vmcnt(0)
	s_mov_b64 exec, s[6:7]
	
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel saveexec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
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
    .name:           saveexec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .symbol:         saveexec_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
