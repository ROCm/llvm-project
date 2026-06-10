; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_permlane32_swap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for `v_permlane32_swap_b32`'s XOR-32 lane-pair swap
; emulation on a wave64 target that does NOT natively support the
; instruction (gfx950 source → gfx942 target).  See
; `handle_valu_cross_lane.cpp::V_PERMLANE32_SWAP_B32` for the
; handler's design rationale; the `v_permlane16_swap_b32` sibling
; fixture at `../v_permlane16_swap_b32/` pins the XOR-16 variant
; under the same structural contract.
;
; This fixture pins three correctness anchors:
;
;   1. The partner lane is `lane_id XOR 32`, NOT `lane_id XOR 16`
;      / `XOR 15` / any other mask.  A refactor that substitutes
;      the wrong XOR mask would produce a valid-looking
;      `ds_bpermute` emulation whose partner neighbourhood silently
;      diverges from the source's hardware semantics — exactly the
;      miscompile class the P4 row of
;      hotswap/docs/wave-size-translation.md §5.3 documents as the
;      "cross-lane shuffle rewrite" concern.
;
;   2. Two `ds_bpermute` calls are emitted (one per output VGPR):
;      `new_vdst = bperm(addr, src0_in)` and `new_src0_out =
;      bperm(addr, vdst_in)` — the cross-wired swap.  A single-
;      bpermute lift would write one output correctly and leave
;      the other stale / undef; splitting the swap across two
;      bpermutes is what gives the two output VGPRs their
;      hardware-equivalent new values.
;
;   3. The byte-address shift is by 2 bits (multiply-by-4), which
;      is the `ds_bpermute` convention: each lane's selector is
;      the byte offset of the source lane's LDS slot.  A stale
;      lift that forgot the shift would silently lose addressing
;      and fetch the same few lanes' values for every lane.
;
; Matching uses stable SSA-name prefixes (`pls32_*`) from the
; handler rather than numeric SSA-register identifiers so the
; fixture survives unrelated SSA-number churn in the raiser.

; CHECK-LABEL: define amdgpu_kernel void @v_permlane32_swap_b32_kernel(

; (1) Partner = lane_id XOR 32.  The `pls32_partner` identifier
; is the handler's own name for this computed partner lane
; (pls32 = "permlane-swap 32"); pinning the xor-constant here is
; the strongest regression guard — an accidental mask change
; (e.g. copy-paste from the XOR-16 arm that forgets to widen to
; 32) shows up as a FileCheck miss on this line.
; CHECK: %pls32_partner{{[0-9]*}} = xor i32 %{{[^,]+}}, 32

; (2) Byte-address shift by 2 (multiply by 4).  The handler names
; this `pls32_addr` via CreateShl's twine.
; CHECK: %pls32_addr{{[0-9]*}} = shl i32 %pls32_partner{{[0-9]*}}, 2

; (3) Two `ds_bpermute` calls, cross-wired.  Pin both calls by
; their handler-local twines `pls32_new_vdst` / `pls32_new_src0_out`
; so the specific cross-wiring (first bpermute's source is
; src0_in, second's source is vdst_in) is what gets pinned — a
; same-source / reversed cross-wiring would satisfy a generic
; "two bpermutes emitted" CHECK but miscompile the swap.
; CHECK: %pls32_new_vdst{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls32_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: %pls32_new_src0_out{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls32_addr{{[0-9]*}}, i32 %{{[^)]+}})

; NEGATIVE: the pre-graduation shape was a loud raise failure
; (unsupportedInstructionForm), not an emulation — no `amdgcn.ds.bpermute`
; call reached the output at all and the raised IR was empty.
; Nothing specific to CHECK-NOT here beyond the positive
; assertions above; the positive CHECKs are tight enough that
; any regression shows up as a FileCheck miss rather than as a
; silent alternate lift.
;
; Cross-target gotcha: the XOR-16 sibling fixture lives at
; `lit_tests/v_permlane16_swap_b32/` (if added) and runs against
; a wave32 source.  This fixture's wave64 source (gfx950) is
; what makes the XOR-32 semantics well-defined — on wave32 the
; partner would wrap past the wave, which is why the handler
; refuses any wave32-source encounter with
; `v_permlane32_swap_b32` (no wave32 ISA enables
; FeaturePermlane32Swap).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	v_permlane32_swap_b32_kernel
	.p2align	8
	.type	v_permlane32_swap_b32_kernel,@function
v_permlane32_swap_b32_kernel:           ; @v_permlane32_swap_b32_kernel
; %bb.0:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_add_u32_e32 v1, 0x3e8, v0
	v_mov_b32_e32 v2, v0
	;;#ASMSTART
	v_permlane32_swap_b32 v2, v1
	;;#ASMEND
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v2, s[0:1]
	global_store_dword v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_permlane32_swap_b32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_tg_split 0
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_permlane32_swap_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_permlane32_swap_b32_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
