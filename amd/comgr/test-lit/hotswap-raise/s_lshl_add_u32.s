; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_lshl_add_u32_kernel \
; RUN:   | %FileCheck %s
;
; Lift fixture for the `s_lshl{1,2,3,4}_add_u32` family: `D.u = (S0.u << N)
; + S1.u`, with `SCC = unsigned carry-out of the full sum`.
;
; Regression this fixture guards (llvm-project PR #179 review). The prior
; handler computed the shift-add in i32 and left `SccHandled=false`, so the
; generic raiser writeback derived `SCC = (truncated_result != 0)`. That is
; wrong: the left shift can push bits past bit 31, so the i32 destination
; loses carry information. Example: `s_lshl1_add_u32 dst, 0x80000000, 0`
; truncates `dst` to 0 while hardware SCC must be 1 (the sum 0x100000000
; overflows 32 bits). Once the shift-add family became reachable through the
; SPE EXEC-writer gate (the allow-list change in this PR), a kernel that
; consumes SCC after `s_lshl*_add_u32` could branch incorrectly.
;
; The fix computes the shift-add once in i64, truncates for the destination,
; and sets SCC from `wide > 0xFFFFFFFF` (i.e. bits [63:32] nonzero).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_lshl_add_u32_kernel
	.p2align	8
	.type	s_lshl_add_u32_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @s_lshl_add_u32_kernel(
s_lshl_add_u32_kernel:                  ; @s_lshl_add_u32_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15

; The shift-add is materialised in i64 and SCC is the carry-out comparison
; against 0xFFFFFFFF (= 4294967295), NOT a nonzero-test of the truncated i32
; result. The negative guards the pre-fix shape that derived SCC from the
; truncated destination value.
; CHECK: %lshl1add_wide = add i64 %{{[^,]+}}, %{{[^,]+}}
; CHECK: %lshl1add = trunc i64 %lshl1add_wide to i32
; CHECK: %lshl1add_scc = icmp ugt i64 %lshl1add_wide, 4294967295
; CHECK-NOT: icmp ne i32 %lshl1add, 0
	s_lshl1_add_u32 s2, s2, s3

; The SCC consumer reads the carry bit we just stored.
; CHECK: %csel = select i1 %{{[^,]+}}
	s_cselect_b32 s2, s2, ttmp9
	v_add_nc_u32_e32 v1, s2, v0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_lshl_add_u32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           s_lshl_add_u32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         s_lshl_add_u32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
