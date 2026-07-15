; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=mbcnt_saveexec_exec_elect_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WN
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=mbcnt_saveexec_exec_elect_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=MODREP

; mbcnt-fed s_and_saveexec_b32 EXEC projection. Both projections lift, but at
; different EXEC widths:
;   - wave-native packs two distinct source waves per target wave, so it
;     projects the mask into target-width (i64) EXEC storage;
;   - modulo-replication maps one source wave per target wave, so the
;     lane-relative mask is correct per replica at source width (i32).
; The classifier picks the projection-aware rewrite; neither is refused.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	mbcnt_saveexec_exec_elect_kernel
	.p2align	8
	.type	mbcnt_saveexec_exec_elect_kernel,@function
; WN-LABEL: define amdgpu_kernel void @mbcnt_saveexec_exec_elect_kernel(
mbcnt_saveexec_exec_elect_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	v_mbcnt_hi_u32_b32 v1, exec_hi, v1
; The v_cmp records a per-lane ballot shadow on its SGPR destination;
; s_and_saveexec_b32 reads it back and ANDs it into EXEC storage: i64
; (target width) under wave-native, i32 (source width) under modrep.
	v_cmp_lt_u32_e64 s2, v1, 16
; MODREP-LABEL: define amdgpu_kernel void @mbcnt_saveexec_exec_elect_kernel(
; WN: %new_exec = and i64 {{.+}}, %{{.+}}
; MODREP: %new_exec = and i32 {{.+}}, %{{.+}}
	s_and_saveexec_b32 s3, s2
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_wait_storecnt 0
	s_or_b32 exec_lo, exec_lo, s3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_saveexec_exec_elect_kernel
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
    .name:           mbcnt_saveexec_exec_elect_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         mbcnt_saveexec_exec_elect_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
