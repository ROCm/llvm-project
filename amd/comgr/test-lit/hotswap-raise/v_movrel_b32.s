; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=movrel_kernel 2>&1 | %FileCheck %s --check-prefix=IR
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=movrels_kernel 2>&1 | %FileCheck %s --check-prefix=IRS
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=movrelsd_kernel 2>&1 | %FileCheck %s --check-prefix=IRSD
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=movrel_dyn_kernel 2>&1 | %FileCheck %s --check-prefix=DYN

; Register-relative move with a raise-time-constant M0 resolves to a
; direct VGPR access:
;   v20 = 7; m0 = 3; v_movreld_b32 v10, v20  => VGPR[10+3] = v20 => v13 = 7
; The stored value 7 threads through v13, proving the index resolution.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	movrel_kernel
	.p2align	8
	.type	movrel_kernel,@function
; IR-LABEL: define amdgpu_kernel void @movrel_kernel(
movrel_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v20, 7
	s_mov_b32 m0, 3
	v_movreld_b32 v10, v20
; IR: store i32 7, ptr addrspace(1)
	global_store_b32 v0, v13, s[0:1]
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel movrel_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 21
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

; V_MOVRELS reads its *source* M0-relative: vdst = VGPR[base(vsrc)+M0].
;   v23 = 7; m0 = 3; v_movrels_b32 v10, v20 => v10 = VGPR[20+3] = v23 = 7
; The stored value 7 threads through v10 (the direct dst), proving the
; source-side index resolution.
	.text
	.globl	movrels_kernel
	.p2align	8
	.type	movrels_kernel,@function
; IRS-LABEL: define amdgpu_kernel void @movrels_kernel(
movrels_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v23, 7
	s_mov_b32 m0, 3
	v_movrels_b32 v10, v20
; IRS: store i32 7, ptr addrspace(1)
	global_store_b32 v0, v10, s[0:1]
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel movrels_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

; V_MOVRELSD is relative on *both* ends:
;   VGPR[base(vdst)+M0] = VGPR[base(vsrc)+M0].
;   v23 = 7; m0 = 3; v_movrelsd_b32 v10, v20 => VGPR[10+3] = VGPR[20+3]
;   => v13 = v23 = 7
; The stored value 7 threads through v13 (the relative dst), proving both
; the source- and dest-side index resolution.
	.text
	.globl	movrelsd_kernel
	.p2align	8
	.type	movrelsd_kernel,@function
; IRSD-LABEL: define amdgpu_kernel void @movrelsd_kernel(
movrelsd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v23, 7
	s_mov_b32 m0, 3
	v_movrelsd_b32 v10, v20
; IRSD: store i32 7, ptr addrspace(1)
	global_store_b32 v0, v13, s[0:1]
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel movrelsd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

; A data-dependent M0 has no statically-known relative index and must be
; refused loudly (stubbed), never silently mistranslated.
; DYN: failed to raise
; DYN-SAME: non-constant M0
	.text
	.globl	movrel_dyn_kernel
	.p2align	8
	.type	movrel_dyn_kernel,@function
movrel_dyn_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x8
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v20, 7
	s_mov_b32 m0, s2
	v_movreld_b32 v10, v20
	global_store_b32 v0, v13, s[0:1]
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel movrel_dyn_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 21
		.amdhsa_next_free_sgpr 3
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
    .name:           movrel_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         movrel_kernel.kd
    .vgpr_count:     21
    .wavefront_size: 32
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           movrels_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         movrels_kernel.kd
    .vgpr_count:     24
    .wavefront_size: 32
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           movrelsd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         movrelsd_kernel.kd
    .vgpr_count:     24
    .wavefront_size: 32
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           movrel_dyn_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         movrel_dyn_kernel.kd
    .vgpr_count:     21
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
