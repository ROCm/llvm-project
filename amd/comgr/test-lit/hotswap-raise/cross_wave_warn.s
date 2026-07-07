; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=cross_wave_writer_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Cross-wave write via v_cmpx warns but still lifts the kernel.
; CHECK-LABEL: define amdgpu_kernel void @cross_wave_writer_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cross_wave_writer_kernel
	.p2align	8
	.type	cross_wave_writer_kernel,@function
cross_wave_writer_kernel:               ; @cross_wave_writer_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 2, v0
	v_mov_b32_e32 v1, 0xaa
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]
	v_cmpx_lt_u32_e64 v0, 64
	global_store_b32 v[2:3], v1, off
	s_wait_storecnt 0
	s_mov_b32 exec_lo, -1
	
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cross_wave_writer_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
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
    .name:           cross_wave_writer_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cross_wave_writer_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
