; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_load_ushort_saddr_kernel 2>/dev/null | %FileCheck %s

; global_load_u16 SADDR scale_offset: x2 scaled offset, i16 zext lift.
; CHECK-LABEL: define amdgpu_kernel void @global_load_ushort_saddr_kernel(
; CHECK:      %voff_sext = sext i32 %{{[^ ,]+}} to i64
; CHECK-NEXT: %scaled_voff = mul i64 %voff_sext, 2
; CHECK-NEXT: %saddr_vaddr = add i64 %{{[^ ,]+}}, %scaled_voff
; The per-lane global address is frozen (cross-widening undef-address
; neutralisation, rocm-systems#157) before the pointer is materialised.
; CHECK:      %mem_addr_frozen = freeze i64 %saddr_vaddr
; CHECK:      %{{[^ ]+}} = inttoptr i64 %mem_addr_frozen to ptr addrspace(1)
; CHECK:      %gload_sub = load i16, ptr addrspace(1) %{{[^ ,]+}}, align 2
; CHECK:      %{{[^ ]+}} = zext i16 %gload_sub to i32
; CHECK-NOT:  mul i64 %voff_sext, 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_ushort_saddr_kernel
	.p2align	8
	.type	global_load_ushort_saddr_kernel,@function
global_load_ushort_saddr_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
	v_cmp_gt_i32_e32 vcc_lo, s2, v0
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_2
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_u16 v1, v0, s[0:1] scale_offset
	s_wait_loadcnt 0x0
	global_store_b32 v0, v1, s[2:3] scale_offset
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_ushort_saddr_kernel
		.amdhsa_kernarg_size 20
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_reserve_vcc 1
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
      - { .actual_access:  read_only, .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .actual_access:  write_only, .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
      - { .offset:         16, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .max_flat_workgroup_size: 1024
    .name:           global_load_ushort_saddr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         global_load_ushort_saddr_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
