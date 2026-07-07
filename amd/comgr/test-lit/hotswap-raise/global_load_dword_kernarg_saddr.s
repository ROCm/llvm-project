; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_load_dword_kernarg_saddr_kernel 2>/dev/null | %FileCheck %s

; global_load_b32 with kernarg-segment SADDR + kernarg provenance.
; CHECK-LABEL: define amdgpu_kernel void @global_load_dword_kernarg_saddr_kernel(ptr addrspace(4) byref([24 x i8]) align 16 %kargs)
; CHECK: %kernarg_ptr = call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
; CHECK: ptrtoint ptr addrspace(4) %kernarg_ptr to i64
; CHECK: inttoptr i64 %{{.*}} to ptr addrspace(1)
; CHECK: getelementptr i8, ptr addrspace(1) %{{.*}}, i64 2
; CHECK: load float, ptr addrspace(1) %{{.*}}, align 4
; CHECK: store i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_dword_kernarg_saddr_kernel
	.p2align	8
	.type	global_load_dword_kernarg_saddr_kernel,@function
global_load_dword_kernarg_saddr_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_mov_b32_e32 v61, 0
	global_load_b32 v1, v61, s[0:1] offset:2
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_dword_kernarg_saddr_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
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
      - { .offset:         0, .size:          12, .value_kind:     by_value }
      - { .actual_access:  write_only, .address_space:  global, .offset:        16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           global_load_dword_kernarg_saddr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         global_load_dword_kernarg_saddr_kernel.kd
    .vgpr_count:     62
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
