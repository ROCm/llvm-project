; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=smem_kernarg_const_delta_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; SMEM kernarg load at a constant delta tracked via Entry+Const provenance.
; CHECK-LABEL: define amdgpu_kernel void @smem_kernarg_const_delta_kernel(
; CHECK-SAME: ptr addrspace(4) byref([24 x i8]) align 16 %kargs
; CHECK: call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
; CHECK: load i32, ptr addrspace(1) %{{[^,]+}}, align 4
; CHECK: add i32 %{{[^ ,]+}}, 16
; CHECK-DAG: phi i32 [ %smem_load{{[0-9]*}}, %{{[a-zA-Z_0-9]+}}
; CHECK-DAG: phi i32 [ %smem_load{{[0-9]*}}, %{{[a-zA-Z_0-9]+}}
; CHECK-NOT: phi i32 [ i32 0, %{{[a-zA-Z_0-9]+}}
; CHECK-NOT: phi i32 [ i32 undef, %{{[a-zA-Z_0-9]+}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	smem_kernarg_const_delta_kernel
	.p2align	8
	.type	smem_kernarg_const_delta_kernel,@function
smem_kernarg_const_delta_kernel:        ; @smem_kernarg_const_delta_kernel
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[8:9], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s5, s4
	s_add_u32  s0, s0, 0x10
	s_addc_u32 s1, s1, 0
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	s_mov_b32 s5, s2
	s_mov_b32 s10, s3
	
	s_add_co_i32 s0, s4, s6
	v_dual_mov_b32 v0, s5 :: v_dual_lshlrev_b32 v3, 3, v0
	s_add_co_i32 s0, s0, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v1, s10 :: v_dual_mov_b32 v2, s0
	global_store_b96 v3, v[0:2], s[8:9]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_kernarg_const_delta_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 11
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
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         12
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           smem_kernarg_const_delta_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .symbol:         smem_kernarg_const_delta_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
