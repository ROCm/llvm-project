; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --emit-ir=divergent_exec_kernel 2>/dev/null | %FileCheck %s

; v_cmpx/s_and_b64 exec scalar EXEC-writers projected to wave-native masks/.
; CHECK-LABEL: define amdgpu_kernel void @divergent_exec_kernel(
; CHECK:       lshr i64 -1, %{{[^ ]+}}
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4
; CHECK:       %cmpx_exec = and i64 -1, %{{[^ ]+}}
; CHECK:       lshr i64 %cmpx_exec, %{{[^ ]+}}
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4
; CHECK:       %cmpx_exec{{[0-9]+}} = and i64 -1, %{{[^ ]+}}
; CHECK:       %and64 = and i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK:       %[[WAVE_MASK_AND64:[^ ]+]] = and i1 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK-NEXT:  %[[WAVE_MASK_EXEC:[^ ]+]] = call i64 @llvm.amdgcn.ballot.i64(i1 %[[WAVE_MASK_AND64]])
; CHECK:       lshr i64 %[[WAVE_MASK_EXEC]], %{{[^ ]+}}
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	divergent_exec_kernel
	.p2align	8
	.type	divergent_exec_kernel,@function
divergent_exec_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v1, 0xaa
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[4:5], s[0:1], 0, v[2:3]
	v_mov_b32_e32 v2, 0xbb
	global_store_dword v[4:5], v3, off
	s_waitcnt vmcnt(0)
	v_cmpx_lt_u32_e64 exec, v0, 16
	global_store_dword v[4:5], v1, off
	s_waitcnt vmcnt(0)
	s_mov_b64 exec, -1
	v_cmpx_ge_u32_e64 exec, v0, 16
	v_cmp_lt_u32_e64 s[4:5], v0, 32
	s_and_b64 exec, exec, s[4:5]
	global_store_dword v[4:5], v2, off
	s_waitcnt vmcnt(0)
	s_mov_b64 exec, -1
	
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel divergent_exec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 8
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
    .name:           divergent_exec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         divergent_exec_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
