; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=ds_load_b96_kernel 2>/dev/null | %FileCheck %s

; Lift ds_store_b96/ds_load_b96 to <3 x i32> addrspace(3) store/load.
; CHECK-LABEL: define amdgpu_kernel void @ds_load_b96_kernel(
; CHECK: store <3 x i32> %{{[^,]+}}, ptr addrspace(3) %{{[^,]+}}
; CHECK: %ds_ld{{[0-9]*}} = load <3 x i32>, ptr addrspace(3) %{{[^,]+}}
; CHECK-NOT: load <2 x i32>, ptr addrspace(3)
; CHECK-NOT: load <4 x i32>, ptr addrspace(3)
; CHECK-NOT: store <2 x i32>, {{.*}}ptr addrspace(3)
; CHECK-NOT: store <4 x i32>, {{.*}}ptr addrspace(3)
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.write

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_load_b96_kernel
	.p2align	8
	.type	ds_load_b96_kernel,@function
ds_load_b96_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_mul_u32_u24_e32 v1, 3, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	v_or_b32_e32 v2, 0xaa000000, v0
	v_or_b32_e32 v3, 0xbb000000, v0
	v_or_b32_e32 v4, 0xcc000000, v0
	v_lshlrev_b32_e32 v0, 2, v1
	ds_store_b96 v0, v[2:4]
	s_wait_dscnt 0
	
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b96 v[2:4], v0
	s_wait_dscnt 0
	
	s_wait_kmcnt 0x0
	global_store_b96 v0, v[2:4], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_load_b96_kernel
		.amdhsa_group_segment_fixed_size 768
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 768
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           ds_load_b96_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         ds_load_b96_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
