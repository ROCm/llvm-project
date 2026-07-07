; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=ds_load_2addr_b32_kernel 2>/dev/null | %FileCheck %s

; Lift ds_load_2addr_b32 and stride64 variant to two scalar i32 loads at computed offsets.
; CHECK-LABEL: define amdgpu_kernel void @ds_load_2addr_b32_kernel(
; CHECK-DAG: add i64 %{{.*}}, 16
; CHECK-DAG: add i64 %{{.*}}, 24
; CHECK-DAG: add i64 %{{.*}}, 512
; CHECK-DAG: add i64 %{{.*}}, 768
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p0, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p1, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p0{{[0-9]+}}, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p1{{[0-9]+}}, align 4
; CHECK-NOT: load <2 x i32>, ptr addrspace(3)
; CHECK-NOT: load <4 x i32>, ptr addrspace(3)
; CHECK-NOT: load i64, ptr addrspace(3)
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_load_2addr_b32_kernel
	.p2align	8
	.type	ds_load_2addr_b32_kernel,@function
ds_load_2addr_b32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	ds_load_2addr_b32 v[2:3], v1 offset0:4 offset1:6
	s_wait_dscnt 0
	
	ds_load_2addr_stride64_b32 v[4:5], v1 offset0:2 offset1:3
	s_wait_dscnt 0
	
	s_wait_kmcnt 0x0
	global_store_b128 v0, v[2:5], s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_load_2addr_b32_kernel
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 6
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
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           ds_load_2addr_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         ds_load_2addr_b32_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
