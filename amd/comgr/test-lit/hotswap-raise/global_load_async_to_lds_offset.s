; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --emit-ir=global_load_async_to_lds_offset_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR

; global_load_async_to_lds_b32 offset form: offset applied to both global and LDS GEPs.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 4
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gptr_off{{[0-9]*}} = getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i64 16
; IR: %async_lptr_off{{[0-9]*}} = getelementptr i8, ptr addrspace(3) %lds_ptr{{[0-9]*}}, i64 16
; IR: %async_gload{{[0-9]*}} = load i32, ptr addrspace(1) %async_gptr_off{{[0-9]*}}, align 4
; IR: store i32 %async_gload{{[0-9]*}}, ptr addrspace(3) %async_lptr_off{{[0-9]*}}, align 4
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_async_to_lds_offset_kernel
	.p2align	8
	.type	global_load_async_to_lds_offset_kernel,@function
global_load_async_to_lds_offset_kernel: ; @global_load_async_to_lds_offset_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_wait_kmcnt 0x0
	global_load_async_to_lds_b32 v1, v0, s[0:1] offset:16 scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_async_to_lds_offset_kernel
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
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
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           global_load_async_to_lds_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_async_to_lds_offset_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
