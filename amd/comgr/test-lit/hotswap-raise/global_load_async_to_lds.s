; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --emit-ir=global_load_async_to_lds_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR

; global_load_async_to_lds_b{8,32,64,128}: gfx942 load+LDS-store expansion vs gfx1250 intrinsic passthrough.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 4
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_lds_inb{{[0-9]*}} = icmp ult i32 {{.*}}, 65536
; IR: br i1 %async_lds_inb{{[0-9]*}}
; IR: %async_gload{{[0-9]*}} = load i32, ptr addrspace(1) %{{[0-9]+}}, align 4
; IR: store i32 %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 4
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 8
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <2 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 8
; IR: store <2 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 8
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 16
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <4 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 16
; IR: store <4 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 16
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %voff_zext{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load i8, ptr addrspace(1) %{{[0-9]+}}, align 1
; IR: store i8 %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 1
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b8
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b32
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b64
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b128

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx1250 --emit-ir=global_load_async_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SAME

; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b32(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b64(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b128(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b8(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_async_to_lds_kernel
	.p2align	8
	.type	global_load_async_to_lds_kernel,@function
global_load_async_to_lds_kernel:        ; @global_load_async_to_lds_kernel
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshl_add_u32 v1, v0, 2, 0x600
	s_wait_kmcnt 0x0
	global_load_async_to_lds_b32 v1, v0, s[4:5] scale_offset
	v_lshl_add_u32 v1, v0, 3, 0x400
	global_load_async_to_lds_b64 v1, v0, s[6:7] scale_offset
	v_lshlrev_b32_e32 v1, 4, v0
	global_load_async_to_lds_b128 v1, v0, s[8:9] scale_offset
	v_add_nc_u32_e32 v1, 0x700, v0
	global_load_async_to_lds_b8 v1, v0, s[10:11]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_async_to_lds_kernel
		.amdhsa_group_segment_fixed_size 1856
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1856
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_async_to_lds_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         global_load_async_to_lds_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
