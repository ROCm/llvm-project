; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=ds_load_tr8_b64_kernel 2>/dev/null | %FileCheck %s

; Emulate ds_load_tr_b64 transpose via mbcnt/ds.bpermute lane gather plus per-byte loads.
; CHECK-LABEL: define amdgpu_kernel void @ds_load_tr8_b64_kernel(
; CHECK: %lane_lo{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK: %lane_id{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane_lo{{[0-9]*}})
; CHECK: %bp_base = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[^,]+}}, i32 %{{[^,]+}})
; CHECK: %tr8_p = inttoptr i64 %{{[^ ]+}} to ptr addrspace(3)
; CHECK: %tr8_b = load i8, ptr addrspace(3) %tr8_p, align 1
; CHECK: %tr8_pack = or i32 0, %{{[^ ]+}}
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.load.tr8.b64
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read.tr8.b64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_load_tr8_b64_kernel
	.p2align	8
	.type	ds_load_tr8_b64_kernel,@function
ds_load_tr8_b64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_lshlrev_b32_e32 v1, 3, v0
	v_add_nc_u32_e32 v3, 0x50607080, v0
	v_add_nc_u32_e32 v2, 0x10203040, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	ds_store_b64 v1, v[2:3]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_tr_b64 v[2:3], v1
	s_wait_dscnt 0
	
	s_wait_kmcnt 0x0
	global_store_b64 v0, v[2:3], s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_load_tr8_b64_kernel
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           ds_load_tr8_b64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         ds_load_tr8_b64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
