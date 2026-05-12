; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=group_segment_fixed_size_attr_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Regression fence for the static-LDS-size propagation in `raiser.cpp`.
; Without this attribute, every lifted kernel with static LDS would
; get `.group_segment_fixed_size: 0` in its HSACO KD, causing every
; LDS access to return zero (the segment is unallocated from the
; hardware's perspective).
;
; The fixture's source `__shared__ int lds[1024]` yields a source
; `.group_segment_fixed_size: 4096` in the gfx1250 HSACO's KD.  The
; raiser must propagate that into an `amdgpu-lds-size` function
; attribute on the lifted kernel so `AMDGPUMachineFunctionInfo`
; picks it up in the cross-targeted gfx942 codegen.
;
; We assert:
;   1) The lifted kernel function carries `"amdgpu-lds-size"="4096,4096"`
;      (min=max=4096 since the source's static size is known exactly).
;   2) The source's `__shared__ int lds[1024]` lowered to at least one
;      addrspace(3) ds_write / ds_load pair in the IR (i.e. the LDS is
;      actually used — otherwise the attribute would be a no-op and
;      the fixture would pass vacuously).

; IR-LABEL: define amdgpu_kernel void @group_segment_fixed_size_attr_kernel(
; IR-DAG: store {{.*}}, ptr addrspace(3)
; IR-DAG: load {{.*}}, ptr addrspace(3)
; IR: attributes #{{[0-9]+}} = { {{.*}}"amdgpu-lds-size"="4096,4096"{{.*}} }

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	group_segment_fixed_size_attr_kernel
	.p2align	8
	.type	group_segment_fixed_size_attr_kernel,@function
group_segment_fixed_size_attr_kernel:   ; @group_segment_fixed_size_attr_kernel
; %bb.0:
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_lshlrev_b32 v2, 2, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 31, v1
	ds_store_b32 v2, v0
	s_wait_dscnt 0x0
	; wave barrier
	v_lshlrev_b32_e32 v1, 2, v1
	ds_load_b32 v1, v1
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel group_segment_fixed_size_attr_kernel
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
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
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           group_segment_fixed_size_attr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         group_segment_fixed_size_attr_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
