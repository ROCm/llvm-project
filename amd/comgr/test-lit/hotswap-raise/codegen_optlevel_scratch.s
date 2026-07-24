; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: rm -rf %t.d0 && HSA_HOTSWAP_DUMP_DIR=%t.d0 raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --write-hsaco=%t.o0 --kernel=codegen_optlevel_regalloc_kernel 2>&1 | %FileCheck %s --check-prefix=PIPE
; RUN: %FileCheck %s --check-prefix=O0 < %t.d0/hotswap-*/codegen_optlevel_regalloc_kernel.s
; RUN: rm -rf %t.d2 && HSA_HOTSWAP_DUMP_DIR=%t.d2 raise_cli %t.hsaco --target-isa=gfx942 -O2 \
; RUN:   --write-hsaco=%t.o2 --kernel=codegen_optlevel_regalloc_kernel 2>&1 | %FileCheck %s --check-prefix=PIPE
; RUN: %FileCheck %s --check-prefix=O2 < %t.d2/hotswap-*/codegen_optlevel_regalloc_kernel.s

; The opt level passed to the pipeline drives register allocation. At O0 the
; AMDGPU backend uses RegAllocFast, which spills the whole high-pressure value
; set to the private (scratch) segment; at O2 it uses the greedy allocator,
; which keeps them in registers and leaves the private segment empty. On a
; high-pressure kernel the O0 spill can overflow the private-segment limit and
; fail codegen (rocm-systems#151), so the opt level is load-bearing here.

; PIPE: raise_cli: wrote
; O0: .amdhsa_private_segment_fixed_size {{[1-9][0-9]*}}
; O2: .amdhsa_private_segment_fixed_size 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	codegen_optlevel_regalloc_kernel
	.p2align	8
	.type	codegen_optlevel_regalloc_kernel,@function
codegen_optlevel_regalloc_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, v0
	v_add_nc_u32_e32 v3, 3, v0
	v_add_nc_u32_e32 v4, 4, v0
	v_add_nc_u32_e32 v5, 5, v0
	v_add_nc_u32_e32 v6, 6, v0
	v_add_nc_u32_e32 v7, 7, v0
	v_add_nc_u32_e32 v8, 8, v0
	v_add_nc_u32_e32 v9, 9, v0
	v_add_nc_u32_e32 v10, 10, v0
	v_add_nc_u32_e32 v11, 11, v0
	v_add_nc_u32_e32 v12, 12, v0
	v_add_nc_u32_e32 v13, 13, v0
	v_add_nc_u32_e32 v14, 14, v0
	v_add_nc_u32_e32 v15, 15, v0
	v_add_nc_u32_e32 v16, 16, v0
	v_add_nc_u32_e32 v17, 17, v0
	v_add_nc_u32_e32 v18, 18, v0
	v_add_nc_u32_e32 v19, 19, v0
	v_add_nc_u32_e32 v20, 20, v0
	v_add_nc_u32_e32 v21, 21, v0
	v_add_nc_u32_e32 v22, 22, v0
	v_add_nc_u32_e32 v23, 23, v0
	v_add_nc_u32_e32 v24, 24, v0
	v_add_nc_u32_e32 v25, 25, v0
	v_add_nc_u32_e32 v26, 26, v0
	v_add_nc_u32_e32 v27, 27, v0
	v_add_nc_u32_e32 v28, 28, v0
	v_add_nc_u32_e32 v29, 29, v0
	v_add_nc_u32_e32 v30, 30, v0
	v_add_nc_u32_e32 v31, 31, v0
	v_add_nc_u32_e32 v32, 32, v0
	v_add_nc_u32_e32 v33, 33, v0
	v_add_nc_u32_e32 v34, 34, v0
	v_add_nc_u32_e32 v35, 35, v0
	v_add_nc_u32_e32 v36, 36, v0
	v_add_nc_u32_e32 v37, 37, v0
	v_add_nc_u32_e32 v38, 38, v0
	v_add_nc_u32_e32 v39, 39, v0
	v_add_nc_u32_e32 v40, 40, v0
	v_add_nc_u32_e32 v41, 41, v0
	v_add_nc_u32_e32 v42, 42, v0
	v_add_nc_u32_e32 v43, 43, v0
	v_add_nc_u32_e32 v44, 44, v0
	v_add_nc_u32_e32 v45, 45, v0
	v_add_nc_u32_e32 v46, 46, v0
	v_add_nc_u32_e32 v47, 47, v0
	v_add_nc_u32_e32 v48, 48, v0
	v_add_nc_u32_e32 v49, 49, v0
	v_add_nc_u32_e32 v50, 50, v0
	v_add_nc_u32_e32 v51, 51, v0
	v_add_nc_u32_e32 v52, 52, v0
	v_add_nc_u32_e32 v53, 53, v0
	v_add_nc_u32_e32 v54, 54, v0
	v_add_nc_u32_e32 v55, 55, v0
	v_add_nc_u32_e32 v56, 56, v0
	v_add_nc_u32_e32 v57, 57, v0
	v_add_nc_u32_e32 v58, 58, v0
	v_add_nc_u32_e32 v59, 59, v0
	v_add_nc_u32_e32 v60, 60, v0
	v_add_nc_u32_e32 v61, 61, v0
	v_add_nc_u32_e32 v62, 62, v0
	v_add_nc_u32_e32 v63, 63, v0
	v_mul_lo_u32 v2, v2, v3
	v_mul_lo_u32 v2, v2, v4
	v_mul_lo_u32 v2, v2, v5
	v_mul_lo_u32 v2, v2, v6
	v_mul_lo_u32 v2, v2, v7
	v_mul_lo_u32 v2, v2, v8
	v_mul_lo_u32 v2, v2, v9
	v_mul_lo_u32 v2, v2, v10
	v_mul_lo_u32 v2, v2, v11
	v_mul_lo_u32 v2, v2, v12
	v_mul_lo_u32 v2, v2, v13
	v_mul_lo_u32 v2, v2, v14
	v_mul_lo_u32 v2, v2, v15
	v_mul_lo_u32 v2, v2, v16
	v_mul_lo_u32 v2, v2, v17
	v_mul_lo_u32 v2, v2, v18
	v_mul_lo_u32 v2, v2, v19
	v_mul_lo_u32 v2, v2, v20
	v_mul_lo_u32 v2, v2, v21
	v_mul_lo_u32 v2, v2, v22
	v_mul_lo_u32 v2, v2, v23
	v_mul_lo_u32 v2, v2, v24
	v_mul_lo_u32 v2, v2, v25
	v_mul_lo_u32 v2, v2, v26
	v_mul_lo_u32 v2, v2, v27
	v_mul_lo_u32 v2, v2, v28
	v_mul_lo_u32 v2, v2, v29
	v_mul_lo_u32 v2, v2, v30
	v_mul_lo_u32 v2, v2, v31
	v_mul_lo_u32 v2, v2, v32
	v_mul_lo_u32 v2, v2, v33
	v_mul_lo_u32 v2, v2, v34
	v_mul_lo_u32 v2, v2, v35
	v_mul_lo_u32 v2, v2, v36
	v_mul_lo_u32 v2, v2, v37
	v_mul_lo_u32 v2, v2, v38
	v_mul_lo_u32 v2, v2, v39
	v_mul_lo_u32 v2, v2, v40
	v_mul_lo_u32 v2, v2, v41
	v_mul_lo_u32 v2, v2, v42
	v_mul_lo_u32 v2, v2, v43
	v_mul_lo_u32 v2, v2, v44
	v_mul_lo_u32 v2, v2, v45
	v_mul_lo_u32 v2, v2, v46
	v_mul_lo_u32 v2, v2, v47
	v_mul_lo_u32 v2, v2, v48
	v_mul_lo_u32 v2, v2, v49
	v_mul_lo_u32 v2, v2, v50
	v_mul_lo_u32 v2, v2, v51
	v_mul_lo_u32 v2, v2, v52
	v_mul_lo_u32 v2, v2, v53
	v_mul_lo_u32 v2, v2, v54
	v_mul_lo_u32 v2, v2, v55
	v_mul_lo_u32 v2, v2, v56
	v_mul_lo_u32 v2, v2, v57
	v_mul_lo_u32 v2, v2, v58
	v_mul_lo_u32 v2, v2, v59
	v_mul_lo_u32 v2, v2, v60
	v_mul_lo_u32 v2, v2, v61
	v_mul_lo_u32 v2, v2, v62
	v_mul_lo_u32 v2, v2, v63
	global_store_b32 v0, v2, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel codegen_optlevel_regalloc_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 64
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           codegen_optlevel_regalloc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         codegen_optlevel_regalloc_kernel.kd
    .vgpr_count:     64
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
