; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_pk_bf16_siblings_kernel 2>/dev/null | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --write-hsaco=%t.out --kernel=v_pk_bf16_siblings_kernel 2>&1 | %FileCheck %s --check-prefix=PIPE
;
; Lift test for the remaining gfx1250 packed BF16 VOP3P arithmetic siblings:
; `v_pk_mul_bf16`, `v_pk_min_num_bf16`, and `v_pk_max_num_bf16`.

; IR-LABEL: define amdgpu_kernel void @v_pk_bf16_siblings_kernel(
; IR: [[MUL_SRC:%[^ ]+]] = bitcast i32 {{[^ ]+}} to <2 x bfloat>
; IR-DAG: [[MUL_LO:%[^ ]+]] = extractelement <2 x bfloat> [[MUL_SRC]], i64 0
; IR-DAG: [[MUL_HI:%[^ ]+]] = extractelement <2 x bfloat> [[MUL_SRC]], i64 1
; IR: [[MUL_NEG_LO:%[^ ]+]] = fneg bfloat [[MUL_HI]]
; IR: insertelement <2 x bfloat> {{.*}}, bfloat [[MUL_NEG_LO]], i64 0
; IR: insertelement <2 x bfloat> {{.*}}, bfloat [[MUL_LO]], i64 1
; IR: fmul <2 x bfloat>
; IR: call <2 x bfloat> @llvm.minimumnum.v2bf16(
; IR: call <2 x bfloat> @llvm.maximumnum.v2bf16(
; IR: [[MIN:%[^ ]+]] = call <2 x bfloat> @llvm.minimumnum.v2bf16(
; IR: [[MIN_CLAMP_LO:%[^ ]+]] = call <2 x bfloat> @llvm.maximumnum.v2bf16(<2 x bfloat> [[MIN]],
; IR: call <2 x bfloat> @llvm.minimumnum.v2bf16(<2 x bfloat> [[MIN_CLAMP_LO]],
; IR: [[MAX:%[^ ]+]] = call <2 x bfloat> @llvm.maximumnum.v2bf16(
; IR: [[MAX_CLAMP_LO:%[^ ]+]] = call <2 x bfloat> @llvm.maximumnum.v2bf16(<2 x bfloat> [[MAX]],
; IR: call <2 x bfloat> @llvm.minimumnum.v2bf16(<2 x bfloat> [[MAX_CLAMP_LO]],
; IR-NOT: unsupported instruction

; PIPE: raise_cli: wrote
; PIPE-SAME: v_pk_bf16_siblings_kernel

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_pk_bf16_siblings_kernel
	.p2align	8
	.type	v_pk_bf16_siblings_kernel,@function
v_pk_bf16_siblings_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, 0x3f803f80
	v_mov_b32_e32 v2, 0x40004000
	;;#ASMSTART
	v_pk_mul_bf16 v3, v1, v2 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1]
	v_pk_min_num_bf16 v4, v1, v2
	v_pk_max_num_bf16 v5, v1, v2
	v_pk_min_num_bf16 v6, v1, v2 clamp
	v_pk_max_num_bf16 v7, v1, v2 clamp
	;;#ASMEND
	global_store_b32 v0, v3, s[0:1] scale_offset
	v_lshlrev_b32_e32 v8, 2, v0
	global_store_b32 v8, v4, s[0:1] scale_offset
	v_lshlrev_b32_e32 v8, 3, v0
	global_store_b32 v8, v5, s[0:1] scale_offset
	v_lshlrev_b32_e32 v8, 4, v0
	global_store_b32 v8, v6, s[0:1] scale_offset
	v_lshlrev_b32_e32 v8, 5, v0
	global_store_b32 v8, v7, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_bf16_siblings_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 9
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
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: v_pk_bf16_siblings_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_pk_bf16_siblings_kernel.kd
    .vgpr_count: 9
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
