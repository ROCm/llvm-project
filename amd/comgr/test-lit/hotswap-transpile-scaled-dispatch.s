; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: env HSA_HOTSWAP_TRANSLATE_KERNEL=hotswap_transpile_scaled_kernel \
; RUN:     hotswap-transpile %t.hsaco amdgcn-amd-amdhsa--gfx1250 \
; RUN:                   amdgcn-amd-amdhsa--gfx942 \
; RUN:   | %FileCheck --check-prefix=NAMED %s
; RUN: hotswap-transpile %t.hsaco amdgcn-amd-amdhsa--gfx1250 \
; RUN:                   amdgcn-amd-amdhsa--gfx942 --use-plain-api \
; RUN:   2>&1 | %FileCheck --check-prefix=PLAIN %s
; RUN: hotswap-transpile %t.hsaco amdgcn-amd-amdhsa--gfx1250 \
; RUN:                   amdgcn-amd-amdhsa--gfx942 \
; RUN:   2>&1 | %FileCheck --check-prefix=ALLKERNELS %s
; RUN: env HSA_HOTSWAP_TRANSLATE_KERNEL=hotswap_transpile_plain_kernel \
; RUN:     hotswap-transpile %t.hsaco amdgcn-amd-amdhsa--gfx1250 \
; RUN:                   amdgcn-amd-amdhsa--gfx942 \
; RUN:   | %FileCheck --check-prefix=UNSCALED %s

; Two kernels, only one of which needs a scaled dispatch. A named request
; reports the factor; a request that cannot report it is refused.

; NAMED: RESULT_INFO: success=1
; NAMED-SAME: scaled_dispatch_factor=2
; NAMED: RESULT: SUCCESS

; PLAIN: RESULT: INVALID_ARGUMENT
; PLAIN-NOT: RESULT: SUCCESS

; ALLKERNELS: RESULT: INVALID_ARGUMENT
; ALLKERNELS: RESULT_INFO: success=0
; ALLKERNELS-SAME: fail_reason=scaled_dispatch_not_reportable
; ALLKERNELS-NOT: RESULT: SUCCESS

; UNSCALED: RESULT_INFO: success=1
; UNSCALED-SAME: scaled_dispatch_factor=1
; UNSCALED: RESULT: SUCCESS

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	hotswap_transpile_scaled_kernel
	.type	hotswap_transpile_scaled_kernel,@function
hotswap_transpile_scaled_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_bfe_u32 v2, v0, 10, 10
	v_mov_b32_e32 v3, 0xdeadbeef
	s_mov_b32 s2, -1
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	v_cmp_lt_u32_e64 s4, v2, 16
	v_cndmask_b32_e64 v3, -1, v3, s4
	buffer_store_b32 v3, v2, s[0:3], null offen
	s_endpgm

	.globl	hotswap_transpile_plain_kernel
	.type	hotswap_transpile_plain_kernel,@function
hotswap_transpile_plain_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0x2a
	s_wait_kmcnt 0x0
	global_store_b32 v1, v2, s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel hotswap_transpile_scaled_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel hotswap_transpile_plain_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  512
    .name:                     hotswap_transpile_scaled_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         hotswap_transpile_scaled_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  512
    .name:                     hotswap_transpile_plain_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         hotswap_transpile_plain_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
