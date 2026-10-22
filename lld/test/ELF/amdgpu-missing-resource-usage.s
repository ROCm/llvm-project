# REQUIRES: amdgpu

## A call target with no resource metadata must be diagnosed. An undefined
## callee, or a defined function with no .amdgpu.info record, used to hit
## "missing resource usage after alias resolution".

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK-DAG: error: AMDGPU: function 'missing' is undefined and has no resource usage metadata; called from 'caller'
# CHECK-DAG: error: AMDGPU: function 'defined_without_info' has no .amdgpu.info resource usage metadata; called from 'caller'

	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	caller
	.p2align	6
	.type	caller,@function
caller:
	s_setpc_b64 s[30:31]
.Lcaller_end:
	.size	caller, .Lcaller_end-caller

	.globl	defined_without_info
	.p2align	6
	.type	defined_without_info,@function
defined_without_info:
	s_setpc_b64 s[30:31]
.Ldefined_without_info_end:
	.size	defined_without_info, .Ldefined_without_info_end-defined_without_info

	.amdgpu_info caller
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_call missing
		.amdgpu_call defined_without_info
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
