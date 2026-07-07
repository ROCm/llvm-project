; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_store_dwordx4_kernel 2>/dev/null | %FileCheck %s

; Raw-buffer store v4i32 lift with no scratch-alloca fallback.
; CHECK-LABEL: define amdgpu_kernel void @buffer_store_dwordx4_kernel(
; CHECK: call void @llvm.amdgcn.raw.buffer.store.v4i32(
; CHECK-NOT: alloca {{.*}}addrspace(5)
; CHECK-NOT: addrspacecast {{.*}}to ptr addrspace(5)
; CHECK-NOT: addrspacecast ptr addrspace(5){{.*}}to ptr
; CHECK-NOT: select i1 {{.*}}ptr addrspace(5)
; CHECK-NOT: @llvm.amdgcn.flat.store
; CHECK-NOT: store {{.*}}, ptr addrspace(5)
; CHECK-NOT: @llvm.amdgcn.raw.buffer.store.i32(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_store_dwordx4_kernel
	.p2align	8
	.type	buffer_store_dwordx4_kernel,@function
buffer_store_dwordx4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_dual_mov_b32 v3, 0xcafebabe :: v_dual_lshlrev_b32 v0, 4, v0
	v_mov_b32_e32 v2, 0xdeadbeef
	v_mov_b32_e32 v4, 0xfeedface
	v_mov_b32_e32 v5, 0xbadc0ffe
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	s_wait_kmcnt 0x0
	buffer_store_b128 v[2:5], v0, s[0:3], null offen scope:SCOPE_DEV
	s_wait_storecnt 0
	
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_dwordx4_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 4
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_dwordx4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_dwordx4_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
