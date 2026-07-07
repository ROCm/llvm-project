; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_prefetch_b8_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR

; global_prefetch_b8: gfx1250 global.prefetch lift; refused with no gfx942 equivalent.
; STDERR: transpiler: FLAT: global_prefetch_b8
; STDERR-SAME: gfx1250 VMEM-prefetch unit
; STDERR-SAME: amdgcn.global.prefetch
; STDERR-SAME: HasVmemPrefInsts
; STDERR: raise_cli: kernel 'global_prefetch_b8_kernel' failed to raise:
; STDERR-SAME: global_prefetch_b8
; STDERR-SAME: [FLAT]

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=global_prefetch_b8_kernel 2>&1 | %FileCheck %s --check-prefix=IR

; IR: %saddr_vaddr{{[0-9]*}} = add i64
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %{{[0-9]+}}, i32 8)
; IR: %saddr_vaddr{{[0-9]*}} = add i64
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %prefetch_addr{{[0-9]*}} = getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i64 256
; IR: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %prefetch_addr{{[0-9]*}}, i32 8)
; IR: declare void @llvm.amdgcn.global.prefetch(ptr addrspace(1) captures(none), i32 immarg)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_prefetch_b8_kernel
	.p2align	8
	.type	global_prefetch_b8_kernel,@function
global_prefetch_b8_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_kmcnt 0x0
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	global_prefetch_b8 v0, s[0:1] offset:256 scope:SCOPE_SE
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_prefetch_b8_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           global_prefetch_b8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_prefetch_b8_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
