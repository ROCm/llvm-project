; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=flat_prefetch_b8_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for FLAT `flat_prefetch_b8` for targets
; that do not support prefetch. 
;
; STDERR-COUNT-3: transpiler: FLAT: flat_prefetch_b8 has no equivalent on the compilation target 
; STDERR: kernel 'flat_prefetch_b8_kernel' failed to raise: unsupported-instruction-form: flat_prefetch_b8 [FLAT]

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=flat_prefetch_b8_kernel 2>&1 | %FileCheck %s --check-prefix=IR
;
; Lift fixture for FLAT `flat_prefetch_b8` — the same-target
; (gfx1250 -> gfx1250) intrinsic-emit path.
;
; Get the 64 bit address directly from a pair of vgprs
; IR: %[[VADDR64:.+]] = inttoptr i64 %{{.*}} to ptr{{$}}
; IR: call void @llvm.amdgcn.flat.prefetch(ptr %[[VADDR64]], i32 0)
;
; Add the vgpr 32 bit offset to the sgpr address
; IR: %[[SADDR64_OFF:.+]] = add i64 %{{.*}}, %{{.*}}
; IR: %[[SADDR64_OFF_CAST:.+]] = inttoptr i64 %[[SADDR64_OFF]] to ptr{{$}}
; IR: call void @llvm.amdgcn.flat.prefetch(ptr %[[SADDR64_OFF_CAST]], i32 8)
;
; Add the vgpr 32 bit offset and the sgpr address, then do a gep of the offset 
; IR: %[[SADDR64_OFF2:.+]] = add i64 %{{.*}}, %{{.*}}
; IR: %[[SADDR64_OFF_CAST2:.+]] = inttoptr i64 %[[SADDR64_OFF2]] to ptr{{$}}
; IR: %[[SADDR64_OFF_CAST_GEP2:.+]] = getelementptr i8, ptr %[[SADDR64_OFF_CAST2]], i64 128 
; IR: call void @llvm.amdgcn.flat.prefetch(ptr %[[SADDR64_OFF_CAST_GEP2]], i32 9)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	flat_prefetch_b8_kernel
	.p2align	8
	.type	flat_prefetch_b8_kernel,@function
flat_prefetch_b8_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b64_e32 v[0:1], 1
	s_wait_kmcnt 0x0
	flat_prefetch_b8 v[0:1]
	flat_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	flat_prefetch_b8 v0, s[0:1] offset:128 th:TH_LOAD_NT scope:SCOPE_SE
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flat_prefetch_b8_kernel
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
      - { .address_space:  generic, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           flat_prefetch_b8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         flat_prefetch_b8_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
