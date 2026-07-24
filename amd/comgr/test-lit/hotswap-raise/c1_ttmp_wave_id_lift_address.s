; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:      --emit-ir=c1_ttmp_wave_id_lift_address_kernel \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:      --emit-ir=c1_ttmp_wave_id_lift_address_kernel \
; RUN:   | %FileCheck %s --check-prefix=NEG

; The canonical wave_id extraction s_bfe_u32 ttmp8, 0x50019 feeding a store
; address must lift source-wave-relative -- workitem.id.x() >> 5 (log2 of
; W_s=32), not >> 6 -- so the address stays correct after WaveNative packs two
; source wave32 warps into one target wave64, and must raise rather than
; refuse. Guards the canonical rescue (isCanonicalWaveIdBfe, handle-sop2.cpp).

; IR-LABEL: define amdgpu_kernel void @c1_ttmp_wave_id_lift_address_kernel(
; IR: call i1 @llvm.amdgcn.init.whole.wave()
; IR: call i32 @llvm.amdgcn.workitem.id.x()
; IR: lshr i32 {{.+}}, 5
; IR: and i32 {{.+}}, 31
; IR: shl i32 {{.+}}, 2
; IR: sext i32 {{.+}} to i64
; IR: add i64
; IR: store i32
; NEG-NOT: lshr i32 {{.+}}, 6

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	c1_ttmp_wave_id_lift_address_kernel
	.p2align	8
	.type	c1_ttmp_wave_id_lift_address_kernel,@function
c1_ttmp_wave_id_lift_address_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	s_bfe_u32 s4, ttmp8, 0x50019
	s_lshl_b32 s5, s4, 2
	v_mov_b32_e32 v0, s5
	v_mov_b32_e32 v1, s4
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c1_ttmp_wave_id_lift_address_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           c1_ttmp_wave_id_lift_address_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c1_ttmp_wave_id_lift_address_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
