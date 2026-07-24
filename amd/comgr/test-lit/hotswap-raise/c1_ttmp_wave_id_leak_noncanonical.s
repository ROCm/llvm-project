; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:      --emit-ir=c1_ttmp_wave_id_leak_noncanonical_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ERR
;
; A non-canonical ttmp8 read under cross-widening is a wave_id leak with no
; rewrite -> refuse. Guards the TtmpWaveIdLeak emission in
; wave-size-obstruction.cpp; the canonical `s_bfe_u32 ttmp8, 0x50019` read is
; filtered by isCanonicalWaveIdBfe and rescued in handle-sop2.cpp instead.
; Immediate 0x50018 (offset=24) reads outside the modeled [29:25] wave_id
; field, so isCanonicalWaveIdBfe is false. No WMMA -> exercises the de-gated
; refusal path.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	c1_ttmp_wave_id_leak_noncanonical_kernel
	.p2align	8
	.type	c1_ttmp_wave_id_leak_noncanonical_kernel,@function
c1_ttmp_wave_id_leak_noncanonical_kernel:
; ERR: pre-translation abort: cross-wave-lane-id-leak
; ERR-SAME: s_bfe_u32
; ERR: TtmpWaveIdLeak
	s_bfe_u32 s0, ttmp8, 0x50018
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c1_ttmp_wave_id_leak_noncanonical_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           c1_ttmp_wave_id_leak_noncanonical_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         c1_ttmp_wave_id_leak_noncanonical_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
