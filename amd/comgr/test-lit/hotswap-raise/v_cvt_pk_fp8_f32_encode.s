; RUN: split-file %s %t

; RUN: %llvm_mc -mcpu=gfx1250 %t/gfx1250.s -o %t/t16.o \
; RUN:   && %ld_lld -shared %t/t16.o -o %t/t16.hsaco \
; RUN:   && raise_cli %t/t16.hsaco --target-isa=gfx942 --emit-ir=cvt_enc_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS \
; RUN:       --implicit-check-not=e4m3_ocp --implicit-check-not=e5m2_ocp

; RUN: %llvm_mc -mcpu=gfx1250 %t/gfx1250.s -o %t/t16.o \
; RUN:   && %ld_lld -shared %t/t16.o -o %t/t16.hsaco \
; RUN:   && raise_cli %t/t16.hsaco --target-isa=gfx1250 --emit-ir=cvt_enc_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME --implicit-check-not=pk_fp8_ocp

; gfx1250 reuses CLAMP on the fp8 converts as a FORMAT select (E4M3 vs E5M3),
; not an output clamp. E5M3 has no target-side equivalent, so refuse.
; RUN: %not raise_cli %t/t16.hsaco --target-isa=gfx942 --emit-ir=cvt_e5m3_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=E5M3

; The gfx9 disassembly has no op_sel operand and no _HI16 dst: the dst half
; selector is the DST_OP_SEL bit of src0_modifiers.
; RUN: %llvm_mc -mcpu=gfx950 %t/gfx950.s -o %t/g9.o \
; RUN:   && %ld_lld -shared %t/g9.o -o %t/g9.hsaco \
; RUN:   && raise_cli %t/g9.hsaco --target-isa=gfx942 \
; RUN:        --emit-ir=cvt_enc_gfx9_kernel \
; RUN:   | %FileCheck %s --check-prefix=GFX9

;--- gfx1250.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_enc_kernel
	.p2align	8
	.type	cvt_enc_kernel,@function
cvt_enc_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0x40400000
	v_mov_b32_e32 v2, 0x40a00000
; Cross-target encode goes straight to OCP bytes. OCP and FNUZ share a
; mantissa width and differ by one in exponent bias, so the target's FNUZ
; encoder applied to x/2 yields the OCP byte for x -- hardware still does the
; rounding. Out-of-range magnitudes are split off first, so the raw encoder
; never overflows and MODE.FP16_OVFL never decides the result. E4M3's 464 is
; the round-half-to-even tie and still reaches 448, so only strictly-above is
; the NaN (0x7F) that FP16_OVFL=0 calls for; E5M2 ties to Inf at 61440.
; CROSS-LABEL: define amdgpu_kernel void @cvt_enc_kernel(
; CROSS-DAG: fcmp ogt float %{{.+}}, 4.640000e+02
; CROSS-DAG: select i1 %{{.+}}, i32 127, i32 %{{.+}}
; CROSS-DAG: fmul float %{{.+}}, 5.000000e-01
; CROSS-DAG: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %{{.+}}, float %{{.+}}, i32 0, i1 false)
; CROSS-DAG: fcmp oge float %{{.+}}, 6.144000e+04
; CROSS-DAG: call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %{{.+}}, float %{{.+}}, i32 0, i1 false)
; CROSS-DAG: %pk_fp8_ocp{{[0-9]*}} = or
; op_sel:[0,0,1] writes the HIGH half and preserves the low one. The
; disassembler drops the op_sel operand and names the dst half instead
; (`v3.h`), so this only works if the dst _HI16 subreg is honoured. v3 is
; never written, so the preserved half is a frozen undef rather than a raw
; one that `and`/`or` could fold to 0 or -1.
; CROSS-DAG: %cvt_pk_old{{[0-9]*}} = freeze i32 undef
; CROSS-DAG: and i32 %cvt_pk_old{{[0-9]*}}, 65535
; CROSS-DAG: shl i32 %pk_fp8_ocp{{[0-9]*}}, 16
; The other convert takes the default op_sel, which writes the LOW half and
; preserves the high one. Without this, a `cvtPkFp8DstIsHi` stuck at true
; satisfies every remaining CROSS directive.
; CROSS-DAG: %[[KEEP:[^ ]+]] = and i32 %cvt_pk_old{{[0-9]*}}, -65536
; CROSS-DAG: or i32 %[[KEEP]], %pk_fp8_ocp{{[0-9]*}}
; No byte-level re-encode is involved on this path any more; the RUN line
; asserts that, since a CHECK-NOT after a DAG group only scans forward from
; the group's last match.
; SAME-LABEL: define amdgpu_kernel void @cvt_enc_kernel(
; SAME: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(
	v_cvt_pk_fp8_f32 v0, v1, v2
	v_cvt_pk_bf8_f32 v3, v1, v2 op_sel:[0,0,1]
	v_mov_b32_e32 v5, 0
	s_wait_kmcnt 0x0
	global_store_b64 v5, v[0:1], s[0:1]
	s_endpgm
	.globl	cvt_e5m3_kernel
	.p2align	8
	.type	cvt_e5m3_kernel,@function
cvt_e5m3_kernel:
	v_mov_b32_e32 v1, 0x40400000
	v_mov_b32_e32 v2, 0x40a00000
; E5M3: kernel 'cvt_e5m3_kernel' failed to raise:
; E5M3-SAME: selects the E5M3 fp8 format
	v_cvt_pk_fp8_f32 v0, v1, v2 clamp
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_e5m3_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel cvt_enc_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
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
    .name:           cvt_enc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cvt_enc_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cvt_e5m3_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         cvt_e5m3_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata

;--- gfx950.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_enc_gfx9_kernel
	.p2align	8
	.type	cvt_enc_gfx9_kernel,@function
cvt_enc_gfx9_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0x40400000
	v_mov_b32_e32 v2, 0x40a00000
; GFX9-LABEL: define amdgpu_kernel void @cvt_enc_gfx9_kernel(
; Default op_sel writes the LOW half and preserves the high one.
; GFX9-DAG: and i32 %cvt_pk_old{{[0-9]*}}, -65536
; op_sel:[0,0,1] arrives as the DST_OP_SEL bit of src0_modifiers here, and
; still has to select the HIGH half. v3 is never written, so the preserved
; half must be a frozen undef, not a foldable one.
; GFX9-DAG: %cvt_pk_old{{[0-9]*}} = freeze i32 undef
; GFX9-DAG: and i32 %cvt_pk_old{{[0-9]*}}, 65535
; GFX9-DAG: shl i32 %pk_fp8_ocp{{[0-9]*}}, 16
; gfx950 CLAMP is an ordinary output clamp, not the gfx1250 E5M3 format
; select, so a clamped convert must still raise.
; GFX9-NOT: E5M3
	v_cvt_pk_fp8_f32 v0, v1, v2
	v_cvt_pk_bf8_f32 v3, v1, v2 op_sel:[0,0,1]
	v_cvt_f32_fp8_e64 v4, v0 clamp
	v_mov_b32_e32 v5, 0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx2 v5, v[0:1], s[0:1]
	global_store_dword v5, v4, s[0:1] offset:8
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_enc_gfx9_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
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
    .name:           cvt_enc_gfx9_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cvt_enc_gfx9_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
