; RUN: split-file %s %t

; RUN: %llvm_mc -mcpu=gfx1250 %t/gfx1250.s -o %t/t16.o \
; RUN:   && %ld_lld -shared %t/t16.o -o %t/t16.hsaco \
; RUN:   && raise_cli %t/t16.hsaco --target-isa=gfx942 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS \
; RUN:       --implicit-check-not=llvm.amdgcn.cvt.pk.f32 \
; RUN:       --implicit-check-not=llvm.amdgcn.cvt.f32.fp8 \
; RUN:       --implicit-check-not=_dec_fnuz

; RUN: %llvm_mc -mcpu=gfx1250 %t/gfx1250.s -o %t/t16.o \
; RUN:   && %ld_lld -shared %t/t16.o -o %t/t16.hsaco \
; RUN:   && raise_cli %t/t16.hsaco --target-isa=gfx1250 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME --implicit-check-not=_dec_

; On gfx9 the word selector is SDWA src0_sel, which never prints `op_sel:`.
; RUN: %llvm_mc -mcpu=gfx950 %t/gfx950.s -o %t/g9.o \
; RUN:   && %ld_lld -shared %t/g9.o -o %t/g9.hsaco \
; RUN:   && raise_cli %t/g9.hsaco --target-isa=gfx942 \
; RUN:        --emit-ir=cvt_dec_gfx9_kernel \
; RUN:   | %FileCheck %s --check-prefix=GFX9

; Only the WORD selects are a legal src0_sel for this opcode; a byte select is
; not a word selector, so refuse instead of decoding the wrong lanes.
; RUN: %not raise_cli %t/g9.hsaco --target-isa=gfx942 \
; RUN:        --emit-ir=cvt_dec_bytesel_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=BYTESEL

;--- gfx1250.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_dec_kernel
	.p2align	8
	.type	cvt_dec_kernel,@function
cvt_dec_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x40404040
; Cross-target: decode the source-format byte exactly in IR rather than
; re-encoding it into the target's narrower format first. 6 fp8 bytes here
; (packed pair, single, op_sel:[1] pair, byte_sel:2 single) plus a bf8 pair;
; bf8 is E5M2, the one with Inf, which only survives on this path. The RUN
; line asserts no decode intrinsic and no FNUZ-direction decode appear.
;
; Each decode is pinned to the byte it reads AND to its consumer, in program
; order: byte 0 is an `and`, bytes 1..3 an `lshr`, a pair feeds two
; insertelements and a single a 32-bit dst. Counting selects alone cannot tell
; the byte_sel:2 lane from the op_sel:[1,0] pair, which reads byte 2 as well.
; CROSS-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; Packed pair, default word: bytes 0 and 1.
; CROSS: and i32 %Vgpr0{{[^,]*}}, 255
; CROSS: %[[P0A:fp8_dec_ocp[0-9]*]] = select
; CROSS: insertelement <2 x float> poison, float %[[P0A]], i64 0
; CROSS: lshr i32 %Vgpr0{{[^,]*}}, 8
; CROSS: %[[P0B:fp8_dec_ocp[0-9]*]] = select
; CROSS: insertelement <2 x float> %{{[^,]+}}, float %[[P0B]], i64 1
; Single with the default byte_sel:0: byte 0, straight to a 32-bit dst.
; CROSS: and i32 %Vgpr0{{[^,]*}}, 255
; CROSS: %[[S0:fp8_dec_ocp[0-9]*]] = select
; CROSS: bitcast float %[[S0]] to i32
; op_sel:[1,0] pair: bytes 2 and 3.
; CROSS: lshr i32 %Vgpr0{{[^,]*}}, 16
; CROSS: %[[P1A:fp8_dec_ocp[0-9]*]] = select
; CROSS: insertelement <2 x float> poison, float %[[P1A]], i64 0
; CROSS: lshr i32 %Vgpr0{{[^,]*}}, 24
; CROSS: %[[P1B:fp8_dec_ocp[0-9]*]] = select
; CROSS: insertelement <2 x float> %{{[^,]+}}, float %[[P1B]], i64 1
; bf8 pair: bytes 0 and 1 read as E5M2.
; CROSS: %[[B0:bf8_dec_ocp[0-9]*]] = select
; CROSS: insertelement <2 x float> poison, float %[[B0]], i64 0
; CROSS: lshr i32 %Vgpr0{{[^,]*}}, 8
; CROSS: %[[B1:bf8_dec_ocp[0-9]*]] = select
; CROSS: insertelement <2 x float> %{{[^,]+}}, float %[[B1]], i64 1
; byte_sel:2 single. gfx1250 prints it as `byte_sel:N`, never `op_sel:`, so a
; textual op_sel guard never sees it; a byte_sel that collapsed to 0 would
; emit the byte-0 `and` here instead of this shift.
; CROSS: lshr i32 %Vgpr0{{[^,]*}}, 16
; CROSS: %[[S2:fp8_dec_ocp[0-9]*]] = select
; CROSS: bitcast float %[[S2]] to i32
; CROSS-NOT: fp8_dec_ocp
; CROSS-NOT: bf8_dec_ocp
; Same-format target keeps the hardware decode, op_sel:[1] included, and
; builds no software decode at all.
; SAME-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; SAME: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %{{[^,]+}}, i1 false)
; SAME: call float @llvm.amdgcn.cvt.f32.fp8(
; SAME: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %{{[^,]+}}, i1 true)
; SAME: call float @llvm.amdgcn.cvt.f32.fp8(i32 %{{[^,]+}}, i32 2)
	v_cvt_pk_f32_fp8 v[2:3], v0
	v_cvt_f32_fp8 v4, v0
	v_cvt_pk_f32_fp8 v[6:7], v0 op_sel:[1,0]
	v_cvt_pk_f32_bf8 v[10:11], v0
	v_cvt_f32_fp8 v12, v0 byte_sel:2
	v_mov_b32_e32 v8, 0
	s_wait_kmcnt 0x0
	global_store_b96 v8, v[2:4], s[0:1]
	global_store_b64 v8, v[6:7], s[0:1] offset:16
	global_store_b64 v8, v[10:11], s[0:1] offset:24
	global_store_b32 v8, v12, s[0:1] offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_dec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 13
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
    .name:           cvt_dec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cvt_dec_kernel.kd
    .vgpr_count:     13
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata

;--- gfx950.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_dec_gfx9_kernel
	.p2align	8
	.type	cvt_dec_gfx9_kernel,@function
cvt_dec_gfx9_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x40404040
; GFX9-LABEL: define amdgpu_kernel void @cvt_dec_gfx9_kernel(
; src0_sel:WORD_1 selects bytes 2 and 3 of src0.
; GFX9-DAG: lshr i32 %Vgpr0{{[^,]*}}, 16
; GFX9-DAG: lshr i32 %Vgpr0{{[^,]*}}, 24
; The non-SDWA form implies WORD_0, i.e. bytes 0 and 1.
; GFX9-DAG: lshr i32 %Vgpr0{{[^,]*}}, 8
; gfx950 CLAMP is an ordinary output clamp, not the gfx1250 E5M3 format
; select, so a clamped convert must still raise.
; GFX9-NOT: E5M3
	v_cvt_pk_f32_fp8_sdwa v[2:3], v0 src0_sel:WORD_1
	v_cvt_pk_f32_fp8 v[6:7], v0
	v_cvt_f32_fp8_e64 v4, v0 clamp
	v_mov_b32_e32 v8, 0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx2 v8, v[2:3], s[0:1]
	global_store_dwordx2 v8, v[6:7], s[0:1] offset:8
	global_store_dword v8, v4, s[0:1] offset:16
	s_endpgm
	.globl	cvt_dec_bytesel_kernel
	.p2align	8
	.type	cvt_dec_bytesel_kernel,@function
cvt_dec_bytesel_kernel:
	v_mov_b32_e32 v0, 0x40404040
; BYTESEL: kernel 'cvt_dec_bytesel_kernel' failed to raise:
; BYTESEL-SAME: requires a WORD src0_sel
	v_cvt_pk_f32_fp8_sdwa v[2:3], v0 src0_sel:BYTE_1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_dec_bytesel_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel cvt_dec_gfx9_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 12
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
    .name:           cvt_dec_gfx9_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cvt_dec_gfx9_kernel.kd
    .vgpr_count:     9
    .wavefront_size: 64
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           cvt_dec_bytesel_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         cvt_dec_bytesel_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
