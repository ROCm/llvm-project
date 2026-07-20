; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=buffer_load_gfx1250_num_records_kernel 2>&1 | %FileCheck %s

; A gfx1250 raw buffer lifted to gfx942 (cross-widening) goes through the
; make.buffer.rsrc path. gfx1250 NUM_RECORDS is resource[101:57], split across
; word1[31:25] (7 low bits), word2 (32 middle bits) and word3[5:0] (6 high
; bits); it must be reconstructed, not read from word2 alone (a 128x-too-small
; bound). The reconstruction is gated on the source ISA carrying a 45-bit
; num_records field (Has45BitNumRecordsBufferResource); a gfx1250 source hits
; it here. The reconstructed extent is then clamped to the gfx942 raw-buffer
; max, which also maps the gfx12 all-ones "OOB disabled" encodings onto it.
; Guards the reconstruction in decodeMubufAddr (mubuf-addr.cpp).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_load_gfx1250_num_records_kernel
	.p2align	8
	.type	buffer_load_gfx1250_num_records_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @buffer_load_gfx1250_num_records_kernel(
buffer_load_gfx1250_num_records_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_mov_b32 s2, 4
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
; word1[31:25] -> 7 low bits of NUM_RECORDS
; CHECK: and i32 %{{.+}}, 127
; word2 supplies the 32 middle bits (shifted left by the 7 low bits)
; CHECK: [[MID:%.+]] = shl i64 %{{.+}}, 7
; word3[5:0] -> 6 high bits at field position [44:39]
; CHECK: and i32 {{.+}}, 63
; CHECK: shl i64 %{{.+}}, 39
; CHECK: %{{.+}} = or i64 %{{.+}}, %{{.+}}
; the reconstructed extent is clamped to the gfx942 raw-buffer max (this also
; folds the gfx12 all-ones "OOB disabled" encodings onto the max)
; CHECK: icmp ugt i64 %{{.+}}, 2147483646
; CHECK: select i1 %{{.+}}, i64 2147483646, i64 %{{.+}}
; CHECK: call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %{{.+}}, i16 0, i64 %{{.+}}, i32 159744)
	buffer_load_dword v4, v1, s[0:3], null offen
	s_wait_loadcnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_load_gfx1250_num_records_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
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
    .max_flat_workgroup_size: 64
    .name:           buffer_load_gfx1250_num_records_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_load_gfx1250_num_records_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
