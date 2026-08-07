; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=buffer_oob_sentinel_clamp_kernel 2>&1 | %FileCheck %s

; A gfx1250 raw buffer lifted to gfx942 restores the source hardware's
; out-of-bounds suppression, which gfx942/gfx950 do not provide, at the MUBUF
; address layer. The Triton source encodes a masked-lane offset as the
; 0x80000000 out-of-bounds sentinel; gfx1250 hardware drops such an access,
; the target hardware faults. Both guards key on the reconstructed runtime
; NUM_RECORDS descriptor bound, not on modeled EXEC / lane position.
; Guards decodeMubufAddr (load clamp) and handleMUBUF (store suppression).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_oob_sentinel_clamp_kernel
	.p2align	8
	.type	buffer_oob_sentinel_clamp_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @buffer_oob_sentinel_clamp_kernel(
buffer_oob_sentinel_clamp_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_mov_b32 s2, 4
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
; A load offset >= NUM_RECORDS is redirected to 0 (reads in-bounds element 0,
; matching the source hardware's return-0), leaving in-bounds offsets untouched.
; CHECK: [[LOOB:%.+]] = icmp uge i64 %{{.+}}, %{{.+}}
; CHECK: select i1 [[LOOB]], i32 0, i32 %{{.+}}
	buffer_load_dword v4, v1, s[0:3], null offen
	s_wait_loadcnt 0
; A store to an offset >= NUM_RECORDS is suppressed (the out-of-bounds lane does
; not issue), rather than clamped, so it cannot corrupt in-bounds element 0.
; CHECK: [[SIB:%.+]] = icmp ult i64 %{{.+}}, %{{.+}}
; CHECK: br i1 [[SIB]], label %st_oob_do, label %st_oob_skip
; CHECK: st_oob_do:
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i32(
	buffer_store_dword v4, v1, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_oob_sentinel_clamp_kernel
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
    .name:           buffer_oob_sentinel_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_oob_sentinel_clamp_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
