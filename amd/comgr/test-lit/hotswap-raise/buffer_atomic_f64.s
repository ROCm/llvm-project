; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s
;
; Combined lift test for the buffer_atomic F64 ops, from a gfx1250 source
; cross-raised to gfx942. One kernel exercises add, max_num and min_num: add
; lifts to the raw-buffer atomic-fadd intrinsic, while the _num min/max select
; the IEEE 754-2019 minimumNumber/maximumNumber lift inside a
; raw.buffer.atomic.cmpswap.i64 CAS loop. (The gfx942 raw fcmp+select spellings
; buffer_atomic_{min,max}_f64 are a different encoding that gfx1250 cannot
; assemble, so they are not covered here.) The second CAS loop reuses the same
; value names, so min_num's SSA names are numeric-suffixed.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_f64_kernel
	.p2align	8
	.type	buffer_atomic_f64_kernel,@function
buffer_atomic_f64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b96 s[0:2], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 3, v0
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, 0x3ff00000
	s_mov_b32 s2, -1
; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_f64_kernel(
; add -> native f64 raw-buffer atomic-fadd intrinsic.
; CHECK: call double @llvm.amdgcn.raw.buffer.atomic.fadd.f64
	buffer_atomic_add_f64 v[2:3], v0, s[0:3], null offen
	s_wait_loadcnt 0
; max_num -> IEEE 754-2019 maximumNumber inside a cmpswap.i64 CAS loop.
; CHECK: %fp64_minmax_init = call i64 @llvm.amdgcn.raw.buffer.load.i64
; CHECK: fp64_minmax_loop:
; CHECK: %fp64_minmax_expected = phi i64
; CHECK: %fp64_minmax_new = call double @llvm.maximumnum.f64(double %fp64_minmax_old, double %fp64_minmax_src)
; CHECK: %fp64_minmax_cas = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %fp64_minmax_new_bits, i64 %fp64_minmax_expected
; CHECK: %fp64_minmax_ok = icmp eq i64 %fp64_minmax_cas, %fp64_minmax_expected
	buffer_atomic_max_num_f64 v[2:3], v0, s[0:3], null offen
	s_wait_loadcnt 0
; min_num -> IEEE minimumNumber, second CAS loop (suffixed SSA names).
; CHECK: %fp64_minmax_new{{[0-9]+}} = call double @llvm.minimumnum.f64(double %fp64_minmax_old{{[0-9]+}}, double %fp64_minmax_src{{[0-9]+}})
; CHECK: %fp64_minmax_cas{{[0-9]+}} = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %fp64_minmax_new_bits{{[0-9]+}}, i64 %fp64_minmax_expected{{[0-9]+}}
	buffer_atomic_min_num_f64 v[2:3], v0, s[0:3], null offen
	s_wait_loadcnt 0
	s_endpgm
; Negative pins: no gfx942 raw fcmp path, no native fmin/fmax, no atomicrmw,
; no f32 fadd.
; CHECK-NOT: fcmp
; CHECK-NOT: atomicrmw
; CHECK-NOT: call float @llvm.amdgcn.raw.buffer.atomic.fadd
; CHECK-NOT: llvm.amdgcn.raw.buffer.atomic.fmin
; CHECK-NOT: llvm.amdgcn.raw.buffer.atomic.fmax
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_f64_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
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
      - { .offset:         8, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_atomic_f64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
