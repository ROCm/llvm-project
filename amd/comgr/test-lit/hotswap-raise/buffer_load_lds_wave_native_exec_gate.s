; RUN: %llvm_mc -mcpu=gfx1030 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=buffer_load_lds_wave_native_exec_gate_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=WN --check-prefix=CHECK
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:   --emit-ir=buffer_load_lds_wave_native_exec_gate_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=MR --check-prefix=CHECK
;
; MUBUF buffer-load-to-LDS EXEC-gating (rocm-systems#148, companion to
; buffer_load_wave_native_exec_gate.s). The buffer load feeding the LDS
; store must itself be inside the `emitUnderExec` diamond so a phantom /
; source-inactive lane never issues it. Guards the BUFFER_LOAD_*_LDS path
; of `handleMUBUF` in handle-mubuf.cpp.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.text
	.globl	buffer_load_lds_wave_native_exec_gate_kernel
	.p2align	8
	.type	buffer_load_lds_wave_native_exec_gate_kernel,@function
; WN-LABEL: define amdgpu_kernel void @buffer_load_lds_wave_native_exec_gate_kernel(
; WN: call i1 @llvm.amdgcn.init.whole.wave()
; MR-LABEL: define amdgpu_kernel void @buffer_load_lds_wave_native_exec_gate_kernel(
buffer_load_lds_wave_native_exec_gate_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_mov_b32 m0, 0
	s_mov_b32 s2, 4
	s_mov_b32 s3, 0x31014000
	s_waitcnt lgkmcnt(0)
	; The load and its LDS store both land inside one diamond.
	; CHECK: spe_do{{.+}}:
	; CHECK: [[LD:%.+]] = call i32 @llvm.amdgcn.raw.buffer.load.i32(
	; CHECK: store i32 [[LD]], ptr addrspace(3)
	buffer_load_dword v1, s[0:3], 0 offen lds
	s_waitcnt vmcnt(0)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6
	.amdhsa_kernel buffer_load_lds_wave_native_exec_gate_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 64
    .name:           buffer_load_lds_wave_native_exec_gate_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_load_lds_wave_native_exec_gate_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
