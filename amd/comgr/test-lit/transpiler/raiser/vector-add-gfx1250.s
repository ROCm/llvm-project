; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s \
; RUN:   -o %t.gfx1250.o
; RUN: %ld.lld -shared %t.gfx1250.o -o %t.gfx1250.hsaco

; RUN: %transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vector_add | %FileCheck %s --check-prefix=IR

; Raise onto the device that is present, lower the result with clang, and check
; the sums the raised kernel writes against the ones the host computes.
; RUN: %if comgr-has-hip-wave64-device %{ %transpile_cli \
; RUN:   %t.gfx1250.hsaco --target-isa=%amdgpu_wave64_arch \
; RUN:   --emit-ir=vector_add > %t.raised.ll %}
; RUN: %if comgr-has-hip-wave64-device %{ %clang -x ir -nogpulib \
; RUN:   --target=amdgcn-amd-amdhsa -mcpu=%amdgpu_wave64_arch \
; RUN:   %t.raised.ll -o %t.device.hsaco %}
; RUN: %if comgr-has-hip-wave64-device %{ %clang %hip_cflags \
; RUN:   %S/../Inputs/run-vector-add.c %hip_ldflags -o %t.run-vector-add %}
; RUN: %if comgr-has-hip-wave64-device %{ %t.run-vector-add %t.device.hsaco \
; RUN:   vector_add 4096 256 | %FileCheck %s --check-prefix=EXECUTE %}
; EXECUTE: RESULT: PASS

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	vector_add
	.p2align	8
	.type	vector_add,@function
; The entry state the source ABI hands the kernel: the workitem id in v0, and
; the workgroup id in the architected TTMPs this source ISA delivers it in.
; IR-LABEL: define amdgpu_kernel void @vector_add(
; IR: [[TID:%.+]] = call i32 @llvm.amdgcn.workitem.id.x()
; IR: [[WGID:%.+]] = call i32 @llvm.amdgcn.workgroup.id.x()
vector_add:
; The three buffer addresses, read out of the kernarg segment.
; IR: [[A_PTR:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i64, ptr addrspace(1) [[A_PTR]], align 4
	s_load_b64 s[4:5], s[0:1], 0x0
; IR: [[B_PTR:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i64, ptr addrspace(1) [[B_PTR]], align 4
	s_load_b64 s[6:7], s[0:1], 0x8
; IR: [[C_PTR:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i64, ptr addrspace(1) [[C_PTR]], align 4
	s_load_b64 s[8:9], s[0:1], 0x10

; The element this lane owns, and its byte offset.
; IR: [[BLOCK:%.+]] = shl i32 [[WGID]], 8
	s_lshl_b32 s3, ttmp9, 8
; IR: [[INDEX:%.+]] = add i32 [[BLOCK]], [[TID]]
	v_add_nc_u32 v1, s3, v0
; IR: shl i32 [[INDEX]], 2
	v_lshlrev_b32 v1, 2, v1
	s_wait_kmcnt 0x0

; Each load is predicated on the lane bit of EXEC, so it sits in its own block.
; IR: load i32, ptr addrspace(1) {{%.+}}, align 4
	global_load_b32 v2, v1, s[4:5]
; IR: load i32, ptr addrspace(1) {{%.+}}, align 4
	global_load_b32 v3, v1, s[6:7]
	s_wait_loadcnt 0x0
; IR: [[LHS:%.+]] = bitcast i32 {{%.+}} to float
; IR: [[RHS:%.+]] = bitcast i32 {{%.+}} to float
; IR: fadd float [[LHS]], [[RHS]]
	v_add_f32 v2, v2, v3
; IR: store i32 {{%.+}}, ptr addrspace(1) {{%.+}}, align 4
	global_store_b32 v1, v2, s[8:9]
; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vector_add
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 10
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 256
    .name:           vector_add
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         vector_add.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
