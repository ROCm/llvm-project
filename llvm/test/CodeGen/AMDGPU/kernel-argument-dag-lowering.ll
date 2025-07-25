; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -amdgpu-ir-lower-kernel-arguments=0 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,HSA-VI,FUNC %s

; Repeat of some problematic tests in kernel-args.ll, with the IR
; argument lowering pass disabled. Struct padding needs to be
; accounted for, as well as legalization of types changing offsets.

; FUNC-LABEL: {{^}}i1_arg:

; GCN: s_load_dword s
; GCN: s_and_b32

; HSA-VI: .amdhsa_kernarg_size 12
define amdgpu_kernel void @i1_arg(ptr addrspace(1) %out, i1 %x) #0 {
  store i1 %x, ptr addrspace(1) %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}v3i8_arg:

; HSA-VI: s_load_dword s{{[0-9]+}}, s[8:9], 0x8
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x0

; HSA-VI: .amdhsa_kernarg_size 12
define amdgpu_kernel void @v3i8_arg(ptr addrspace(1) nocapture %out, <3 x i8> %in) #0 {
entry:
  store <3 x i8> %in, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v5i8_arg:
; GCN: s_load_dwordx2 s[0:1], s[8:9], 0x0
define amdgpu_kernel void @v5i8_arg(<5 x i8> %in) nounwind {
  store <5 x i8> %in, ptr addrspace(1) null
  ret void
}

; FUNC-LABEL: {{^}}v6i8_arg:
; GCN: s_load_dwordx2 s[0:1], s[8:9], 0x0
define amdgpu_kernel void @v6i8_arg(<6 x i8> %in) nounwind {
  store <6 x i8> %in, ptr addrspace(1) null
  ret void
}

; FUNC-LABEL: {{^}}v5i16_arg:
; GCN: s_load_dwordx4 s[0:3], s[8:9], 0x0
define amdgpu_kernel void @v5i16_arg(<5 x i16> %in) nounwind {
  store <5 x i16> %in, ptr addrspace(1) null
  ret void
}

; FUNC-LABEL: {{^}}v6i16_arg:
; GCN-DAG: s_load_dwordx4 s[0:3], s[8:9], 0x0
define amdgpu_kernel void @v6i16_arg(<6 x i16> %in) nounwind {
  store <6 x i16> %in, ptr addrspace(1) null
  ret void
}

; FUNC-LABEL: {{^}}v5i32_arg:
; GCN: s_load_dwordx4 s[0:3], s[8:9], 0x0
define amdgpu_kernel void @v5i32_arg(<5 x i32> %in) nounwind {
  store <5 x i32> %in, ptr addrspace(1) null
  ret void
}

; FUNC-LABEL: {{^}}v6i32_arg:
; GCN: s_load_dwordx4 s[0:3], s[8:9], 0x0
define amdgpu_kernel void @v6i32_arg(<6 x i32> %in) nounwind {
  store <6 x i32> %in, ptr addrspace(1) null
  ret void
}

; FUNC-LABEL: {{^}}i65_arg:
; HSA-VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x0

; HSA-VI: .amdhsa_kernarg_size 24
define amdgpu_kernel void @i65_arg(ptr addrspace(1) nocapture %out, i65 %in) #0 {
entry:
  store i65 %in, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}empty_struct_arg:
; HSA-VI: .amdhsa_kernarg_size 0
define amdgpu_kernel void @empty_struct_arg({} %in) #0 {
  ret void
}

; The correct load offsets for these:
; load 4 from 0,
; load 8 from 8
; load 4 from 24
; load 8 from 32

; With the SelectionDAG argument lowering, the alignments for the
; struct members is not properly considered, making these wrong.

; FIXME: Total argument size is computed wrong
; FUNC-LABEL: {{^}}struct_argument_alignment:
; HSA-VI: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x8
; HSA-VI: s_load_dword s{{[0-9]+}}, s[8:9], 0x18
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x20

; HSA-VI: .amdhsa_kernarg_size 40
define amdgpu_kernel void @struct_argument_alignment({i32, i64} %arg0, i8, {i32, i64} %arg1) #0 {
  %val0 = extractvalue {i32, i64} %arg0, 0
  %val1 = extractvalue {i32, i64} %arg0, 1
  %val2 = extractvalue {i32, i64} %arg1, 0
  %val3 = extractvalue {i32, i64} %arg1, 1
  store volatile i32 %val0, ptr addrspace(1) null
  store volatile i64 %val1, ptr addrspace(1) null
  store volatile i32 %val2, ptr addrspace(1) null
  store volatile i64 %val3, ptr addrspace(1) null
  ret void
}

; No padding between i8 and next struct, but round up at end to 4 byte
; multiple.
; FUNC-LABEL: {{^}}packed_struct_argument_alignment:
; HSA-VI-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; HSA-VI: global_load_dword v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:13
; HSA-VI: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:17
; HSA-VI: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x4

; HSA-VI: .amdhsa_kernarg_size 28
define amdgpu_kernel void @packed_struct_argument_alignment(<{i32, i64}> %arg0, i8, <{i32, i64}> %arg1) #0 {
  %val0 = extractvalue <{i32, i64}> %arg0, 0
  %val1 = extractvalue <{i32, i64}> %arg0, 1
  %val2 = extractvalue <{i32, i64}> %arg1, 0
  %val3 = extractvalue <{i32, i64}> %arg1, 1
  store volatile i32 %val0, ptr addrspace(1) null
  store volatile i64 %val1, ptr addrspace(1) null
  store volatile i32 %val2, ptr addrspace(1) null
  store volatile i64 %val3, ptr addrspace(1) null
  ret void
}

; GCN-LABEL: {{^}}struct_argument_alignment_after:
; HSA-VI: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x8
; HSA-VI: s_load_dword s{{[0-9]+}}, s[8:9], 0x18
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x20
; HSA-VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x30

; HSA-VI: .amdhsa_kernarg_size 64
define amdgpu_kernel void @struct_argument_alignment_after({i32, i64} %arg0, i8, {i32, i64} %arg2, i8, <4 x i32> %arg4) #0 {
  %val0 = extractvalue {i32, i64} %arg0, 0
  %val1 = extractvalue {i32, i64} %arg0, 1
  %val2 = extractvalue {i32, i64} %arg2, 0
  %val3 = extractvalue {i32, i64} %arg2, 1
  store volatile i32 %val0, ptr addrspace(1) null
  store volatile i64 %val1, ptr addrspace(1) null
  store volatile i32 %val2, ptr addrspace(1) null
  store volatile i64 %val3, ptr addrspace(1) null
  store volatile <4 x i32> %arg4, ptr addrspace(1) null
  ret void
}

; GCN-LABEL: {{^}}array_3xi32:
; HSA-VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x0
define amdgpu_kernel void @array_3xi32(i16 %arg0, [3 x i32] %arg1) {
  store volatile i16 %arg0, ptr addrspace(1) poison
  store volatile [3 x i32] %arg1, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}array_3xi16:
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x0
define amdgpu_kernel void @array_3xi16(i8 %arg0, [3 x i16] %arg1) {
  store volatile i8 %arg0, ptr addrspace(1) poison
  store volatile [3 x i16] %arg1, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}v2i15_arg:
; GCN: s_load_dword [[DWORD:s[0-9]+]]
; GCN-DAG: s_bfe_u32 [[BFE:s[0-9]+]], [[DWORD]], 0x100010{{$}}
; GCN-DAG: s_and_b32 [[AND:s[0-9]+]], [[DWORD]], 0x7fff{{$}}
define amdgpu_kernel void @v2i15_arg(ptr addrspace(1) nocapture %out, <2 x i15> %in) #0 {
entry:
  store <2 x i15> %in, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v3i15_arg:
; GCN: s_load_dwordx4 [[DWORDX4:s\[[0-9]+:[0-9]+\]]]
; GCN: s_lshl_b64
; GCN: s_and_b32
; GCN: s_and_b32
; GCN: s_or_b32
define amdgpu_kernel void @v3i15_arg(ptr addrspace(1) nocapture %out, <3 x i15> %in) #0 {
entry:
  store <3 x i15> %in, ptr addrspace(1) %out, align 4
  ret void
}

; Byref pointers should only be treated as offsets from kernarg
; GCN-LABEL: {{^}}byref_constant_i8_arg:
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN: global_load_ubyte v{{[0-9]+}}, [[ZERO]], s[8:9] offset:8
; GCN: .amdhsa_kernarg_size 12
define amdgpu_kernel void @byref_constant_i8_arg(ptr addrspace(1) nocapture %out, ptr addrspace(4) byref(i8) %in.byref) #0 {
  %in = load i8, ptr addrspace(4) %in.byref
  %ext = zext i8 %in to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_constant_i16_arg:
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN: global_load_ushort v{{[0-9]+}}, [[ZERO]], s[8:9] offset:8
; GCN: .amdhsa_kernarg_size 12
define amdgpu_kernel void @byref_constant_i16_arg(ptr addrspace(1) nocapture %out, ptr addrspace(4) byref(i16) %in.byref) #0 {
  %in = load i16, ptr addrspace(4) %in.byref
  %ext = zext i16 %in to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_constant_i32_arg:
; GCN: s_load_dwordx4 [[LOAD:s\[[0-9]+:[0-9]+\]]], s[8:9], 0x0{{$}}
; GCN: .amdhsa_kernarg_size 16
define amdgpu_kernel void @byref_constant_i32_arg(ptr addrspace(1) nocapture %out, ptr addrspace(4) byref(i32) %in.byref, i32 %after.offset) #0 {
  %in = load i32, ptr addrspace(4) %in.byref
  store volatile i32 %in, ptr addrspace(1) %out, align 4
  store volatile i32 %after.offset, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_constant_v4i32_arg:
; GCN: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x10{{$}}
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x20{{$}}
; GCN: .amdhsa_kernarg_size 36
define amdgpu_kernel void @byref_constant_v4i32_arg(ptr addrspace(1) nocapture %out, ptr addrspace(4) byref(<4 x i32>) %in.byref, i32 %after.offset) #0 {
  %in = load <4 x i32>, ptr addrspace(4) %in.byref
  store volatile <4 x i32> %in, ptr addrspace(1) %out, align 4
  store volatile i32 %after.offset, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_align_constant_i32_arg:
; GCN-DAG: s_load_dwordx2 s[[[IN:[0-9]+]]:[[AFTER_OFFSET:[0-9]+]]], s[8:9], 0x100{{$}}
; GCN-DAG: v_mov_b32_e32 [[V_IN:v[0-9]+]], s[[IN]]
; GCN-DAG: v_mov_b32_e32 [[V_AFTER_OFFSET:v[0-9]+]], s[[AFTER_OFFSET]]
; GCN: global_store_dword v{{[0-9]+}}, [[V_IN]], s
; GCN: global_store_dword v{{[0-9]+}}, [[V_AFTER_OFFSET]], s
; GCN: .amdhsa_kernarg_size 264
define amdgpu_kernel void @byref_align_constant_i32_arg(ptr addrspace(1) nocapture %out, ptr addrspace(4) byref(i32) align(256) %in.byref, i32 %after.offset) #0 {
  %in = load i32, ptr addrspace(4) %in.byref
  store volatile i32 %in, ptr addrspace(1) %out, align 4
  store volatile i32 %after.offset, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_natural_align_constant_v16i32_arg:
; GCN-DAG: s_load_dword s{{[0-9]+}}, s[8:9], 0x80
; GCN-DAG: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[8:9], 0x40{{$}}
; GCN: .amdhsa_kernarg_size 132
define amdgpu_kernel void @byref_natural_align_constant_v16i32_arg(ptr addrspace(1) nocapture %out, i8, ptr addrspace(4) byref(<16 x i32>) align(64) %in.byref, i32 %after.offset) #0 {
  %in = load <16 x i32>, ptr addrspace(4) %in.byref
  store volatile <16 x i32> %in, ptr addrspace(1) %out, align 4
  store volatile i32 %after.offset, ptr addrspace(1) %out, align 4
  ret void
}

; Also accept byref kernel arguments with other global address spaces.
; GCN-LABEL: {{^}}byref_global_i32_arg:
; GCN: s_load_dword [[IN:s[0-9]+]], s[8:9], 0x8{{$}}
; GCN: .amdhsa_kernarg_size 12
define amdgpu_kernel void @byref_global_i32_arg(ptr addrspace(1) nocapture %out, ptr addrspace(1) byref(i32) %in.byref) #0 {
  %in = load i32, ptr addrspace(1) %in.byref
  store i32 %in, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_flat_i32_arg:
; GCN: flat_load_dword [[IN:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}} offset:8{{$}}
define amdgpu_kernel void @byref_flat_i32_arg(ptr addrspace(1) nocapture %out, ptr byref(i32) %in.byref) #0 {
  %in = load i32, ptr %in.byref
  store i32 %in, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_constant_32bit_i32_arg:
; GCN: s_add_i32 s[[PTR_LO:[0-9]+]], s8, 8
; GCN: s_mov_b32 s[[PTR_HI:[0-9]+]], 0{{$}}
; GCN: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x0{{$}}
define amdgpu_kernel void @byref_constant_32bit_i32_arg(ptr addrspace(1) nocapture %out, ptr addrspace(6) byref(i32) %in.byref) #0 {
  %in = load i32, ptr addrspace(6) %in.byref
  store i32 %in, ptr addrspace(1) %out, align 4
  ret void
}

; define amdgpu_kernel void @byref_unknown_as_i32_arg(ptr addrspace(1) nocapture %out, ptr addrspace(999) byref %in.byref) {
;   %in = load i32, ptr addrspace(999) %in.byref
;   store i32 %in, ptr addrspace(1) %out, align 4
;   ret void
; }

; GCN-LABEL: {{^}}multi_byref_constant_i32_arg:
; GCN: s_load_dwordx4 {{s\[[0-9]+:[0-9]+\]}}, s[8:9], 0x0
; GCN: .amdhsa_kernarg_size 20
define amdgpu_kernel void @multi_byref_constant_i32_arg(ptr addrspace(1) nocapture %out, ptr addrspace(4) byref(i32) %in0.byref, ptr addrspace(4) byref(i32) %in1.byref, i32 %after.offset) #0 {
  %in0 = load i32, ptr addrspace(4) %in0.byref
  %in1 = load i32, ptr addrspace(4) %in1.byref
  store volatile i32 %in0, ptr addrspace(1) %out, align 4
  store volatile i32 %in1, ptr addrspace(1) %out, align 4
  store volatile i32 %after.offset, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_constant_i32_arg_offset0:
; GCN-NOT: s4
; GCN-NOT: s5
; GCN: s_load_dword {{s[0-9]+}}, s[8:9], 0x0{{$}}
; GCN: .amdhsa_kernarg_size 4
define amdgpu_kernel void @byref_constant_i32_arg_offset0(ptr addrspace(4) byref(i32) %in.byref) #0 {
  %in = load i32, ptr addrspace(4) %in.byref
  store i32 %in, ptr addrspace(1) poison, align 4
  ret void
}

attributes #0 = { "amdgpu-no-implicitarg-ptr" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
