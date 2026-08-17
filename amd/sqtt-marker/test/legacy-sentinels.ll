; RUN: %opt -load-pass-plugin=%sqtt-marker-plugin \
; RUN:   -sqtt-marker-scope-cu=-1 -sqtt-marker-scope-simd=-1 \
; RUN:   -sqtt-marker-mem-barrier=none -passes='default<O0>' \
; RUN:   -S %s -o - | %FileCheck %s
; REQUIRES: amdgpu-registered-target

target triple = "amdgcn-amd-amdhsa"

@scope = private addrspace(1) constant [7 x i8] c"legacy\00"
@point = private addrspace(1) constant [6 x i8] c"point\00"
@data = private addrspace(1) constant [5 x i8] c"data\00"

declare void @__sqtt_named_marker_enter(ptr addrspace(1))
declare void @__sqtt_named_marker_exit(ptr addrspace(1))
declare void @__sqtt_named_marker_point(ptr addrspace(1))
declare void @__sqtt_named_marker_data(ptr addrspace(1), i32)

define amdgpu_kernel void @legacy_kernel() #0 {
entry:
  call void @__sqtt_named_marker_enter(ptr addrspace(1) @scope)
  call void @__sqtt_named_marker_point(ptr addrspace(1) @point)
  call void @__sqtt_named_marker_data(ptr addrspace(1) @data, i32 42)
  call void @__sqtt_named_marker_exit(ptr addrspace(1) @scope)
  ret void
}

attributes #0 = { "target-cpu"="gfx1200" }

; CHECK: c"K:legacy_kernel\0AU:1:legacy\0AP:2:point\0AP:3:data\0AR:3:extra_payload_count=1\0A\00"
; CHECK: call void @llvm.amdgcn.s.ttracedata.imm(i16 6)
; CHECK: call void @llvm.amdgcn.s.ttracedata.imm(i16 8)
; CHECK: call void @llvm.amdgcn.s.ttracedata.imm(i16 12)
; CHECK: call i32 asm sideeffect
; CHECK-SAME: "={m0},i"(i32 42)
; CHECK-SAME: !sqtt.raw_payload
; CHECK: call void @llvm.amdgcn.s.ttracedata.imm(i16 1)
; CHECK-NOT: declare void @__sqtt_named_marker_
