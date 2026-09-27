; Test the AMDGPU call type identifiers in the module summary. They compare
; the sizes of the types, not the IR types. The summary stores the MD5 hash of
; each identifier:
;   3430040695729964317  = "amdgpu.v1:cc0:ii"
;   5970146889962867861  = "amdgpu.v1:cc0:vl"
;   11963068178503280946 = "amdgpu.v1:cc100:vl"

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

; float(float) and i32(i32) have the same identifier.
; CHECK-DAG: (name: "f_float", {{.*}}, callTypeId: 3430040695729964317)))
; CHECK-DAG: (name: "f_i32", {{.*}}, callTypeId: 3430040695729964317)))
; Two functions with the same type and different calling conventions have
; different identifiers.
; CHECK-DAG: (name: "device", {{.*}}, callTypeId: 5970146889962867861)))
; CHECK-DAG: (name: "gfx", {{.*}}, callTypeId: 11963068178503280946)))
; A kernel cannot be called, so it has no identifier. The two indirect calls in
; the kernel have the same identifier.
; CHECK-DAG: (name: "kernel", {{.*}}, indirectCallTypeIds: (3430040695729964317))))

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define float @f_float(float %x) {
  ret float %x
}

define i32 @f_i32(i32 %x) {
  ret i32 %x
}

define void @device(ptr %p) {
  ret void
}

define amdgpu_gfx void @gfx(ptr %p) {
  ret void
}

define amdgpu_kernel void @kernel(ptr %fp) {
  %r = call float %fp(float 1.0)
  %s = call i32 %fp(i32 1)
  ret void
}
