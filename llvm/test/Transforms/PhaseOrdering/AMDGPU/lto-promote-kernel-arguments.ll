; Check that a pointer loaded through a kernel argument is promoted to the
; global address space in the full-LTO pipeline.
;
; RUN: opt -mtriple=amdgpu -S -passes='lto<O2>' %s -o - | FileCheck %s

; CHECK-LABEL: define amdgpu_kernel void @kernel(
; CHECK:         store i32 0, ptr addrspace(1)
define amdgpu_kernel void @kernel(ptr %desc) {
  %p = load ptr, ptr %desc, align 8
  store i32 0, ptr %p, align 4
  ret void
}
