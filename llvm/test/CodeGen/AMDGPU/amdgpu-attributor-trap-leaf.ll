; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=amdgpu-attributor %s | FileCheck %s

; Trap-like intrinsics such as llvm.trap and llvm.debugtrap do not have the
; nocallback attribute, so the AMDGPU attributor used to conservatively drop
; all implicitly-known inputs and AGPR allocation information. Make sure we
; still infer that no implicit inputs are required and that the AGPR allocation
; stays at zero.

declare void @llvm.trap()

declare void @llvm.debugtrap()

define amdgpu_kernel void @trap_kernel() {
; CHECK-LABEL: define amdgpu_kernel void @trap_kernel(
; CHECK-SAME: ) #[[ATTR_KERNEL:[0-9]+]] {
; CHECK-NEXT:    call void @llvm.trap()
; CHECK-NEXT:    ret void
;
  call void @llvm.trap()
  ret void
}

define amdgpu_kernel void @debugtrap_kernel() {
; CHECK-LABEL: define amdgpu_kernel void @debugtrap_kernel(
; CHECK-SAME: ) #[[ATTR_KERNEL]] {
; CHECK-NEXT:    call void @llvm.debugtrap()
; CHECK-NEXT:    ret void
;
  call void @llvm.debugtrap()
  ret void
}

; CHECK: attributes #[[ATTR_KERNEL]] = { {{.*}}amdgpu-agpr-alloc"="0"{{.*}}amdgpu-no-implicitarg-ptr{{.*}} }
