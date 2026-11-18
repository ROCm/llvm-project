; RUN: not --crash llc -mtriple=amdgpu9.42-amd-amdhsa -O2 -global-isel -amdgpu-late-wave-transform=1 -o /dev/null %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Early tail duplication makes the loop irreducible, and the wave transform
; asserts.

; CHECK: Assertion `SuccNodeIt != NodeForBlock.end()' failed.

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @divergent_header(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %tid, 0
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  br i1 %cond, label %then, label %else

then:
  store volatile i32 1, ptr addrspace(1) %out
  br label %latch

else:
  store volatile i32 2, ptr addrspace(1) %out
  %done = icmp eq i32 %i, %n
  br i1 %done, label %exit, label %latch

latch:
  %i.next = add i32 %i, 1
  br label %header

exit:
  ret void
}
