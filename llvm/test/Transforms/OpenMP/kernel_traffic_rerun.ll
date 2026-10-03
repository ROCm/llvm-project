; Check that a kernel whose estimate an earlier run of the pass already recorded
; keeps it, while one whose earlier run failed is analyzed again.
;
; RUN: opt -passes=openmp-kernel-traffic -mtriple=amdgpu -S < %s | FileCheck %s

@retry_exec_mode = weak_odr protected addrspace(1) constant i8 2
@keep_exec_mode = weak_odr protected addrspace(1) constant i8 2

; Recorded as failed (4 = opaque call) by an earlier run.
@retry_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 4 }
; Recorded as valid by an earlier run; deliberately not what this run would
; compute.
@keep_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 0 }
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @retry_kernel_traffic to ptr), ptr addrspacecast (ptr addrspace(1) @keep_kernel_traffic to ptr)], section "llvm.metadata"

; CHECK: @keep_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 0 }
; CHECK: @retry_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 1, i32 6, i32 0 }
; CHECK: @llvm.compiler.used = appending {{.*}}global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @keep_kernel_traffic to ptr), ptr addrspacecast (ptr addrspace(1) @retry_kernel_traffic to ptr)]

define amdgpu_kernel void @retry(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define amdgpu_kernel void @keep(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
