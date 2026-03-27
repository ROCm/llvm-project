; RUN: opt -mtriple=amdgcn-amd-amdhsa -aa-pipeline=basic-aa,scoped-noalias-aa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; Test that getModRefInfo(FenceInst, Loc) uses scoped noalias metadata on the
; fence to prove it cannot affect a given memory location. Without this, the
; load-store vectorizer conservatively treats fences as potential clobbers,
; preventing vectorization of adjacent loads across fences.
;
; This models what happens after AMDGPULowerKernelArguments: noalias kernel
; pointer arguments are replaced with loads from the kernarg segment, losing
; the readonly attribute. The !alias.scope/!noalias metadata partially
; compensates, but getModRefInfo(FenceInst) did not check it.

define void @vectorize_loads_across_fence_with_noalias(ptr addrspace(1) %ptr) {
; CHECK-LABEL: @vectorize_loads_across_fence_with_noalias(
; CHECK:         load <2 x i32>
; CHECK-NOT:     load i32
  %gep1 = getelementptr i32, ptr addrspace(1) %ptr, i64 1
  %load0 = load i32, ptr addrspace(1) %ptr, align 8, !alias.scope !0, !noalias !3
  fence syncscope("workgroup") release, !noalias !5
  fence syncscope("workgroup") acquire, !noalias !5
  %load1 = load i32, ptr addrspace(1) %gep1, align 4, !alias.scope !0, !noalias !3
  %sum = add i32 %load0, %load1
  call void @use(i32 %sum)
  ret void
}

; Negative test: without !noalias on the fence, vectorization is blocked.
define void @no_vectorize_loads_across_fence_without_noalias(ptr addrspace(1) %ptr) {
; CHECK-LABEL: @no_vectorize_loads_across_fence_without_noalias(
; CHECK:         load i32
; CHECK:         load i32
; CHECK-NOT:     load <2 x i32>
  %gep1 = getelementptr i32, ptr addrspace(1) %ptr, i64 1
  %load0 = load i32, ptr addrspace(1) %ptr, align 8, !alias.scope !0, !noalias !3
  fence syncscope("workgroup") release
  fence syncscope("workgroup") acquire
  %load1 = load i32, ptr addrspace(1) %gep1, align 4, !alias.scope !0, !noalias !3
  %sum = add i32 %load0, %load1
  call void @use(i32 %sum)
  ret void
}

declare void @use(i32)

; Metadata: two noalias scopes in the same domain.
; Loads access memory in "arg_scope". Fences declare !noalias for both scopes,
; meaning they do not concern memory in either scope.
!0 = !{!1}
!1 = distinct !{!1, !2, !"arg_scope"}
!2 = distinct !{!2, !"kernel_domain"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"other_arg_scope"}
!5 = !{!1, !4}
