; RUN: not opt -mtriple=amdgcn -passes=verify -disable-output %s 2>&1 | FileCheck %s

; CHECK: unknown SQTT event type 99
define void @unknown_type() {
  call void @llvm.amdgcn.sqtt.event(metadata !0, i32 poison, i1 true)
  ret void
}

; CHECK: SQTT event payload must be a string
define void @bad_payload() {
  call void @llvm.amdgcn.sqtt.event(metadata !1, i32 poison, i1 true)
  ret void
}

; CHECK: SQTT event metadata must be a node
define void @not_a_node() {
  call void @llvm.amdgcn.sqtt.event(metadata i32 0, i32 poison, i1 true)
  ret void
}

; CHECK: SQTT event metadata must not be empty
define void @empty() {
  call void @llvm.amdgcn.sqtt.event(metadata !2, i32 poison, i1 true)
  ret void
}

; CHECK: SQTT event metadata must be a 2-operand node
define void @wrong_arity() {
  call void @llvm.amdgcn.sqtt.event(metadata !3, i32 poison, i1 true)
  ret void
}

; CHECK: SQTT event type must be an integer
define void @non_integer_type() {
  call void @llvm.amdgcn.sqtt.event(metadata !4, i32 poison, i1 true)
  ret void
}

; CHECK: unknown SQTT event type 99
define void @merged_with_invalid_event() {
  call void @llvm.amdgcn.sqtt.event(metadata !7, i32 poison, i1 true)
  ret void
}

; CHECK: SQTT SetEventTableBase payload must be an integer
define void @set_base_bad_payload() {
  call void @llvm.amdgcn.sqtt.event(metadata !8, i32 poison, i1 true)
  ret void
}

!0 = !{i32 99, !"bogus"}
!1 = !{i32 0, i32 1}
!2 = !{}
!3 = !{i32 0, !"entry", !"extra"}
!4 = !{!"not-an-int", !"payload"}
!5 = !{i32 0, !"a"}
!6 = !{i32 99, !"b"}
!7 = !{!5, !6}
!8 = !{i32 4, !"not-an-int"}
