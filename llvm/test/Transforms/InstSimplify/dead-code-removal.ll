; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=instsimplify -S < %s | FileCheck %s

define void @foo(i1 %arg) nounwind {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    br i1 %arg, label [[TMP1:%.*]], label [[TMP2:%.*]]
; CHECK:       1:
; CHECK-NEXT:    br label [[TMP1]]
; CHECK:       2:
; CHECK-NEXT:    ret void
;
  br i1 %arg, label %1, label %4

; <label>:1                                       ; preds = %1, %0
  %2 = phi i32 [ %3, %1 ], [ undef, %0 ]
  %3 = sub i32 0, undef
  br label %1

; <label>:4                                       ; preds = %0
  ret void
}
