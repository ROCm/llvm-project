; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-max-pi-accesses=4 -S < %s | FileCheck %s
;
; The access cap is per accessing function: @g has 7 accesses spread over 6
; functions and must stay precise under a cap of 4, while @dense has 6 in one
; function and must go pessimistic.

@g = internal global i32 0, align 4
@dense = internal global [8 x i32] zeroinitializer, align 4

define i32 @writer() {
; CHECK-LABEL: @writer(
; CHECK: ret i32 7
  store i32 7, ptr @g, align 4
  %v = load i32, ptr @g, align 4
  ret i32 %v
}

define i32 @r1() {
  %v = load i32, ptr @g, align 4
  ret i32 %v
}

define i32 @r2() {
  %v = load i32, ptr @g, align 4
  ret i32 %v
}

define i32 @r3() {
  %v = load i32, ptr @g, align 4
  ret i32 %v
}

define i32 @r4() {
  %v = load i32, ptr @g, align 4
  ret i32 %v
}

define i32 @r5() {
  %v = load i32, ptr @g, align 4
  ret i32 %v
}

define i32 @dense_fn() {
; CHECK-LABEL: @dense_fn(
; CHECK: load i32
  store i32 1, ptr @dense, align 4
  %p1 = getelementptr inbounds [8 x i32], ptr @dense, i64 0, i64 1
  store i32 2, ptr %p1, align 4
  %p2 = getelementptr inbounds [8 x i32], ptr @dense, i64 0, i64 2
  store i32 3, ptr %p2, align 4
  %p3 = getelementptr inbounds [8 x i32], ptr @dense, i64 0, i64 3
  store i32 4, ptr %p3, align 4
  %p4 = getelementptr inbounds [8 x i32], ptr @dense, i64 0, i64 4
  store i32 5, ptr %p4, align 4
  %v = load i32, ptr @dense, align 4
  ret i32 %v
}
