; RUN: opt < %s -passes=mssaargpromotion -S | FileCheck %s

; Test that stores to both arguments are promoted and returned.
; The stores happen in the order: P1, P2, P1, so both values need to be returned.

; CHECK-LABEL: define internal { i32, i32 } @test_store_no_clobber1()
define internal void @test_store_no_clobber1(i32* %P1, i32* %P2) {
  store i32 42, i32* %P1
  store i32 1, i32* %P2
  %V2 = load i32, i32* %P2
  store i32 43, i32* %P1
  ret void
; CHECK: %[[RET0:.*]] = insertvalue { i32, i32 } undef, i32 43, 0
; CHECK: %[[RET1:.*]] = insertvalue { i32, i32 } %[[RET0]], i32 1, 1
; CHECK: ret { i32, i32 } %[[RET1]]
}

; CHECK-LABEL: define void @test_store_no_clobber1_caller(ptr %P1, ptr %P2)
define void @test_store_no_clobber1_caller(i32* %P1, i32* %P2) {
  call void @test_store_no_clobber1(i32* %P1, i32* %P2)
; CHECK: %[[CALL:.*]] = call { i32, i32 } @test_store_no_clobber1()
; CHECK: %[[P2VAL:.*]] = extractvalue { i32, i32 } %[[CALL]], 1
; CHECK: store i32 %[[P2VAL]], ptr %P2
; CHECK: %[[P1VAL:.*]] = extractvalue { i32, i32 } %[[CALL]], 0
; CHECK: store i32 %[[P1VAL]], ptr %P1
  ret void
}
