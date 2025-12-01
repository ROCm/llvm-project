; RUN: opt < %s -passes=mssaargpromotion -S | FileCheck %s

;CHECK-LABEL: define internal { i32, i32 } @test_store_no_clobber1
define internal void @test_store_no_clobber1(i32* %P1, i32* %P2) { ; P1 may alias P2
  store i32 42, i32* %P1
  store i32 1, i32* %P2
  %V2 = load i32, i32* %P2
  store i32 43, i32* %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

;CHECK-LABEL: define void @test_store_no_clobber1_caller
define void @test_store_no_clobber1_caller(i32* %P1, i32* %P2) {
  call void @test_store_no_clobber1(i32* %P1, i32* %P2)
;CHECK: %1 = call { i32, i32 } @test_store_no_clobber1
; store P2 and then P1 to preserve order in @test_store_no_clobber1
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}
