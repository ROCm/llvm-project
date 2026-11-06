; RUN: opt < %s -passes=mssaargpromotion -S | FileCheck %s

;------- chain of calls
; CHECK-LABEL: define internal i32 @inner_fx(i32 %P.0.val) {
; CHECK-NEXT:   %V = add i32 %P.0.val, 1
; CHECK-NEXT:   ret i32 %V
define internal void @inner_fx(ptr %P) {
  %L = load i32, ptr %P;
  %V = add i32 %L, 1;
  store i32 %V, ptr %P
  ret void
}

; CHECK-LABEL: define internal i32 @outer_fx(i32 %P.0.val) {
; CHECK-NEXT:   %V1 = add i32 %P.0.val, 2
; CHECK-NEXT:   %1 = call i32 @inner_fx(i32 %V1)
; CHECK-NEXT:   %V2 = add i32 %1, 3
; CHECK-NEXT:   ret i32 %V2
define internal void @outer_fx(ptr %P) {
  %L1 = load i32, ptr %P;
  %V1 = add i32 %L1, 2;
  store i32 %V1, ptr %P
  call void @inner_fx(ptr %P)
  %L2 = load i32, ptr %P;
  %V2 = add i32 %L2, 3;
  store i32 %V2, ptr %P
  ret void
}

; CHECK-LABEL: define void @test_chain_of_calls(ptr %P) {
; CHECK-NEXT:   %P.val = load i32, ptr %P, align 4
; CHECK-NEXT:   %1 = call i32 @outer_fx(i32 %P.val)
; CHECK-NEXT:   store i32 %1, ptr %P, align 4
define void @test_chain_of_calls(ptr %P) {
  call void @outer_fx(ptr %P)
  ret void
}


;-------
;CHECK-LABEL: define internal { i32, i32 } @test_not_all_path_store(i1 %c, i32 %P.0.val) {
;CHECK-NEXT:  br i1 %c, label %exit1, label %exit2
;CHECK-LABEL: exit1:
;CHECK-NEXT:   %test_not_all_path_store.exit1.ret = insertvalue { i32, i32 } undef, i32 1, 0
;CHECK-NEXT:   %test_not_all_path_store.exit1.ret1 = insertvalue { i32, i32 } %test_not_all_path_store.exit1.ret, i32 42, 1
;CHECK-NEXT:   ret { i32, i32 } %test_not_all_path_store.exit1.ret1
;CHECK-LABEL: exit2:
;CHECK-NEXT:   %test_not_all_path_store.exit2.ret = insertvalue { i32, i32 } undef, i32 2, 0
;CHECK-NEXT:   %test_not_all_path_store.exit2.ret1 = insertvalue { i32, i32 } %test_not_all_path_store.exit2.ret, i32 %P.0.val, 1
;CHECK-NEXT:   ret { i32, i32 } %test_not_all_path_store.exit2.ret1
define internal i32 @test_not_all_path_store(i1 %c, ptr %P) {
  br i1 %c, label %exit1, label %exit2

exit1:
  store i32 42, ptr %P
  ret i32 1

exit2:
  ret i32 2
}

;CHECK-LABEL: define i32 @test_not_all_path_store_caller(i1 %c) {
;CHECK-NEXT:  %M = alloca i32, align 4
;CHECK-NEXT:  %M.val = load i32, ptr %M, align 4
;CHECK-NEXT:  %R = call { i32, i32 } @test_not_all_path_store(i1 %c, i32 %M.val)
;CHECK-NEXT:  %R.ret = extractvalue { i32, i32 } %R, 0
;CHECK-NEXT:  %M.val.ret = extractvalue { i32, i32 } %R, 1
;CHECK-NEXT:  store i32 %M.val.ret, ptr %M, align 4
;CHECK-NEXT:  %V = load i32, ptr %M, align 4
;CHECK-NEXT:  %Sum = add i32 %R.ret, %V
;CHECK-NEXT:  ret i32 %Sum
define i32 @test_not_all_path_store_caller(i1 %c) {
  %M = alloca i32;
  %R = call i32 @test_not_all_path_store(i1 %c, ptr %M)
  %V = load i32, ptr %M
  %Sum = add i32 %R, %V
  ret i32 %Sum
}

;-------  test that clobber of L2 load by P1 store is detected
;CHECK-LABEL: define internal void @test_getInOutArgClobber_visited
define internal void @test_getInOutArgClobber_visited(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
  br i1 %c, label %left, label %right

;CHECK-LABEL: left:
;CHECK-NEXT: store
left:
  store i32 1, ptr %P1 ; clobbers L2 load
  br i1 %c, label %exit1, label %exit2

right:
  br i1 %c, label %exit1, label %exit2

exit1:
  %L1 = load i32, ptr %P1
  ret void

exit2:
  %L2 = load i32, ptr %P2
  ret void
}

define void @test_getInOutArgClobber_visited_caller(i1 %c, ptr %P2) {
  %M = alloca i32
  call void @test_getInOutArgClobber_visited(i1 %c, ptr %M, ptr %P2);
  ret void
}

;------- check store clobbering other loads

;CHECK-LABEL: define internal void @test_store_clobber1
define internal void @test_store_clobber1(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
;CHECK: store i32 42, ptr %P1
;CHECK: store i32 1, ptr %P3
;CHECK: %V2 = load i32, ptr %P2
  store i32 42, ptr %P1 ; this store clobbers V2 load (e.g. P1 == P2 != P3)
  store i32 1, ptr %P3
  %V2 = load i32, ptr %P2
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P3 store
  ret void
}

define void @test_store_clobber1_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_clobber1(ptr %P1, ptr %P2, ptr %P3)
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_store_no_clobber1
define internal void @test_store_no_clobber1(ptr %P1, ptr %P2) { ; P1 may alias P2
  store i32 42, ptr %P1
  store i32 1, ptr %P2
  %V2 = load i32, ptr %P2
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

;CHECK-LABEL: define void @test_store_no_clobber1_caller
define void @test_store_no_clobber1_caller(ptr %P1, ptr %P2) {
  call void @test_store_no_clobber1(ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_store_no_clobber1
; store P2 and then P1 to preserve order in @test_store_no_clobber1
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_store_diamond_clobber1
define internal void @test_store_diamond_clobber1(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
  br i1 %c, label %st1, label %st2
;CHECK-LABEL: st1:
;CHECK-NEXT: br
st1:
  store i32 42, ptr %P1
  br label %exit

st2:
  br label %exit

exit:
  store i32 1, ptr %P2
  %V2 = load i32, ptr %P2
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

;CHECK-LABEL: define void @test_store_diamond_clobber1_caller
define void @test_store_diamond_clobber1_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_store_diamond_clobber1(i1 %c, ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_store_diamond_clobber1
; store P2 and then P1 to preserve order in @test_store_diamond_clobber1
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_store_diamond_clobber2
define internal void @test_store_diamond_clobber2(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-NEXT: br
  store i32 42, ptr %P1
  store i32 1, ptr %P2
  %V2 = load i32, ptr %P2
  br i1 %c, label %st1, label %st2

st1:
  br label %exit

st2:
  br label %exit

exit:
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

;CHECK-LABEL: define void @test_store_diamond_clobber2_caller
define void @test_store_diamond_clobber2_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_store_diamond_clobber2(i1 %c, ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_store_diamond_clobber2
; store P2 and then P1 to preserve order in @test_store_diamond_clobber2
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_store_diamond_clobber3
define internal void @test_store_diamond_clobber3(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-NEXT: br
  store i32 42, ptr %P1
  store i32 1, ptr %P2
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: br
st1:
  %V2 = load i32, ptr %P2
  br label %exit

st2:
  br label %exit

exit:
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

;CHECK-LABEL: define void @test_store_diamond_clobber3_caller
define void @test_store_diamond_clobber3_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_store_diamond_clobber3(i1 %c, ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_store_diamond_clobber3
; store P2 and then P1 to preserve order in @test_store_diamond_clobber3
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}


;CHECK-LABEL: define internal i32 @test_store_diamond_clobber4
define internal void @test_store_diamond_clobber4(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-NEXT: br
  store i32 42, ptr %P1
  br i1 %c, label %st1, label %st2

st1:
  store i32 1, ptr %P2
  %V2 = load i32, ptr %P2
  br label %exit

st2:
  br label %exit

exit:
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

define void @test_store_diamond_clobber4_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_store_diamond_clobber4(i1 %c, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal void @test_store_diamond_clobber5
define internal void @test_store_diamond_clobber5(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-NEXT: store
  store i32 42, ptr %P1 ; clobbers V2 load on st2 path
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: store
st1:
  store i32 1, ptr %P2
  br label %exit

st2:
  br label %exit

exit:
  %V2 = load i32, ptr %P2
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

define void @test_store_diamond_clobber5_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_store_diamond_clobber5(i1 %c, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal i32 @test_store_diamond_clobber6
define internal void @test_store_diamond_clobber6(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-NEXT: br
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: store i32 1, ptr %P2
st1:
  store i32 42, ptr %P1
  store i32 1, ptr %P2
  br label %exit

st2:
  br label %exit

exit:
  %V2 = load i32, ptr %P2
  store i32 43, ptr %P1 ; this store makes sure that P1 pointee isn't clobbered by P2 store
  ret void
}

define void @test_store_diamond_clobber6_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_store_diamond_clobber6(i1 %c, ptr %P1, ptr %P2)
  ret void
}


;------- check clobbering in diamond

;CHECK-LABEL: define internal void @test_clobber1
define internal void @test_clobber1(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-NEXT: store
  store i32 42, ptr %P2 ; clobbered by P1 stores, cannot promote
  %V1 = load i32, ptr %P1 ; clobbered by P2 store above, cannot promote
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: store
st1:
  store i32 1, ptr %P1
  br label %exit

;CHECK-LABEL: st2:
;CHECK-NEXT: store
st2:
  store i32 2, ptr %P1
  br label %exit

;CHECK-LABEL: exit:
;CHECK-NEXT: load
exit:
  %V2 = load i32, ptr %P1
  ret void
}

define void @test_clobber1_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_clobber1(i1 %c, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_clobber2
define internal void @test_clobber2(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
  %V1 = load i32, ptr %P1 ; no clobber
  store i32 42, ptr %P2 ; clobbered by P1 stores, but unclobbered by P1 promotion
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: br
st1:
  store i32 1, ptr %P1
  br label %exit

;CHECK-LABEL: st2:
;CHECK-NEXT: br
st2:
  store i32 2, ptr %P1
  br label %exit

exit:
  %V2 = load i32, ptr %P1 ; no clobber as every path writes by P1
  ret void
}

;CHECK-LABEL: define void @test_clobber2_caller
define void @test_clobber2_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_clobber2(i1 %c, ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_clobber2
; store P2 and then P1 to preserve order in @test_clobber2
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}


;CHECK-LABEL: define internal i32 @test_clobber3
define internal void @test_clobber3(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
  %V1 = load i32, ptr %P1 ; no clobber
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: store i32 42, ptr %P2
;CHECK-NEXT: br
st1:
  ; P2 isn't selected for promotion: not all paths have stores and
  ; not a valid threal-local ptr
  store i32 42, ptr %P2
  store i32 1, ptr %P1
  br label %exit

;CHECK-LABEL: st2:
;CHECK-NEXT: br
st2:
  store i32 2, ptr %P1
  br label %exit

exit:
  %V2 = load i32, ptr %P1 ; no clobber as every path writes by P1
  ret void
}

;CHECK-LABEL: define void @test_clobber3_caller
define void @test_clobber3_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_clobber3(i1 %c, ptr %P1, ptr %P2)
;CHECK: %1 = call i32 @test_clobber3
;CHECK: store i32 %1, ptr %P1
  ret void
}


;CHECK-LABEL: define internal void @test_clobber4
define internal void @test_clobber4(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
  %V1 = load i32, ptr %P1 ; no clobber
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: store i32 1, ptr %P1
;CHECK-NEXT: store i32 42, ptr %P2
;CHECK-NEXT: br
st1:
  store i32 1, ptr %P1
  ; P2 isn't selected for promotion: not all paths have stores and
  ; not a valid threal-local ptr
  store i32 42, ptr %P2
  br label %exit

;CHECK-LABEL: st2:
;CHECK-NEXT: store i32 2, ptr %P1
st2:
  store i32 2, ptr %P1
  br label %exit

exit:
  %V2 = load i32, ptr %P1 ; clobbered by P2 write
  ret void
}

define void @test_clobber4_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_clobber4(i1 %c, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal void @test_clobber5
define internal void @test_clobber5(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
  %V1 = load i32, ptr %P1 ; no clobber
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: store
st1:
  store i32 1, ptr %P1
  br label %exit

;CHECK-LABEL: st2:
;CHECK-NEXT: store
st2:
  store i32 2, ptr %P1
  br label %exit

exit:
  store i32 42, ptr %P2 ; clobbers V2 load
  %V2 = load i32, ptr %P1 ; clobbered by P2 write
  ret void
}

define void @test_clobber5_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_clobber5(i1 %c, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_clobber6
define internal void @test_clobber6(i1 %c, ptr %P1, ptr %P2) { ; P1 may alias P2
  %V1 = load i32, ptr %P1 ; no clobber
  br i1 %c, label %st1, label %st2

;CHECK-LABEL: st1:
;CHECK-NEXT: br
st1:
  store i32 1, ptr %P1
  br label %exit

;CHECK-LABEL: st2:
;CHECK-NEXT: br
st2:
  store i32 2, ptr %P1
  br label %exit

;CHECK-LABEL: exit:
;CHECK-NOT: load
exit:
  %V2 = load i32, ptr %P1
  ; P1 pointee is clobbered by P2 write, but unclobbered after P2 promotion
  store i32 42, ptr %P2
  ret void
}

;CHECK-LABEL: define void @test_clobber6_caller
define void @test_clobber6_caller(i1 %c, ptr %P1, ptr %P2) {
  call void @test_clobber6(i1 %c, ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_clobber6
;CHECK: store i32 %P1
;CHECK: store i32 %P2
  ret void
}

;------- check clobbering in loops

;CHECK-LABEL: define internal void @test_loop_clobber1
define internal void @test_loop_clobber1(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: store i32 42, ptr %P2
;CHECK-NEXT: %V1 = load i32, ptr %P1
entry:
  store i32 42, ptr %P2 ; clobbers V1 load
  %V1 = load i32, ptr %P1
  store i32 1, ptr %P1 ; clobbers P2 store
  br label %loop_header

loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

loop:
  store i32 2, ptr %P1
  %i.next = sub i32 %i, 1
  br label %loop_header

exit:
  %V2 = load i32, ptr %P1
  ret void
}

define void @test_loop_clobber1_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber1(i32 %n, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_loop_clobber2
define internal void @test_loop_clobber2(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: br
entry:
  %V1 = load i32, ptr %P1
  store i32 42, ptr %P2 ; clobbered by P1 store, but then unclobbered
  store i32 1, ptr %P1
  br label %loop_header

loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

;CHECK-LABEL: loop:
;CHECK-NEXT: %i.next = sub i32 %i, 1
loop:
  store i32 2, ptr %P1
  %i.next = sub i32 %i, 1
  br label %loop_header

exit:
  %V2 = load i32, ptr %P1
  ret void
}

;CHECK-LABEL: define void @test_loop_clobber2_caller
define void @test_loop_clobber2_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber2(i32 %n, ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_loop_clobber2
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}


;CHECK-LABEL: define internal void @test_loop_clobber3
define internal void @test_loop_clobber3(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: %V1 = load i32, ptr %P1
;CHECK-NEXT: store i32 1, ptr %P1
;CHECK-NEXT: store i32 42, ptr %P2
entry:
  %V1 = load i32, ptr %P1
  store i32 1, ptr %P1
  store i32 42, ptr %P2 ; clobbered by P1 store in loop BB
  br label %loop_header

loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

loop:
  store i32 2, ptr %P1
  %i.next = sub i32 %i, 1
  br label %loop_header

exit:
  %V2 = load i32, ptr %P1 ; clobbered by P2 store
  ret void
}

define void @test_loop_clobber3_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber3(i32 %n, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal void @test_loop_clobber4
define internal void @test_loop_clobber4(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: %V1 = load i32, ptr %P1
;CHECK-NEXT: store i32 1, ptr %P1
entry:
  %V1 = load i32, ptr %P1
  store i32 1, ptr %P1
  br label %loop_header

;CHECK-LABEL: loop_header:
;CHECK: store i32 42, ptr %P2
loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  store i32 42, ptr %P2 ; clobbered by P1 store in loop BB
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

loop:
  store i32 2, ptr %P1
  %i.next = sub i32 %i, 1
  br label %loop_header

exit:
  %V2 = load i32, ptr %P1 ; clobbered by P2 store
  ret void
}

define void @test_loop_clobber4_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber4(i32 %n, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal i32 @test_loop_clobber5
define internal void @test_loop_clobber5(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: br
entry:
  %V1 = load i32, ptr %P1
  store i32 1, ptr %P1
  br label %loop_header

;CHECK-LABEL: loop_header:
;CHECK-NEXT: %P1.0.val.loop_header.phi = phi i32 [ 2, %loop ], [ 1, %entry ]
loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

;CHECK-LABEL: loop:
;CHECK-NEXT: store i32 42, ptr %P2
;CHECK-NEXT: %i.next = sub i32 %i, 1
loop:
  store i32 42, ptr %P2 ; not selected for promotion (no stores at every path)
  store i32 2, ptr %P1
  %i.next = sub i32 %i, 1
  br label %loop_header

exit:
  %V2 = load i32, ptr %P1
  ret void
}

;CHECK-LABEL: define void @test_loop_clobber5_caller
define void @test_loop_clobber5_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber5(i32 %n, ptr %P1, ptr %P2)
;CHECK: %1 = call i32 @test_loop_clobber5
;CHECK: store i32 %1, ptr %P1
  ret void
}


;CHECK-LABEL: define internal void @test_loop_clobber6
define internal void @test_loop_clobber6(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: %V1 = load i32, ptr %P1
;CHECK-NEXT: store i32 1, ptr %P1
entry:
  %V1 = load i32, ptr %P1
  store i32 1, ptr %P1
  br label %loop_header

loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

;CHECK-LABEL: loop:
;CHECK-NEXT: store i32 2, ptr %P1
;CHECK-NEXT: store i32 42, ptr %P2
loop:
  store i32 2, ptr %P1
  store i32 42, ptr %P2 ; not selected for promotion (no stores at every path)
  %i.next = sub i32 %i, 1
  br label %loop_header

exit:
  %V2 = load i32, ptr %P1 ; clobbered by P2 store
  ret void
}

define void @test_loop_clobber6_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber6(i32 %n, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal void @test_loop_clobber7
define internal void @test_loop_clobber7(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: %V1 = load i32, ptr %P1
;CHECK-NEXT: store i32 1, ptr %P1
entry:
  %V1 = load i32, ptr %P1
  store i32 1, ptr %P1
  br label %loop_header

loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

loop:
  store i32 2, ptr %P1
  %i.next = sub i32 %i, 1
  br label %loop_header

;CHECK-LABEL: exit:
;CHECK-NEXT: store i32 42, ptr %P2
exit:
  store i32 42, ptr %P2 ; clobbers V2 load
  %V2 = load i32, ptr %P1 ; clobbered by P2 store
  ret void
}

define void @test_loop_clobber7_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber7(i32 %n, ptr %P1, ptr %P2)
  ret void
}


;CHECK-LABEL: define internal { i32, i32 } @test_loop_clobber8
define internal void @test_loop_clobber8(i32 %n, ptr %P1, ptr %P2) { ; P1 may alias P2
;CHECK-LABEL: entry:
;CHECK-NEXT: br
entry:
  %V1 = load i32, ptr %P1
  store i32 1, ptr %P1
  br label %loop_header

loop_header:
  %i = phi i32 [%i.next, %loop], [%n, %entry]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %exit, label %loop

;CHECK-LABEL: loop:
;CHECK-NEXT: %i.next = sub i32 %i, 1
loop:
  store i32 2, ptr %P1
  %i.next = sub i32 %i, 1
  br label %loop_header

exit:
  %V2 = load i32, ptr %P1
  store i32 42, ptr %P2 ; clobbers P1 pointee but it is unclobbered after P2 promotion
  ret void
}

;CHECK-LABEL: define void @test_loop_clobber8_caller
define void @test_loop_clobber8_caller(i32 %n, ptr %P1, ptr %P2) {
  call void @test_loop_clobber8(i32 %n, ptr %P1, ptr %P2)
;CHECK: %1 = call { i32, i32 } @test_loop_clobber8
;CHECK: store i32 %P1
;CHECK: store i32 %P2
  ret void
}

; -----------------------------------------------------------------------------
; Test declobbering sequences

;CHECK-LABEL: define internal { i32, i32, i32 } @test_store_unclobber1
define internal void @test_store_unclobber1(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 1, ptr %P1
  store i32 2, ptr %P2
  store i32 3, ptr %P3
; note that values are inserted in the order of arguments of the function
;CHECK: [[R:%[a-zA-Z0-9_]+]].ret0 = insertvalue { i32, i32, i32 } undef, i32 1, 0
;CHECK-DAG: [[R]].ret1 = insertvalue { i32, i32, i32 } [[R]].ret0, i32 2, 1
;CHECK-DAG: [[R]].ret2 = insertvalue { i32, i32, i32 } [[R]].ret1, i32 3, 2
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber1_caller
define void @test_store_unclobber1_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_unclobber1(ptr %P1, ptr %P2, ptr %P3)
;CHECK: %1 = call { i32, i32, i32 } @test_store_unclobber1
;CHECK: store i32 %P1
;CHECK: store i32 %P2
;CHECK: store i32 %P3
  ret void
}

;CHECK-LABEL: define internal { i32, i32, i32 } @test_store_unclobber2
define internal void @test_store_unclobber2(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 1, ptr %P1
  store i32 3, ptr %P3
  store i32 2, ptr %P2
; note that values are inserted in the order of arguments of the function
;CHECK: [[R:%[a-zA-Z0-9_]+]].ret0 = insertvalue { i32, i32, i32 } undef, i32 1, 0
;CHECK-DAG: [[R]].ret1 = insertvalue { i32, i32, i32 } [[R]].ret0, i32 2, 1
;CHECK-DAG: [[R]].ret2 = insertvalue { i32, i32, i32 } [[R]].ret1, i32 3, 2
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber2_caller
define void @test_store_unclobber2_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_unclobber2(ptr %P1, ptr %P2, ptr %P3)
;CHECK: %1 = call { i32, i32, i32 } @test_store_unclobber2
;CHECK: store i32 %P1
;CHECK: store i32 %P3
;CHECK: store i32 %P2
  ret void
}

;CHECK-LABEL: define internal { i32, i32, i32 } @test_store_unclobber3
define internal void @test_store_unclobber3(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 2, ptr %P2
  store i32 1, ptr %P1
  store i32 3, ptr %P3
; note that values are inserted in the order of arguments of the function
;CHECK: [[R:%[a-zA-Z0-9_]+]].ret0 = insertvalue { i32, i32, i32 } undef, i32 1, 0
;CHECK-DAG: [[R]].ret1 = insertvalue { i32, i32, i32 } [[R]].ret0, i32 2, 1
;CHECK-DAG: [[R]].ret2 = insertvalue { i32, i32, i32 } [[R]].ret1, i32 3, 2
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber3_caller
define void @test_store_unclobber3_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_unclobber3(ptr %P1, ptr %P2, ptr %P3)
;CHECK: %1 = call { i32, i32, i32 } @test_store_unclobber3
;CHECK: store i32 %P2
;CHECK: store i32 %P1
;CHECK: store i32 %P3
  ret void
}

;CHECK-LABEL: define internal { i32, i32, i32 } @test_store_unclobber4
define internal void @test_store_unclobber4(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 2, ptr %P2
  store i32 3, ptr %P3
  store i32 1, ptr %P1
; note that values are inserted in the order of arguments of the function
;CHECK: [[R:%[a-zA-Z0-9_]+]].ret0 = insertvalue { i32, i32, i32 } undef, i32 1, 0
;CHECK-DAG: [[R]].ret1 = insertvalue { i32, i32, i32 } [[R]].ret0, i32 2, 1
;CHECK-DAG: [[R]].ret2 = insertvalue { i32, i32, i32 } [[R]].ret1, i32 3, 2
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber4_caller
define void @test_store_unclobber4_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_unclobber4(ptr %P1, ptr %P2, ptr %P3)
;CHECK: %1 = call { i32, i32, i32 } @test_store_unclobber4
;CHECK: store i32 %P2
;CHECK: store i32 %P3
;CHECK: store i32 %P1
  ret void
}

;CHECK-LABEL: define internal { i32, i32, i32 } @test_store_unclobber5
define internal void @test_store_unclobber5(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 3, ptr %P3
  store i32 1, ptr %P1
  store i32 2, ptr %P2
; note that values are inserted in the order of arguments of the function
;CHECK: [[R:%[a-zA-Z0-9_]+]].ret0 = insertvalue { i32, i32, i32 } undef, i32 1, 0
;CHECK-DAG: [[R]].ret1 = insertvalue { i32, i32, i32 } [[R]].ret0, i32 2, 1
;CHECK-DAG: [[R]].ret2 = insertvalue { i32, i32, i32 } [[R]].ret1, i32 3, 2
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber5_caller
define void @test_store_unclobber5_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_unclobber5(ptr %P1, ptr %P2, ptr %P3)
;CHECK: %1 = call { i32, i32, i32 } @test_store_unclobber5
;CHECK: store i32 %P3
;CHECK: store i32 %P1
;CHECK: store i32 %P2
  ret void
}

;CHECK-LABEL: define internal { i32, i32, i32 } @test_store_unclobber6
define internal void @test_store_unclobber6(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 3, ptr %P3
  store i32 2, ptr %P2
  store i32 1, ptr %P1
; note that values are inserted in the order of arguments of the function
;CHECK: [[R:%[a-zA-Z0-9_]+]].ret0 = insertvalue { i32, i32, i32 } undef, i32 1, 0
;CHECK-DAG: [[R]].ret1 = insertvalue { i32, i32, i32 } [[R]].ret0, i32 2, 1
;CHECK-DAG: [[R]].ret2 = insertvalue { i32, i32, i32 } [[R]].ret1, i32 3, 2
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber6_caller
define void @test_store_unclobber6_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_unclobber6(ptr %P1, ptr %P2, ptr %P3)
;CHECK: %1 = call { i32, i32, i32 } @test_store_unclobber6
;CHECK: store i32 %P3
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}

;CHECK-LABEL: define internal { i32, i32, i32 } @test_store_unclobber6_2x
define internal void @test_store_unclobber6_2x(ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 3, ptr %P3
  store i32 2, ptr %P2
  store i32 1, ptr %P1

  store i32 5, ptr %P3
  store i32 6, ptr %P2
  store i32 4, ptr %P1
; note that values are inserted in the order of arguments of the function
;CHECK: [[R:%[a-zA-Z0-9_]+]].ret0 = insertvalue { i32, i32, i32 } undef, i32 4, 0
;CHECK-DAG: [[R]].ret1 = insertvalue { i32, i32, i32 } [[R]].ret0, i32 6, 1
;CHECK-DAG: [[R]].ret2 = insertvalue { i32, i32, i32 } [[R]].ret1, i32 5, 2
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber6_2x_caller
define void @test_store_unclobber6_2x_caller(ptr %P1, ptr %P2, ptr %P3) {
  call void @test_store_unclobber6_2x(ptr %P1, ptr %P2, ptr %P3)
;CHECK: %1 = call { i32, i32, i32 }
;CHECK: store i32 %P3
;CHECK: store i32 %P2
;CHECK: store i32 %P1
  ret void
}

;CHECK-LABEL: define internal void @test_store_unclobber_fail1
define internal void @test_store_unclobber_fail1(i1 %c, ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 3, ptr %P1
  br i1 %c, label %st1, label %st2
st1:
  store i32 1, ptr %P2
  store i32 2, ptr %P3
  br label %exit
st2:
  store i32 1, ptr %P3
  store i32 2, ptr %P2
  br label %exit
exit:
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber_fail1_caller
define void @test_store_unclobber_fail1_caller(i1 %c, ptr %P1, ptr %P2, ptr %P3) {
; CHECK: call void
  call void @test_store_unclobber_fail1(i1 %c, ptr %P1, ptr %P2, ptr %P3)
  ret void
}


;CHECK-LABEL: define internal void @test_store_unclobber_fail2
define internal void @test_store_unclobber_fail2(i1 %c, ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 1, ptr %P2
  br i1 %c, label %st1, label %st2
st1:
  store i32 3, ptr %P1
  store i32 2, ptr %P3
  br label %exit
st2:
  store i32 1, ptr %P3
  store i32 3, ptr %P1
  br label %exit
exit:
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber_fail2_caller
define void @test_store_unclobber_fail2_caller(i1 %c, ptr %P1, ptr %P2, ptr %P3) {
; CHECK: call void
  call void @test_store_unclobber_fail2(i1 %c, ptr %P1, ptr %P2, ptr %P3)
  ret void
}


;CHECK-LABEL: define internal void @test_store_unclobber_fail3
define internal void @test_store_unclobber_fail3(i1 %c, ptr %P1, ptr %P2, ptr %P3) { ; P1, P2, P3 may alias
  store i32 2, ptr %P3
  br i1 %c, label %st1, label %st2
st1:
  store i32 3, ptr %P1
  store i32 1, ptr %P2
  br label %exit
st2:
  store i32 1, ptr %P2
  store i32 3, ptr %P1
  br label %exit
exit:
  ret void
}

;CHECK-LABEL: define void @test_store_unclobber_fail3_caller
define void @test_store_unclobber_fail3_caller(i1 %c, ptr %P1, ptr %P2, ptr %P3) {
; CHECK: call void
  call void @test_store_unclobber_fail3(i1 %c, ptr %P1, ptr %P2, ptr %P3)
  ret void
}

; -----------------------------------------------------------------------------
; Test declobbering in a more complicated CFG
;CHECK-LABEL: define internal { i32, i32, i32 } @nested_diamond
define internal i32 @nested_diamond(i1 %D1C, i1 %D2C, i32 %X, i32 %Y, i32 *%P1, ptr %P2) {
;        D1
;      /   \
;    D2     \
; D2L  D2R   D1R
;    D2E    /
;      \   /
;       D1E
D1:
  br i1 %D1C, label %D2, label %D1R

D2:
  br i1 %D2C, label %D2L, label %D2R

D2L:
;CHECK-LABEL: D2L:
;CHECK-NEXT: br
  store i32 %Y, ptr %P1
  store i32 %X, ptr %P1
  store i32 %X, ptr %P2
  store i32 %Y, ptr %P2
  br label %D2E

D2R:
;CHECK-LABEL: D2R:
;CHECK-NEXT: br
  store i32 %Y, ptr %P1
  store i32 %X, ptr %P2
  br label %D2E

D2E:
;CHECK-LABEL: D2E:
;CHECK-NEXT: %P1.0.val.D2E.phi = phi i32 [ %Y, %D2R ], [ %X, %D2L ]
;CHECK-NEXT: %P2.0.val.D2E.phi = phi i32 [ %X, %D2R ], [ %Y, %D2L ]
  br label %D1E

D1R:
;CHECK-LABEL: D1R:
;CHECK-NEXT: br
  store i32 %X, ptr %P1
  store i32 %Y, ptr %P2
  br label %D1E

D1E:
;CHECK-LABEL: D1E:
;CHECK-NEXT: %P1.0.val.D1E.phi = phi i32 [ %X, %D1R ], [ %P1.0.val.D2E.phi, %D2E ]
;CHECK-NEXT: %P2.0.val.D1E.phi = phi i32 [ %Y, %D1R ], [ %P2.0.val.D2E.phi, %D2E ]
;CHECK-NEXT: [[R1:%.*]] = insertvalue { i32, i32, i32 } undef, i32 42, 0
;CHECK-NEXT: [[R2:%.*]] = insertvalue { i32, i32, i32 } [[R1]], i32 %P1.0.val.D1E.phi, 1
;CHECK-NEXT: [[R3:%.*]] = insertvalue { i32, i32, i32 } [[R2]], i32 %P2.0.val.D1E.phi, 2
;CHECK-NEXT: ret { i32, i32, i32 } [[R3]]
  ret i32 42
}

;CHECK-LABEL: define i32 @nested_diamond_caller
define i32 @nested_diamond_caller(i1 %D1C, i1 %D2C, i32 %X, i32 %Y, ptr %P1, ptr %P2) {
  %C = call i32 @nested_diamond(i1 %D1C, i1 %D2C, i32 %X, i32 %Y, ptr %P1, ptr %P2)
; CHECK: %C = call { i32, i32, i32 } @nested_diamond(i1 %D1C, i1 %D2C, i32 %X, i32 %Y)
; CHECK-NEXT: %P1.val.ret = extractvalue { i32, i32, i32 } %C, 1
; CHECK-NEXT: store i32 %P1.val.ret, ptr %P1, align 4
; CHECK-NEXT: %P2.val.ret = extractvalue { i32, i32, i32 } %C, 2
; CHECK-NEXT: store i32 %P2.val.ret, ptr %P2, align 4
  %V1 = load i32, ptr %P1
  %V2 = load i32, ptr %P2
  %Sum = add i32 %V1, %V2
  ret i32 %Sum
}
