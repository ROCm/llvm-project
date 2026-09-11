; Check that the loop of a static worksharing construct, and only that loop, is
; marked for openmp-kernel-traffic: the outermost loop between the call that
; computes a thread's share and the call that ends the construct, or, for the
; entry points that take the loop body as a callback, the outermost loop in the
; entry point that calls it.
;
; RUN: opt -passes=openmp-mark-work-loops -S < %s | FileCheck %s

declare void @__kmpc_for_static_init_8u(ptr, i32)
declare void @__kmpc_for_static_fini(ptr, i32)
declare void @__kmpc_distribute_static_init_8u(ptr, i32)
declare void @__kmpc_distribute_static_fini(ptr, i32)

; Loops before the construct and after it are not marked. Existing loop
; properties are kept.
; CHECK-LABEL: define void @for_loop(
; CHECK:         br i1 %before.c, label %before, label %init{{$}}
; CHECK:         br i1 %work.c, label %work, label %fini, !llvm.loop [[FOR_LOOP:![0-9]+]]
; CHECK:         br i1 %after.c, label %after, label %exit{{$}}
define void @for_loop(i64 %n) {
entry:
  br label %before

before:
  %b = phi i64 [ 0, %entry ], [ %b.next, %before ]
  %b.next = add nuw i64 %b, 1
  %before.c = icmp ult i64 %b.next, 4
  br i1 %before.c, label %before, label %init

init:
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %work

work:
  %i = phi i64 [ 0, %init ], [ %i.next, %work ]
  %i.next = add nuw i64 %i, 1
  %work.c = icmp ult i64 %i.next, %n
  br i1 %work.c, label %work, label %fini, !llvm.loop !0

fini:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  br label %after

after:
  %a = phi i64 [ 0, %fini ], [ %a.next, %after ]
  %a.next = add nuw i64 %a, 1
  %after.c = icmp ult i64 %a.next, 64
  br i1 %after.c, label %after, label %exit

exit:
  ret void
}

; A chunked schedule nests the loop over a chunk in the loop over the chunks:
; only the outer loop is marked, the inner one belongs to it.
; CHECK-LABEL: define void @chunked(
; CHECK:         br i1 %inner.c, label %inner, label %latch{{$}}
; CHECK:         br i1 %outer.c, label %outer, label %fini, !llvm.loop [[CHUNKED:![0-9]+]]
define void @chunked(i64 %n) {
entry:
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %latch ]
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
  %i.next = add nuw i64 %i, 1
  %inner.c = icmp ult i64 %i.next, 16
  br i1 %inner.c, label %inner, label %latch

latch:
  %c.next = add nuw i64 %c, 1
  %outer.c = icmp ult i64 %c.next, %n
  br i1 %outer.c, label %outer, label %fini

fini:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  ret void
}

; A loop around the whole construct is not its loop; the one inside it is.
; CHECK-LABEL: define void @around(
; CHECK:         br i1 %work.c, label %work, label %fini, !llvm.loop [[AROUND:![0-9]+]]
; CHECK:         br i1 %repeat.c, label %repeat, label %exit{{$}}
define void @around(i64 %n) {
entry:
  br label %repeat

repeat:
  %t = phi i64 [ 0, %entry ], [ %t.next, %fini ]
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %work

work:
  %i = phi i64 [ 0, %repeat ], [ %i.next, %work ]
  %i.next = add nuw i64 %i, 1
  %work.c = icmp ult i64 %i.next, %n
  br i1 %work.c, label %work, label %fini

fini:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  %t.next = add nuw i64 %t, 1
  %repeat.c = icmp ult i64 %t.next, 8
  br i1 %repeat.c, label %repeat, label %exit

exit:
  ret void
}

; The distribute construct is lowered the same way.
; CHECK-LABEL: define void @distribute(
; CHECK:         br i1 %work.c, label %work, label %fini, !llvm.loop [[DISTRIBUTE:![0-9]+]]
define void @distribute(i64 %n) {
entry:
  call void @__kmpc_distribute_static_init_8u(ptr null, i32 0)
  br label %work

work:
  %i = phi i64 [ 0, %entry ], [ %i.next, %work ]
  %i.next = add nuw i64 %i, 1
  %work.c = icmp ult i64 %i.next, %n
  br i1 %work.c, label %work, label %fini

fini:
  call void @__kmpc_distribute_static_fini(ptr null, i32 0)
  ret void
}

; A loop that is not certain to end in the construct's end is not its loop.
; CHECK-LABEL: define void @bypassed_fini(
; CHECK:         br i1 %work.c, label %work, label %done{{$}}
define void @bypassed_fini(i64 %n, i1 %skip) {
entry:
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %work

work:
  %i = phi i64 [ 0, %entry ], [ %i.next, %work ]
  %i.next = add nuw i64 %i, 1
  %work.c = icmp ult i64 %i.next, %n
  br i1 %work.c, label %work, label %done

done:
  br i1 %skip, label %exit, label %fini

fini:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  br label %exit

exit:
  ret void
}

; A loop between the end of one construct and the start of the next belongs to
; neither of them.
; CHECK-LABEL: define void @between_constructs(
; CHECK:         br i1 %work1.c, label %work1, label %fini1, !llvm.loop [[BETWEEN1:![0-9]+]]
; CHECK:         br i1 %seq.c, label %seq, label %init2{{$}}
; CHECK:         br i1 %work2.c, label %work2, label %fini2, !llvm.loop [[BETWEEN2:![0-9]+]]
define void @between_constructs(i64 %n) {
entry:
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %work1

work1:
  %i = phi i64 [ 0, %entry ], [ %i.next, %work1 ]
  %i.next = add nuw i64 %i, 1
  %work1.c = icmp ult i64 %i.next, %n
  br i1 %work1.c, label %work1, label %fini1

fini1:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  br label %seq

seq:
  %s = phi i64 [ 0, %fini1 ], [ %s.next, %seq ]
  %s.next = add nuw i64 %s, 1
  %seq.c = icmp ult i64 %s.next, %n
  br i1 %seq.c, label %seq, label %init2

init2:
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %work2

work2:
  %j = phi i64 [ 0, %init2 ], [ %j.next, %work2 ]
  %j.next = add nuw i64 %j, 1
  %work2.c = icmp ult i64 %j.next, %n
  br i1 %work2.c, label %work2, label %fini2

fini2:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  ret void
}

; The end of one construct and the start of the next can share a block. Both
; loops are still marked.
; CHECK-LABEL: define void @adjacent_constructs(
; CHECK:         br i1 %work1.c, label %work1, label %next, !llvm.loop [[ADJACENT1:![0-9]+]]
; CHECK:         br i1 %work2.c, label %work2, label %fini2, !llvm.loop [[ADJACENT2:![0-9]+]]
define void @adjacent_constructs(i64 %n) {
entry:
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %work1

work1:
  %i = phi i64 [ 0, %entry ], [ %i.next, %work1 ]
  %i.next = add nuw i64 %i, 1
  %work1.c = icmp ult i64 %i.next, %n
  br i1 %work1.c, label %work1, label %next

next:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  call void @__kmpc_for_static_init_8u(ptr null, i32 0)
  br label %work2

work2:
  %j = phi i64 [ 0, %next ], [ %j.next, %work2 ]
  %j.next = add nuw i64 %j, 1
  %work2.c = icmp ult i64 %j.next, %n
  br i1 %work2.c, label %work2, label %fini2

fini2:
  call void @__kmpc_for_static_fini(ptr null, i32 0)
  ret void
}

; An entry point that takes the loop body as a callback runs the loop itself.
; The loop that calls the body is marked, other loops in it are not. In the
; chunked variant, the loop over a chunk belongs to the loop over the chunks.
; CHECK-LABEL: define void @__kmpc_distribute_for_static_loop_4u(
; CHECK:         br i1 %spin.c, label %spin, label %outer{{$}}
; CHECK:         br i1 %inner.c, label %inner, label %latch{{$}}
; CHECK:         br i1 %outer.c, label %outer, label %exit, !llvm.loop [[CALLBACK:![0-9]+]]
define void @__kmpc_distribute_for_static_loop_4u(ptr %loc, ptr %fn, ptr %arg, i32 %n, i1 %flag) {
entry:
  br label %spin

spin:
  %spin.c = phi i1 [ %flag, %entry ], [ false, %spin ]
  br i1 %spin.c, label %spin, label %outer

outer:
  %iv = phi i32 [ 0, %spin ], [ %iv.next, %latch ]
  br label %inner

inner:
  %c = phi i32 [ %iv, %outer ], [ %c.next, %inner ]
  call void %fn(i32 %c, ptr %arg)
  %c.next = add nuw i32 %c, 1
  %inner.c = icmp ult i32 %c.next, 4
  br i1 %inner.c, label %inner, label %latch

latch:
  %iv.next = add nuw i32 %iv, 64
  %outer.c = icmp ult i32 %iv.next, %n
  br i1 %outer.c, label %outer, label %exit

exit:
  ret void
}

; A function of the same shape that is no such entry point is left alone.
; CHECK-LABEL: define void @apply_loop_body(
; CHECK:         br i1 %loop.c, label %loop, label %exit{{$}}
define void @apply_loop_body(ptr %loc, ptr %fn, ptr %arg, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void %fn(i32 %i, ptr %arg)
  %i.next = add nuw i32 %i, 1
  %loop.c = icmp ult i32 %i.next, %n
  br i1 %loop.c, label %loop, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}

; CHECK-DAG: [[FOR_LOOP]] = {{(distinct )?}}!{[[FOR_LOOP]], [[MUSTPROGRESS:![0-9]+]], [[WORK:![0-9]+]]}
; CHECK-DAG: [[MUSTPROGRESS]] = !{!"llvm.loop.mustprogress"}
; CHECK-DAG: [[WORK]] = !{!"llvm.loop.omp.work"}
; CHECK-DAG: [[CHUNKED]] = {{(distinct )?}}!{[[CHUNKED]], [[WORK]]}
; CHECK-DAG: [[AROUND]] = {{(distinct )?}}!{[[AROUND]], [[WORK]]}
; CHECK-DAG: [[DISTRIBUTE]] = {{(distinct )?}}!{[[DISTRIBUTE]], [[WORK]]}
; CHECK-DAG: [[CALLBACK]] = {{(distinct )?}}!{[[CALLBACK]], [[WORK]]}
; CHECK-DAG: [[BETWEEN1]] = {{(distinct )?}}!{[[BETWEEN1]], [[WORK]]}
; CHECK-DAG: [[BETWEEN2]] = {{(distinct )?}}!{[[BETWEEN2]], [[WORK]]}
; CHECK-DAG: [[ADJACENT1]] = {{(distinct )?}}!{[[ADJACENT1]], [[WORK]]}
; CHECK-DAG: [[ADJACENT2]] = {{(distinct )?}}!{[[ADJACENT2]], [[WORK]]}
