;; Test that the thin link propagates nounwind through a cycle of functions in
;; two modules. No function in the cycle may throw, so @ping and @pong get
;; nounwind. They are in a cycle, so they do not get norecurse.
;; A function keeps the flags that it has: @unknown_nounwind has an indirect
;; call, but it is nounwind, so its caller @call_unknown_nounwind gets nounwind.

; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %t/a.bc
; RUN: opt -module-summary %t/b.ll -o %t/b.bc
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 %t/a.bc %t/b.bc -o %t1 \
; RUN:   -save-temps -import-instr-limit=0 \
; RUN:   -r=%t/a.bc,ping,px -r=%t/a.bc,pong, \
; RUN:   -r=%t/a.bc,call_unknown_nounwind,px -r=%t/a.bc,unknown_nounwind, \
; RUN:   -r=%t/b.bc,pong,px -r=%t/b.bc,ping, -r=%t/b.bc,unknown_nounwind,px
; RUN: llvm-dis %t1.1.3.import.bc -o - | FileCheck %s --check-prefix=A
; RUN: llvm-dis %t1.2.3.import.bc -o - | FileCheck %s --check-prefix=B

; A: define void @ping(i32 %n) [[NOUNWIND:#[0-9]+]]
; A: define void @call_unknown_nounwind(ptr %fp) [[NOUNWIND]]
; A: attributes [[NOUNWIND]] = { nounwind }

; B: define void @pong(i32 %n) [[NOUNWIND:#[0-9]+]]
; B: attributes [[NOUNWIND]] = { nounwind }

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @pong(i32)
declare void @unknown_nounwind(ptr)

define void @ping(i32 %n) {
  call void @pong(i32 %n)
  ret void
}

define void @call_unknown_nounwind(ptr %fp) {
  call void @unknown_nounwind(ptr %fp)
  ret void
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @ping(i32)

define void @pong(i32 %n) {
  %c = icmp eq i32 %n, 0
  br i1 %c, label %done, label %rec
rec:
  %m = sub i32 %n, 1
  call void @ping(i32 %m)
  br label %done
done:
  ret void
}

define void @unknown_nounwind(ptr %fp) nounwind {
  call void %fp()
  ret void
}
