; Test that the ThinLTO cache key covers the function flags that the thin link
; propagates. The two links differ only in the callee. main.bc does not import
; it, so only the propagated flags of @caller can change the key of main.bc.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-hash -module-summary main.ll -o main.bc
; RUN: opt -module-hash -module-summary callee-nounwind.ll -o callee1.bc
; RUN: opt -module-hash -module-summary callee-may-unwind.ll -o callee2.bc

; In the first link @callee does not unwind, so @caller becomes nounwind. In
; the second link @callee calls an external function, so @caller does not.
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 -import-instr-limit=0 \
; RUN:   main.bc callee1.bc -o out1 -cache-dir cache \
; RUN:   -r=main.bc,caller,px -r=main.bc,callee, -r=callee1.bc,callee,px
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 -import-instr-limit=0 \
; RUN:   main.bc callee2.bc -o out2 -cache-dir cache \
; RUN:   -r=main.bc,caller,px -r=main.bc,callee, -r=callee2.bc,callee,px \
; RUN:   -r=callee2.bc,external,

; Each link adds one entry for main.bc and one for the callee module.
; RUN: ls cache | count 4

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @callee()

define void @caller() {
  call void @callee()
  ret void
}

;--- callee-nounwind.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @callee() nounwind norecurse {
  ret void
}

;--- callee-may-unwind.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @external()

define void @callee() {
  call void @external()
  ret void
}
