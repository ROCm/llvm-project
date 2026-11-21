;; Test that the thin link computes the parameter access ranges of the stack
;; safety analysis also after a custom pipeline of thin-link passes. @caller
;; passes its parameter to @callee, which writes 4 bytes. The range of @caller
;; alone is empty, with a call that is not resolved. A backend reads only the
;; range, so without the thin-link step it would take the parameter as not
;; accessed.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt -module-summary callee.ll -o callee.bc

; RUN: llvm-lto2 run main.bc callee.bc -o default -thinlto-distributed-indexes \
; RUN:   -r=main.bc,caller,plx -r=main.bc,callee, -r=callee.bc,callee,plx
; RUN: llvm-dis main.bc.thinlto.bc -o - | FileCheck %s
; RUN: llvm-lto2 run main.bc callee.bc -o custom -thinlto-distributed-indexes \
; RUN:   -thinlink-pipeline=thinlink-function-attrs \
; RUN:   -r=main.bc,caller,plx -r=main.bc,callee, -r=callee.bc,callee,plx
; RUN: llvm-dis main.bc.thinlto.bc -o - | FileCheck %s

; CHECK: calls: ((callee: ^{{[0-9]+}})), {{.*}}params: ((param: 0, offset: [0, 3])))))

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @callee(ptr)

define void @caller(ptr %p) sanitize_memtag {
  call void @callee(ptr %p)
  ret void
}

;--- callee.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @callee(ptr %p) sanitize_memtag {
  store i32 0, ptr %p
  ret void
}
