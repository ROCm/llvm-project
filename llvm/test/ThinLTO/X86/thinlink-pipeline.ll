;; Test the pipeline of summary passes that runs at the thin link.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt -module-summary callee.ll -o callee.bc

;; Print the call graph after the default passes.
; RUN: llvm-lto2 run main.bc callee.bc -o graph \
; RUN:   -thinlink-pipeline='default,print<summary-callgraph>' \
; RUN:   -r=main.bc,main,px -r=main.bc,caller,px -r=main.bc,rec1,p \
; RUN:   -r=main.bc,rec2,p -r=main.bc,callee, -r=main.bc,external, \
; RUN:   -r=callee.bc,callee,p 2>&1 | FileCheck %s --check-prefix=GRAPH

; GRAPH:      Summary call graph:
; GRAPH-NEXT:   <external callers>
; GRAPH-NEXT:     calls {{[0-9]+}} (main)
; GRAPH-NEXT:     calls {{[0-9]+}} (caller)
; GRAPH-NEXT:   {{[0-9]+}} (rec1) [main.bc]
; GRAPH-NEXT:     calls {{[0-9]+}} (rec2)
; GRAPH-NEXT:   {{[0-9]+}} (external) [unknown]
; GRAPH-NEXT:     calls <external callees>
; GRAPH-NEXT:   {{[0-9]+}} (rec2) [main.bc]
; GRAPH-NEXT:     calls {{[0-9]+}} (rec1)
; GRAPH-NEXT:     calls {{[0-9]+}} (external)
; GRAPH-NEXT:   {{[0-9]+}} (callee) [callee.bc]
; GRAPH-NEXT:   {{[0-9]+}} (main) [main.bc]
; GRAPH-NEXT:     calls {{[0-9]+}} (callee)
; GRAPH-NEXT:     calls {{[0-9]+}} (rec1)
; GRAPH-NEXT:   {{[0-9]+}} (caller) [main.bc]
; GRAPH-NEXT:     calls {{[0-9]+}} (callee)
; GRAPH-NEXT:   <external callees>
; GRAPH-NEXT: SCCs in bottom-up order:
; GRAPH-NEXT:   [{{[0-9]+}} (callee)]
; GRAPH-NEXT:   [<external callees>]
; GRAPH-NEXT:   [{{[0-9]+}} (external)]
; GRAPH-NEXT:   [{{[0-9]+}} (rec2), {{[0-9]+}} (rec1)]
; GRAPH-NEXT:   [{{[0-9]+}} (main)]
; GRAPH-NEXT:   [{{[0-9]+}} (caller)]
; GRAPH-NEXT:   [<external callers>]

;; The pipeline replaces the default pipeline. Without the function attribute
;; pass, the thin link does not find that @caller does not unwind.
; RUN: llvm-lto2 run main.bc callee.bc -o default -save-temps \
; RUN:   -disable-thinlto-funcattrs=0 -thinlink-pipeline=default \
; RUN:   -r=main.bc,main,px -r=main.bc,caller,px -r=main.bc,rec1,p \
; RUN:   -r=main.bc,rec2,p -r=main.bc,callee, -r=main.bc,external, \
; RUN:   -r=callee.bc,callee,p
; RUN: llvm-dis default.1.3.import.bc -o - | FileCheck %s --check-prefix=ATTRS
; RUN: llvm-lto2 run main.bc callee.bc -o noattrs -save-temps \
; RUN:   -disable-thinlto-funcattrs=0 -thinlink-pipeline=thinlink-param-access \
; RUN:   -r=main.bc,main,px -r=main.bc,caller,px -r=main.bc,rec1,p \
; RUN:   -r=main.bc,rec2,p -r=main.bc,callee, -r=main.bc,external, \
; RUN:   -r=callee.bc,callee,p
; RUN: llvm-dis noattrs.1.3.import.bc -o - | FileCheck %s --check-prefix=NOATTRS

; ATTRS: define void @caller() #[[#ATTR:]] {{.*}}{
; ATTRS: attributes #[[#ATTR]] = { norecurse nounwind }
; NOATTRS: define void @caller() {{[^#]*}}{
; NOATTRS-NOT: attributes #

; RUN: not llvm-lto2 run main.bc callee.bc -o error \
; RUN:   -thinlink-pipeline=no-such-pass \
; RUN:   -r=main.bc,main,px -r=main.bc,caller,px -r=main.bc,rec1,p \
; RUN:   -r=main.bc,rec2,p -r=main.bc,callee, -r=main.bc,external, \
; RUN:   -r=callee.bc,callee,p 2>&1 | FileCheck %s --check-prefix=ERROR

; ERROR: unable to parse the thin-link pipeline 'no-such-pass': unknown summary pass 'no-such-pass'

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @callee()
declare void @external()

define void @main() {
  call void @callee()
  call void @rec1()
  ret void
}

define void @caller() {
  call void @callee()
  ret void
}

define void @rec1() {
  call void @rec2()
  ret void
}

define void @rec2() {
  call void @rec1()
  call void @external()
  ret void
}

;--- callee.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @callee() {
  ret void
}
