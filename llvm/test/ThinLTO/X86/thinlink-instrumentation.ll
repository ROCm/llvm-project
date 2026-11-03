;; Test the standard pass instrumentation on the summary passes at the thin
;; link.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt -module-summary callee.ll -o callee.bc

; RUN: llvm-lto2 run main.bc callee.bc -o debug -debug-pass-manager \
; RUN:   -r=main.bc,main,px -r=main.bc,callee, -r=callee.bc,callee,p 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEBUG

; DEBUG:      Running pass: ThinLinkFunctionAttrsPass on [summary index]
; DEBUG-NEXT: Running analysis: ThinLinkInfoAnalysis on [summary index]
; DEBUG-NEXT: Running pass: ThinLinkParamAccessPass on [summary index]

;; -print-after selects a summary pass by its name in the pipeline text.
; RUN: llvm-lto2 run main.bc callee.bc -o print \
; RUN:   -print-after=thinlink-function-attrs \
; RUN:   -r=main.bc,main,px -r=main.bc,callee, -r=callee.bc,callee,p 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PRINT

; PRINT:      ; *** IR Dump After ThinLinkFunctionAttrsPass on [summary index] ***
; PRINT:      ^0 = module: (path: "callee.bc"
; PRINT-NEXT: ^1 = module: (path: "main.bc"
; PRINT-DAG:  = gv: (name: "callee", summaries: (function: (module: ^0,
; PRINT-DAG:  = gv: (name: "main", summaries: (function: (module: ^1,
; PRINT-NOT:  on [summary index]

; RUN: llvm-lto2 run main.bc callee.bc -o all -print-after-all \
; RUN:   -r=main.bc,main,px -r=main.bc,callee, -r=callee.bc,callee,p 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ALL

; ALL: ; *** IR Dump After ThinLinkFunctionAttrsPass on [summary index] ***
; ALL: ; *** IR Dump After ThinLinkParamAccessPass on [summary index] ***

;; A filter of functions does not select the summary index.
; RUN: llvm-lto2 run main.bc callee.bc -o filter -print-after-all \
; RUN:   -filter-print-funcs=main \
; RUN:   -r=main.bc,main,px -r=main.bc,callee, -r=callee.bc,callee,p 2>&1 \
; RUN:   | FileCheck %s --check-prefix=FILTER

; FILTER-NOT: [summary index]
; FILTER:     IR Dump After

;; The change reporters do not support the summary index.
; RUN: llvm-lto2 run main.bc callee.bc -o changed -print-changed \
; RUN:   -r=main.bc,main,px -r=main.bc,callee, -r=callee.bc,callee,p 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHANGED

; CHANGED: *** IR Dump After ThinLinkFunctionAttrsPass on [summary index] filtered out ***

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @callee()

define void @main() {
  call void @callee()
  ret void
}

;--- callee.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @callee() {
  ret void
}
