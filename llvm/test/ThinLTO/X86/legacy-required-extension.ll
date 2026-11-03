;; Test that the legacy ThinLTO code generator reports an error for a summary
;; index that needs a summary extension: it applies none.

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-lto -thinlto-action=thinlink %t.bc -o %t.index.bc 2>&1 \
; RUN:   | FileCheck %s
; RUN: not llvm-lto -thinlto-action=run %t.bc 2>&1 | FileCheck %s

; CHECK: the summary index needs the summary extension 'test.results' version 1, which the legacy ThinLTO code generator does not apply

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f() {
  ret void
}

^0 = module: (path: "", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "f", summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^2 = extension: (name: "test.results", version: 1, required: 1, entries: ((gv: ^1, module: ^0, strings: ("attr"))))
