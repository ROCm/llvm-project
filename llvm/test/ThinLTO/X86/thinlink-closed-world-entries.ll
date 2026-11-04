;; Test that in a closed world, the external calling node also calls the
;; functions that a module lists as constructors, destructors, or used values,
;; also through a variable: the runtime can call them. @ctor is a constructor,
;; and @via_used_table is in a used variable. @via_exported_table is in a
;; preserved variable, but in a closed world, code outside the summaries does
;; not call through function pointers.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt nosummary.ll -o nosummary.bc

; RUN: llvm-lto2 run main.bc -o closed -thinlink-closed-world \
; RUN:   -thinlink-pipeline='print<summary-callgraph><indirect-calls>' \
; RUN:   -r=main.bc,entry,px -r=main.bc,via_exported_table,p \
; RUN:   -r=main.bc,helper,p -r=main.bc,exported_table,px 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CLOSED

; CLOSED:      Summary call graph:
; CLOSED-NEXT:   <external callers>
; CLOSED-NEXT:     calls {{[0-9]+}} (entry){{$}}
; CLOSED-NEXT:     calls {{[0-9]+}} (via_used_table) (indirect)
; CLOSED-NEXT:     calls {{[0-9]+}} (ctor) (indirect)
; CLOSED-NEXT:   {{[0-9]+}} (entry) [main.bc]

;; A module without a summary can call any function, so the link does not
;; assume a closed world.
; RUN: llvm-lto2 run main.bc nosummary.bc -o nosummary -thinlink-closed-world \
; RUN:   -thinlink-pipeline='print<summary-callgraph><indirect-calls>' \
; RUN:   -r=main.bc,entry,px -r=main.bc,via_exported_table,p \
; RUN:   -r=main.bc,helper,p -r=main.bc,exported_table,px \
; RUN:   -r=nosummary.bc,plain,px 2>&1 | FileCheck %s --check-prefix=OPEN

; OPEN:      Summary call graph:
; OPEN-NEXT:   <external callers>
; OPEN-NEXT:     calls {{[0-9]+}} (entry) (also indirect)
; OPEN-NEXT:     calls {{[0-9]+}} (via_used_table) (indirect)
; OPEN-NEXT:     calls {{[0-9]+}} (ctor) (indirect)
; OPEN-NEXT:     calls {{[0-9]+}} (via_exported_table) (indirect)
; OPEN-NEXT:   {{[0-9]+}} (entry) [main.bc]

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]
@llvm.used = appending global [1 x ptr] [ptr @used_table], section "llvm.metadata"
@used_table = internal global [1 x ptr] [ptr @via_used_table]
@exported_table = global [1 x ptr] [ptr @via_exported_table]

define void @entry() {
  call void @helper()
  ret void
}

define internal void @ctor() {
  call void @helper()
  ret void
}

define internal void @via_used_table() {
  ret void
}

define void @via_exported_table() {
  ret void
}

define void @helper() {
  ret void
}

;--- nosummary.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @plain() {
  ret void
}
