;; Test the summary call graph with and without the indirect calls, in an open
;; world and in a closed world. @caller has an indirect call. @target has the
;; call type ID of the call, and @table takes its address. @other has the same
;; ID, but its address is not taken. @asm_caller calls inline assembly, which
;; has no call type ID. It is visible outside the summaries, so code outside can
;; take its address, and it is also a possible target of the indirect call.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt -module-summary target.ll -o target.bc

;; Without the indirect calls, the indirect call calls the calls external node,
;; and the external calling node calls each function whose address can be
;; taken. The world does not change this form.
; RUN: llvm-lto2 run main.bc target.bc -o out \
; RUN:   -thinlink-pipeline='print<summary-callgraph>' \
; RUN:   -r=main.bc,caller,px -r=main.bc,asm_caller,px -r=main.bc,table,px \
; RUN:   -r=main.bc,target, -r=main.bc,other, -r=target.bc,target,p \
; RUN:   -r=target.bc,other,p 2>&1 | FileCheck %s --check-prefix=CALLS
; RUN: llvm-lto2 run main.bc target.bc -o out -thinlink-closed-world \
; RUN:   -thinlink-pipeline='print<summary-callgraph>' \
; RUN:   -r=main.bc,caller,px -r=main.bc,asm_caller,px -r=main.bc,table,px \
; RUN:   -r=main.bc,target, -r=main.bc,other, -r=target.bc,target,p \
; RUN:   -r=target.bc,other,p 2>&1 | FileCheck %s --check-prefix=CALLS

; CALLS:      Summary call graph:
; CALLS-NEXT:   <external callers>
; CALLS-NEXT:     calls {{[0-9]+}} (asm_caller) (also indirect)
; CALLS-NEXT:     calls {{[0-9]+}} (caller) (also indirect)
; CALLS-NEXT:     calls {{[0-9]+}} (target) (indirect)
; CALLS-NEXT:   {{[0-9]+}} (asm_caller) [main.bc]
; CALLS-NEXT:     calls <external callees> (indirect)
; CALLS-NEXT:   {{[0-9]+}} (other) [target.bc]
; CALLS-NEXT:   {{[0-9]+}} (target) [target.bc]
; CALLS-NEXT:   {{[0-9]+}} (caller) [main.bc]
; CALLS-NEXT:     calls {{[0-9]+}} (other)
; CALLS-NEXT:     calls <external callees> (indirect)
; CALLS-NEXT:   <external callees>
; CALLS-NEXT: SCCs in bottom-up order:

;; With the indirect calls, the indirect call also calls @target and
;; @asm_caller. In an open world, the external calling node still calls the
;; functions whose address can be taken, because their address can escape to
;; code outside the summaries, and the indirect call can reach code outside the
;; summaries.
; RUN: llvm-lto2 run main.bc target.bc -o out \
; RUN:   -thinlink-pipeline='print<summary-callgraph><indirect-calls>' \
; RUN:   -r=main.bc,caller,px -r=main.bc,asm_caller,px -r=main.bc,table,px \
; RUN:   -r=main.bc,target, -r=main.bc,other, -r=target.bc,target,p \
; RUN:   -r=target.bc,other,p 2>&1 | FileCheck %s --check-prefix=OPEN

; OPEN:      Summary call graph:
; OPEN-NEXT:   <external callers>
; OPEN-NEXT:     calls {{[0-9]+}} (asm_caller) (also indirect)
; OPEN-NEXT:     calls {{[0-9]+}} (caller) (also indirect)
; OPEN-NEXT:     calls {{[0-9]+}} (target) (indirect)
; OPEN-NEXT:   {{[0-9]+}} (asm_caller) [main.bc]
; OPEN-NEXT:     calls <external callees> (indirect)
; OPEN-NEXT:   {{[0-9]+}} (other) [target.bc]
; OPEN-NEXT:   {{[0-9]+}} (target) [target.bc]
; OPEN-NEXT:   {{[0-9]+}} (caller) [main.bc]
; OPEN-NEXT:     calls {{[0-9]+}} (other)
; OPEN-NEXT:     calls {{[0-9]+}} (target) (indirect)
; OPEN-NEXT:     calls {{[0-9]+}} (asm_caller) (indirect)
; OPEN-NEXT:     calls <external callees> (indirect)
; OPEN-NEXT:   <external callees>
; OPEN-NEXT: SCCs in bottom-up order:

;; In a closed world, code outside the summaries calls only the preserved
;; symbols, and the indirect call calls only its targets. The call to inline
;; assembly still calls the calls external node.
; RUN: llvm-lto2 run main.bc target.bc -o out -thinlink-closed-world \
; RUN:   -thinlink-pipeline='print<summary-callgraph><indirect-calls>' \
; RUN:   -r=main.bc,caller,px -r=main.bc,asm_caller,px -r=main.bc,table,px \
; RUN:   -r=main.bc,target, -r=main.bc,other, -r=target.bc,target,p \
; RUN:   -r=target.bc,other,p 2>&1 | FileCheck %s --check-prefix=CLOSED

; CLOSED:      Summary call graph:
; CLOSED-NEXT:   <external callers>
; CLOSED-NEXT:     calls {{[0-9]+}} (asm_caller){{$}}
; CLOSED-NEXT:     calls {{[0-9]+}} (caller){{$}}
; CLOSED-NEXT:   {{[0-9]+}} (asm_caller) [main.bc]
; CLOSED-NEXT:     calls <external callees> (indirect)
; CLOSED-NEXT:   {{[0-9]+}} (other) [target.bc]
; CLOSED-NEXT:   {{[0-9]+}} (target) [target.bc]
; CLOSED-NEXT:   {{[0-9]+}} (caller) [main.bc]
; CLOSED-NEXT:     calls {{[0-9]+}} (other)
; CLOSED-NEXT:     calls {{[0-9]+}} (target) (indirect)
; CLOSED-NEXT:     calls {{[0-9]+}} (asm_caller) (indirect)
; CLOSED-NEXT:   <external callees>
; CLOSED-NEXT: SCCs in bottom-up order:

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@table = global ptr @target

declare void @target()
declare void @other()

define void @caller(ptr %fp) {
  call void %fp()
  call void @other()
  ret void
}

define void @asm_caller() {
  call void asm sideeffect "nop", ""()
  ret void
}

;--- target.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @target() {
  ret void
}

define void @other() {
  ret void
}
