; Hints place existing scheduling intrinsics in every block that calls a named
; function (here @matrix_op). No debug info is involved. A group pipeline goes
; to the end of the block, a plain barrier before the first matching call, and
; a hint that matches nothing is reported.

; RUN: env PSCHED_LEDGER=%S/Inputs/hints.json %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - 2>/dev/null | %FileCheck %s
; RUN: env PSCHED_LEDGER=%S/Inputs/hints.json %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - 2>&1 >/dev/null | %FileCheck %s --check-prefix=MSG
; RUN: %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - | %FileCheck %s --check-prefix=NONE

; CHECK-LABEL: @kernel(
; CHECK:      entry:
; CHECK-NOT:  amdgcn.sched
; CHECK:      work:
; CHECK-NEXT:   call void @llvm.amdgcn.sched.barrier(i32 0)
; CHECK-NEXT:   call void @matrix_op()
; CHECK-NEXT:   call void @matrix_op()
; CHECK-NEXT:   call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
; CHECK-NEXT:   call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 4, i32 0)
; CHECK-NEXT:   call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
; CHECK-NEXT:   call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 4, i32 0)
; CHECK-NEXT:   ret void

; MSG: psched: warning: hint #2 of entry 'kernel' matched no instruction (stale ledger?)

; NONE-NOT: amdgcn.sched

target triple = "amdgcn-amd-amdhsa"

declare void @matrix_op()

define amdgpu_kernel void @kernel(i1 %c) {
entry:
  br i1 %c, label %work, label %done

work:
  call void @matrix_op()
  call void @matrix_op()
  ret void

done:
  ret void
}
