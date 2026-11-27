; The mul24 choice replaces a 32-bit multiply by the 24-bit one only where both
; operands are proven to fit in 24 signed bits. Anything else is left as it is
; and counted, so a ledger cannot change a result.

; RUN: env PSCHED_LEDGER=%S/Inputs/mul24.json %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - 2>/dev/null | %FileCheck %s
; RUN: env PSCHED_LEDGER=%S/Inputs/mul24.json %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - 2>/dev/null | %FileCheck %s --check-prefix=MARK
; RUN: env PSCHED_LEDGER=%S/Inputs/mul24.json PSCHED_VERBOSE=1 %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - 2>&1 >/dev/null | %FileCheck %s --check-prefix=MSG
; RUN: %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - | %FileCheck %s --check-prefix=NONE

; CHECK-LABEL: @mul_proven(
; CHECK: call i32 @llvm.amdgcn.mul.i24.i32(i32 %a, i32 %b)
; CHECK-NOT: mul i32

; CHECK-LABEL: @mul_unknown(
; CHECK-NOT: call i32 @llvm.amdgcn.mul.i24
; CHECK: mul i32 %x, %y

; CHECK-LABEL: @mul_too_wide(
; CHECK-NOT: call i32 @llvm.amdgcn.mul.i24
; CHECK: mul i32 %a, %b
; CHECK-NEXT: ret i32

; MSG-DAG: psched: mul_proven <- choose mul24: 1 applied, 0 not provable (left unchanged)
; MSG-DAG: psched: mul_unknown <- choose mul24: 0 applied, 1 not provable (left unchanged)
; MSG-DAG: psched: mul_too_wide <- choose mul24: 0 applied, 1 not provable (left unchanged)
; MSG-DAG: psched: warning: unknown choice 'no_such_choice'; ignored

; NONE-NOT: amdgcn.mul.i24

; The function a choice changed is marked, so the backend can warn if it spills; an
; unchanged one is not.
; MARK: define i32 @mul_proven(i8 %p, i16 %q) #[[M:[0-9]+]]
; MARK: define i32 @mul_unknown(i32 %x, i32 %y) #[[U:[0-9]+]]
; MARK: attributes #[[M]] = { "amdgpu-sched-ledger-changed"="mul24" "amdgpu-sched-ledger-tu"="mul24.ll" }
; MARK: attributes #[[U]] = { "amdgpu-sched-ledger-tu"="mul24.ll" }

target triple = "amdgcn-amd-amdhsa"

define i32 @mul_proven(i8 %p, i16 %q) {
  %a = sext i8 %p to i32
  %b = sext i16 %q to i32
  %m = mul i32 %a, %b
  ret i32 %m
}

define i32 @mul_unknown(i32 %x, i32 %y) {
  %m = mul i32 %x, %y
  ret i32 %m
}

; 25 significant bits on one side: not provable.
define i32 @mul_too_wide(i32 %p, i8 %q) {
  %a = and i32 %p, 33554431
  %b = sext i8 %q to i32
  %m = mul i32 %a, %b
  ret i32 %m
}
