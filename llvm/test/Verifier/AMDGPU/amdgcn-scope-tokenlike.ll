; RUN: split-file %s %t
; RUN: not llvm-as %t/function-arg.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-ARG %s
; RUN: not llvm-as %t/function-return.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-RETURN %s
; RUN: not llvm-as %t/load.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-LOAD %s
; RUN: not llvm-as %t/store.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-STORE %s
; RUN: not llvm-as %t/alloca.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-ALLOCA %s

; In this reduced form target("amdgcn.scope") is token-like: it may only flow
; from a producer intrinsic to a consumer intrinsic within a function.

;--- function-arg.ll
define void @f(target("amdgcn.scope") %scope) {
  ret void
}
; CHECK-ARG: Function takes token but isn't an intrinsic

;--- function-return.ll
define target("amdgcn.scope") @f() {
  ret target("amdgcn.scope") poison
}
; CHECK-RETURN: Function returns a token but isn't an intrinsic

;--- load.ll
define void @f(ptr %p) {
  %scope = load target("amdgcn.scope"), ptr %p
  ret void
}
; CHECK-LOAD: loading unsized types is not allowed

;--- store.ll
define void @f(target("amdgcn.scope") %scope, ptr %p) {
  store target("amdgcn.scope") %scope, ptr %p
  ret void
}
; CHECK-STORE: storing unsized types is not allowed

;--- alloca.ll
define void @f() {
  %p = alloca target("amdgcn.scope")
  ret void
}
; CHECK-ALLOCA: Cannot allocate unsized type
