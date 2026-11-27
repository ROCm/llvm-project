; Function attributes: only the scheduling attributes on the whitelist are set,
; only on functions whose name contains "match"; anything else is refused, and
; an entry that matches no function is reported.

; RUN: %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - | %FileCheck %s --check-prefix=NONE
; RUN: env PSCHED_LEDGER=%S/Inputs/attrs.json %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - 2>/dev/null | %FileCheck %s --check-prefix=ATTR
; RUN: env PSCHED_LEDGER=%S/Inputs/attrs.json PSCHED_VERBOSE=1 %opt -load-pass-plugin=%psched-plugin -passes='default<O0>' -S %s -o - 2>&1 >/dev/null | %FileCheck %s --check-prefix=MSG

; NONE-NOT: amdgpu-sched-strategy

; ATTR: define amdgpu_kernel void @kernel_a() #[[A:[0-9]+]]
; ATTR: define amdgpu_kernel void @kernel_b() {{[{]$}}
; ATTR: attributes #[[A]] = { "amdgpu-sched-strategy"="max-ilp" }
; ATTR-NOT: alwaysinline

; MSG-DAG: psched: warning: attribute 'alwaysinline' is not a scheduling attribute; ignored
; MSG-DAG: psched: kernel_a <- amdgpu-sched-strategy=max-ilp
; MSG-DAG: psched: warning: ledger entry 'no_such_function' matched no function (stale ledger?)

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel_a() {
  ret void
}

define amdgpu_kernel void @kernel_b() {
  ret void
}
