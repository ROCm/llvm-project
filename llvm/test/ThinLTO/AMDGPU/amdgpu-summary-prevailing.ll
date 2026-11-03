; Test that the occupancy pass uses the facts of the copy of a function that
; prevails. Each copy of @f has its own amdgpu-waves-per-eu: "2" in a.bc, and
; "7" in b.bc, which prevails. The kernel has no waves-per-EU limit, so both
; copies of @f get the limit of the copy that prevails. Each index file keeps
; the facts of its own copy.
; RUN: split-file %s %t
; RUN: opt -module-summary < %t/a.ll -o %t/a.bc
; RUN: opt -module-summary < %t/b.ll -o %t/b.bc
; RUN: llvm-lto2 run %t/a.bc %t/b.bc -o %t/o \
; RUN:   -thinlto-distributed-indexes -thinlink-pipeline=amdgpu-thinlink-occupancy \
; RUN:   -r=%t/a.bc,f, -r=%t/a.bc,kernel,px \
; RUN:   -r=%t/b.bc,f,px
; RUN: llvm-dis %t/a.bc.thinlto.bc -o - | FileCheck %s --check-prefix=A
; RUN: llvm-dis %t/b.bc.thinlto.bc -o - | FileCheck %s --check-prefix=B

; A kernel has facts also without occupancy attributes.
; A: ^[[F:[0-9]+]] = gv: (guid: {{[0-9]+}}, summaries: (function: (module: ^0, flags: (linkage: available_externally,
; A: ^[[KERNEL:[0-9]+]] = gv: (guid: {{[0-9]+}}, {{.*}}calls: ((callee: ^[[F]])))))
; A: = extension: (name: "amdgpu-occupancy", version: 1, required: 0, entries: ((gv: ^[[F]], module: ^0, ints: (0), strings: ("amdgpu-waves-per-eu", "2")), (gv: ^[[KERNEL]], module: ^0, ints: (91))))
; A: = extension: (name: "fn-attrs", version: 1, required: 1, entries: ((gv: ^[[F]], module: ^0, ints: (0), strings: ({{.*}}, "amdgpu-waves-per-eu", "7"))))

; B: ^[[F:[0-9]+]] = gv: (guid: {{[0-9]+}}, summaries: (function: (module: ^0, flags: (linkage: weak_odr,
; B: = extension: (name: "amdgpu-occupancy", version: 1, required: 0, entries: ((gv: ^[[F]], module: ^0, ints: (0), strings: ("amdgpu-waves-per-eu", "7"))))
; B: = extension: (name: "fn-attrs", version: 1, required: 1, entries: ((gv: ^[[F]], module: ^0, ints: (0), strings: ({{.*}}, "amdgpu-waves-per-eu", "7"))))

;--- a.ll
target triple = "amdgcn-amd-amdhsa"

define weak_odr void @f() #0 {
  ret void
}

define amdgpu_kernel void @kernel() {
  call void @f()
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="2" }

;--- b.ll
target triple = "amdgcn-amd-amdhsa"

define weak_odr void @f() #0 {
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="7" }
