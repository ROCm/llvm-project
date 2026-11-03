; Test that the summaries that the writer builds again for a module with type
; metadata get the AMDGPU occupancy facts, like the summary that
; ModuleSummaryIndexAnalysis builds.

; The writer promotes the type IDs and builds the summary again.
; RUN: opt -thinlto-bc %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

; The writer splits the module. The kernel is in the ThinLTO part.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t.split.bc
; RUN: llvm-modextract -b -n 0 %t.split.bc -o - | llvm-dis -o - | FileCheck %s

; CHECK: ^[[KERNEL:[0-9]+]] = gv: (name: "kernel",
; CHECK: = extension: (name: "amdgpu-occupancy", version: 1, required: 0, entries: ((gv: ^[[KERNEL]], module: ^0, ints: (91), strings: ("amdgpu-flat-work-group-size", "64,256"))))

target triple = "amdgcn-amd-amdhsa"

@vt = constant [1 x ptr] [ptr @vf], !type !0

define i32 @vf(ptr %this) {
  ret i32 0
}

define amdgpu_kernel void @kernel() #0 {
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" }

!0 = !{i32 0, !"typeid"}
