; Test that the summary holds the AMDGPU occupancy facts in the
; amdgpu-occupancy summary extension, and that the facts reach the index file
; of a distributed backend. A kernel and each function with an occupancy
; attribute get an entry: the calling convention, then the name and the value
; of each occupancy attribute.
; RUN: opt -mtriple=amdgcn-amd-amdhsa -module-summary < %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s
; RUN: llvm-lto2 run %t.bc -o %t.o -thinlto-distributed-indexes \
; RUN:   -r=%t.bc,kernel,px -r=%t.bc,device_func,px -r=%t.bc,plain,px
; RUN: llvm-dis %t.bc.thinlto.bc -o - | FileCheck %s

; The summary of the module has the names of the values. The index file has
; only their GUIDs.
; CHECK: ^[[DEVICE:[0-9]+]] = gv: ({{name: "device_func"|guid: 6995213134000367975}},
; CHECK: ^[[KERNEL:[0-9]+]] = gv: ({{name: "kernel"|guid: 17571550346185820240}},
; CHECK: = extension: (name: "amdgpu-occupancy", version: 1, required: 0, entries: ((gv: ^[[DEVICE]], module: ^0, ints: (0), strings: ("amdgpu-waves-per-eu", "4")), (gv: ^[[KERNEL]], module: ^0, ints: (91), strings: ("amdgpu-flat-work-group-size", "64,256", "amdgpu-max-num-workgroups", "16,16,1", "amdgpu-waves-per-eu", "2,8"))))

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

define void @device_func(ptr %p) #1 {
  store i32 42, ptr %p
  ret void
}

define void @plain() {
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="2,8" "amdgpu-max-num-workgroups"="16,16,1" }
attributes #1 = { "amdgpu-waves-per-eu"="4" }
