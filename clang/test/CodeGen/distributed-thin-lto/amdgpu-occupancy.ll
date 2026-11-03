; REQUIRES: amdgpu-registered-target

;; Test that a distributed ThinLTO backend sets the function attributes that the
;; thin link stores in the index file. An AMDGPU device function gets the
;; occupancy attributes of the kernel in another module that calls it.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernel.ll -o kernel.bc
; RUN: opt -thinlto-bc device.ll -o device.bc
; RUN: llvm-lto2 run kernel.bc device.bc -o out -thinlto-distributed-indexes \
; RUN:   -r=kernel.bc,kernel,px -r=kernel.bc,device_func,l \
; RUN:   -r=device.bc,device_func,px
; RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -O2 -emit-llvm \
; RUN:   -fthinlto-index=device.bc.thinlto.bc -x ir device.bc -o - | FileCheck %s

; CHECK: define {{.*}}void @device_func({{.*}}) {{.*}}#[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="64,256"
; CHECK-SAME: "amdgpu-max-num-workgroups"="16,16,1"
; CHECK-SAME: "amdgpu-waves-per-eu"="2,8"

;--- kernel.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @device_func(ptr)

define amdgpu_kernel void @kernel(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="2,8" "amdgpu-max-num-workgroups"="16,16,1" }

;--- device.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @device_func(ptr %p) {
  store i32 42, ptr %p
  ret void
}
