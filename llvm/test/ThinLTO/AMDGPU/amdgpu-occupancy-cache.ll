; Test that the cache key of a module covers the occupancy attributes that the
; thin link sets on its functions. The key of device.bc changes when the
; attributes of the kernel in the other module change, and only then.
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernel.ll -o kernel.bc
; RUN: opt -thinlto-bc kernel-body.ll -o kernel-body.bc
; RUN: opt -thinlto-bc kernel-attrs.ll -o kernel-attrs.bc
; RUN: opt -thinlto-bc device.ll -o device.bc

; RUN: cp kernel.bc k.bc
; RUN: llvm-lto2 run k.bc device.bc -o out -cache-dir cache \
; RUN:   -r=k.bc,kernel,px -r=k.bc,device_func,l -r=device.bc,device_func,p
; RUN: ls cache/llvmcache-* | count 2

; The kernel has another body and the same attributes. Only k.bc gets a new
; entry.
; RUN: cp kernel-body.bc k.bc
; RUN: llvm-lto2 run k.bc device.bc -o out -cache-dir cache \
; RUN:   -r=k.bc,kernel,px -r=k.bc,device_func,l -r=device.bc,device_func,p
; RUN: ls cache/llvmcache-* | count 3

; The kernel has other attributes. Both modules get a new entry.
; RUN: cp kernel-attrs.bc k.bc
; RUN: llvm-lto2 run k.bc device.bc -o out -cache-dir cache \
; RUN:   -r=k.bc,kernel,px -r=k.bc,device_func,l -r=device.bc,device_func,p
; RUN: ls cache/llvmcache-* | count 5

;--- kernel.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @device_func(ptr)

define amdgpu_kernel void @kernel(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" }

;--- kernel-body.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @device_func(ptr)

define amdgpu_kernel void @kernel(ptr %p) #0 {
  store i32 0, ptr %p
  call void @device_func(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" }

;--- kernel-attrs.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @device_func(ptr)

define amdgpu_kernel void @kernel(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="32,128" }

;--- device.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @device_func(ptr %p) {
  store i32 42, ptr %p
  ret void
}
