; Test that the occupancy attributes reach the distributed backends. The index
; file of each module holds the fn-attrs entries of the functions that its
; backend defines or imports.
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernel.ll -o kernel.bc
; RUN: opt -thinlto-bc device.ll -o device.bc
; RUN: llvm-lto2 run kernel.bc device.bc -o out -thinlto-distributed-indexes \
; RUN:   -thinlink-pipeline=amdgpu-thinlink-occupancy \
; RUN:   -r=kernel.bc,kernel,px -r=kernel.bc,device_func,l \
; RUN:   -r=device.bc,device_func,px
; RUN: llvm-dis device.bc.thinlto.bc -o - | FileCheck %s --check-prefix=DEVICE
; RUN: llvm-dis kernel.bc.thinlto.bc -o - | FileCheck %s --check-prefix=KERNEL

; The backend of device.bc defines @device_func.
; DEVICE:      ^[[MOD:[0-9]+]] = module: (path: "device{{(\.thinlink)?}}.bc",
; DEVICE-NEXT: ^[[FN:[0-9]+]] = gv: (guid: {{[0-9]+}}, summaries: (function: (module: ^[[MOD]],
; DEVICE-NEXT: = extension: (name: "amdgpu-implicit-args",
; DEVICE-NEXT: = extension: (name: "fn-attrs", version: 1, required: 1, entries: ((gv: ^[[FN]], module: ^[[MOD]], ints: (0), strings: ("amdgpu-flat-work-group-size", "64,256", "amdgpu-max-num-workgroups", "16,16,1", "amdgpu-waves-per-eu", "2,8"))))

; The backend of kernel.bc imports @device_func. The kernel keeps its own
; attributes, so it has no entry.
; KERNEL:      ^[[MOD:[0-9]+]] = module: (path: "device.bc",
; KERNEL:      ^[[FN:[0-9]+]] = gv: (guid: {{[0-9]+}}, summaries: (function: (module: ^[[MOD]],
; KERNEL:      = extension: (name: "fn-attrs", version: 1, required: 1, entries: ((gv: ^[[FN]], module: ^[[MOD]], ints: (0), strings: ("amdgpu-flat-work-group-size", "64,256", "amdgpu-max-num-workgroups", "16,16,1", "amdgpu-waves-per-eu", "2,8"))))

; The minimized bitcode files for the thin link also hold the facts.
; RUN: opt -thinlto-bc kernel.ll -thin-link-bitcode-file=kernel.thinlink.bc \
; RUN:   -o kernel.bc
; RUN: opt -thinlto-bc device.ll -thin-link-bitcode-file=device.thinlink.bc \
; RUN:   -o device.bc
; RUN: llvm-lto2 run kernel.thinlink.bc device.thinlink.bc -o out \
; RUN:   -thinlto-distributed-indexes -thinlink-pipeline=amdgpu-thinlink-occupancy \
; RUN:   -r=kernel.thinlink.bc,kernel,px \
; RUN:   -r=kernel.thinlink.bc,device_func,l -r=device.thinlink.bc,device_func,px
; RUN: llvm-dis device.thinlink.bc.thinlto.bc -o - | FileCheck %s --check-prefix=DEVICE

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
