; Test that the AMDGPU target adds its passes to the end of the default
; pipeline of the thin link. The modules use two spellings of the amdgcn
; triple, and each pass runs once. The link runs ThinLinkParamAccessPass after
; the pipeline. The occupancy pass reads the facts of the
; kernel in the amdgpu-launch-attrs summary extension, and it stores the
; attributes of @device_func in the fn-attrs summary extension.
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernel.ll -o kernel.bc
; RUN: opt -thinlto-bc device.ll -o device.bc
; RUN: llvm-lto2 run kernel.bc device.bc -o out -thinlto-distributed-indexes \
; RUN:   -debug-pass-manager -print-after=amdgpu-thinlink-occupancy \
; RUN:   -r=kernel.bc,kernel,px -r=kernel.bc,device_func,l \
; RUN:   -r=device.bc,device_func,p 2>&1 | FileCheck %s

; CHECK:      Running pass: ThinLinkFunctionAttrsPass on [summary index]
; CHECK-NEXT: Running analysis: ThinLinkInfoAnalysis on [summary index]
; CHECK-NEXT: Running pass: AMDGPUThinLinkOccupancyPass on [summary index]
; CHECK-NEXT: Running analysis: SummaryCallGraphAnalysis on [summary index]
; CHECK-NEXT: Running analysis: SummaryAddressTakenAnalysis on [summary index]
; CHECK-NEXT: ; *** IR Dump After AMDGPUThinLinkOccupancyPass on [summary index] ***
; CHECK:      ^[[MOD:[0-9]+]] = module: (path: "device.bc",
; CHECK:      ^[[KMOD:[0-9]+]] = module: (path: "kernel.bc",
; CHECK:      ^[[FN:[0-9]+]] = gv: (name: "device_func",
; CHECK:      ^[[KERNEL:[0-9]+]] = gv: (name: "kernel",
; CHECK:      = extension: (name: "amdgpu-launch-attrs", version: 1, required: 0, entries: ((gv: ^[[KERNEL]], module: ^[[KMOD]], ints: (91), strings: ("amdgpu-flat-work-group-size", "64,256", "amdgpu-max-num-workgroups", "16,16,1", "amdgpu-waves-per-eu", "2,8"))))
; CHECK:      = extension: (name: "fn-attrs", version: 1, required: 1, entries: ((gv: ^[[FN]], module: ^[[MOD]], ints: (0), strings: ("amdgpu-flat-work-group-size", "64,256", "amdgpu-max-num-workgroups", "16,16,1", "amdgpu-waves-per-eu", "2,8"))))
; CHECK-NEXT: = flags:
; CHECK-NEXT: = blockcount:
; CHECK-NEXT: Running pass: AMDGPUThinLinkUniformWorkGroupSizePass on [summary index]
; CHECK-NEXT: Running pass: AMDGPUThinLinkClusterDimsPass on [summary index]
; CHECK-NEXT: Running pass: AMDGPUThinLinkImplicitArgsPass on [summary index]
; CHECK-NEXT: Running pass: ThinLinkParamAccessPass on [summary index]
; CHECK-NOT:  Running pass

; The pipeline text can name the pass. The link still runs
; ThinLinkParamAccessPass after it.
; RUN: llvm-lto2 run kernel.bc device.bc -o out -thinlto-distributed-indexes \
; RUN:   -thinlink-pipeline=amdgpu-thinlink-occupancy -debug-pass-manager \
; RUN:   -r=kernel.bc,kernel,px -r=kernel.bc,device_func,l \
; RUN:   -r=device.bc,device_func,p 2>&1 | FileCheck %s --check-prefix=PIPELINE

; PIPELINE-NOT: Running pass
; PIPELINE:     Running pass: AMDGPUThinLinkOccupancyPass on [summary index]
; PIPELINE-NOT: Running pass
; PIPELINE:     Running pass: ThinLinkParamAccessPass on [summary index]
; PIPELINE-NOT: Running pass

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
target triple = "amdgcn--amdhsa"

define void @device_func(ptr %p) {
  store i32 42, ptr %p
  ret void
}
