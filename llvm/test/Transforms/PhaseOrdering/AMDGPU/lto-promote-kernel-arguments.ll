; Check that a pointer loaded through a kernel argument is promoted to the
; global address space in the full-LTO pipeline.
;
; RUN: opt -mtriple=amdgpu -S -passes='lto<O2>' %s -o - | FileCheck %s
;
; For this to work correctly, AMDGPUPromoteKernelArguments has to run after
; EarlyCSE from the FullLinkTimeOptimizationLast callback.
;
; RUN: opt -mtriple=amdgpu -disable-output -passes='lto<O2>' \
; RUN:   -print-pipeline-passes %s | FileCheck %s --check-prefix=PIPELINE

; PIPELINE: function(early-cse<memssa>),function(amdgpu-promote-kernel-arguments),{{.*}}function(infer-address-spaces)

; CHECK-LABEL: define amdgpu_kernel void @kernel(
; CHECK:         store i32 0, ptr addrspace(1)
define amdgpu_kernel void @kernel(ptr %desc) {
  %p = load ptr, ptr %desc, align 8
  store i32 0, ptr %p, align 4
  ret void
}
