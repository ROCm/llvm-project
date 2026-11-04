; Test that occupancy attributes propagate through indirect calls. An indirect
; call can reach every function with the same call type identifier whose
; address can be taken: some summary references it, or it is visible outside
; the summaries (for example to native objects).
;
; kernel1:      flat_wg=[64,256], waves=[2,8], max_wg=[16,16,1]
;               indirect call through a void(ptr) pointer
; kernel2:      flat_wg=[32,128], waves=[4,6], max_wg=[8,8,8]
;               direct calls to @both and @not_taken
; kernel_taken: flat_wg=[1,64], waves=[3,5], max_wg=[4,4,4]
;               its address is taken; direct call to @helper
;
; Targets of the indirect call in kernel1:
; - @target and @both are void(ptr) functions in @table. @both gets the merge
;   of kernel1 and kernel2 (as in amdgpu-occupancy-merge.ll).
; - @exported is a void(ptr) function that no summary references, but it is
;   visible to regular objects (resolution x), so native code can take its
;   address. It gets the same attributes as @target.
;
; Not targets:
; - @other_sig is in @table, but it has another signature. It has the same
;   attributes as @unreferenced, which no kernel can reach.
; - @not_taken has the same signature, but its address cannot be taken. It
;   gets only the attributes of kernel2.
; - @kernel_taken has the same type, but a kernel cannot be called. So
;   @helper gets only the attributes of kernel_taken.
;
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/kernels.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/device.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o \
; RUN:   -r %t1.bc,kernel1,px \
; RUN:   -r %t1.bc,kernel2,px \
; RUN:   -r %t1.bc,kernel_taken,px \
; RUN:   -r %t1.bc,prefetch_table,px \
; RUN:   -r %t1.bc,both,l \
; RUN:   -r %t1.bc,not_taken,l \
; RUN:   -r %t1.bc,helper,l \
; RUN:   -r %t2.bc,target,px \
; RUN:   -r %t2.bc,other_sig,px \
; RUN:   -r %t2.bc,exported,px \
; RUN:   -r %t2.bc,not_taken,p \
; RUN:   -r %t2.bc,both,px \
; RUN:   -r %t2.bc,helper,p \
; RUN:   -r %t2.bc,unreferenced,px \
; RUN:   -r %t2.bc,table,px \
; RUN:   -save-temps
; RUN: llvm-dis %t.o.2.5.precodegen.bc -o - | FileCheck %s

; CHECK: define {{.*}}void @target({{.*}}) {{.*}}#[[TARGET:[0-9]+]]
; CHECK: define {{.*}}void @other_sig({{.*}}) {{.*}}#[[NONE:[0-9]+]]
; CHECK: define {{.*}}void @exported({{.*}}) {{.*}}#[[TARGET]]
; CHECK: define {{.*}}void @not_taken({{.*}}) {{.*}}#[[NOT_TAKEN:[0-9]+]]
; CHECK: define {{.*}}void @both({{.*}}) {{.*}}#[[BOTH:[0-9]+]]
; CHECK: define {{.*}}void @helper({{.*}}) {{.*}}#[[HELPER:[0-9]+]]
; CHECK: define {{.*}}void @unreferenced({{.*}}) {{.*}}#[[NONE]]

; CHECK: attributes #[[TARGET]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="64,256"
; CHECK-SAME: "amdgpu-max-num-workgroups"="16,16,1"
; CHECK-SAME: "amdgpu-waves-per-eu"="2,8"
; CHECK: attributes #[[NOT_TAKEN]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="32,128"
; CHECK-SAME: "amdgpu-max-num-workgroups"="8,8,8"
; CHECK-SAME: "amdgpu-waves-per-eu"="4,6"
; CHECK: attributes #[[BOTH]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="32,256"
; CHECK-SAME: "amdgpu-max-num-workgroups"="16,16,8"
; CHECK-SAME: "amdgpu-waves-per-eu"="4,8"
; CHECK: attributes #[[HELPER]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="1,64"
; CHECK-SAME: "amdgpu-max-num-workgroups"="4,4,4"
; CHECK-SAME: "amdgpu-waves-per-eu"="3,5"

;--- kernels.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@prefetch_table = global ptr @kernel_taken

declare void @both(ptr)
declare void @not_taken(ptr)
declare void @helper(ptr)

define amdgpu_kernel void @kernel1(ptr %fp, ptr %p) #0 {
  call void %fp(ptr %p)
  ret void
}

define amdgpu_kernel void @kernel2(ptr %p) #1 {
  call void @both(ptr %p)
  call void @not_taken(ptr %p)
  ret void
}

define amdgpu_kernel void @kernel_taken(ptr %p) #2 {
  call void @helper(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="2,8" "amdgpu-max-num-workgroups"="16,16,1" }
attributes #1 = { "amdgpu-flat-work-group-size"="32,128" "amdgpu-waves-per-eu"="4,6" "amdgpu-max-num-workgroups"="8,8,8" }
attributes #2 = { "amdgpu-flat-work-group-size"="1,64" "amdgpu-waves-per-eu"="3,5" "amdgpu-max-num-workgroups"="4,4,4" }

;--- device.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@table = global [3 x ptr] [ptr @target, ptr @other_sig, ptr @both]

define void @target(ptr %p) {
  ret void
}

define void @other_sig(i32 %x) {
  ret void
}

define void @exported(ptr %p) {
  ret void
}

define void @not_taken(ptr %p) {
  ret void
}

define void @both(ptr %p) {
  ret void
}

define void @helper(ptr %p) {
  ret void
}

define void @unreferenced(i32 %x) {
  ret void
}
