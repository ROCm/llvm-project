; Test that a function that a global constructor calls gets the attributes of
; an unknown launch, also in a closed world. After the thin link, the backend
; calls the constructors from a new kernel (amdgcn.device.init), whose launch
; the thin link does not know. @helper is also called by @kernel, which has
; flat_wg=[256,256] and uniform work groups.
;
; RUN: opt -thinlto-bc %s -o %t.bc
; RUN: llvm-lto2 run %t.bc -o %t.o -save-temps -thinlink-closed-world \
; RUN:   -r=%t.bc,helper,p -r=%t.bc,kernel,px -r=%t.bc,g,px
; RUN: llvm-dis %t.o.1.5.precodegen.bc -o - | FileCheck %s

; CHECK: define {{.*}}void @helper() {{.*}}#[[HELPER:[0-9]+]]
; CHECK: attributes #[[HELPER]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="1,1024"
; CHECK-NOT: "uniform-work-group-size"="true"
; CHECK-SAME: }

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]

@g = addrspace(1) global i32 0

define internal void @ctor() {
  call void @helper()
  ret void
}

define void @helper() noinline {
  store volatile i32 1, ptr addrspace(1) @g
  ret void
}

define amdgpu_kernel void @kernel() #0 {
  call void @helper()
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size"="true" }
