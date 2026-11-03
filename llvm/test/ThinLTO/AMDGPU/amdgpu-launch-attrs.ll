; Test that the thin link propagates uniform-work-group-size and
; amdgpu-cluster-dims from the kernels to the device functions in another
; module, like the attributor:
; - @f_uniform: both kernels that call it have the same attributes.
; - @f_mixed: @k_plain has no attributes.
; - @f_cluster_var: the kernels use clusters with different dimensions.
; - @f_cluster_unknown: only one kernel uses clusters.
; - @f_exported: code outside the summaries can call it (resolution x).
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernels.ll -o kernels.bc
; RUN: opt -thinlto-bc device.ll -o device.bc
; RUN: llvm-lto2 run kernels.bc device.bc -o out -thinlto-distributed-indexes \
; RUN:   -thinlink-pipeline=amdgpu-thinlink-uniform-work-group-size,amdgpu-thinlink-cluster-dims \
; RUN:   -print-after=amdgpu-thinlink-cluster-dims \
; RUN:   -r=kernels.bc,k_uniform_a,px -r=kernels.bc,k_uniform_b,px \
; RUN:   -r=kernels.bc,k_plain,px -r=kernels.bc,k_cluster_4,px \
; RUN:   -r=kernels.bc,k_no_cluster,px -r=kernels.bc,f_uniform, \
; RUN:   -r=kernels.bc,f_mixed, -r=kernels.bc,f_cluster_var, \
; RUN:   -r=kernels.bc,f_cluster_unknown, -r=kernels.bc,f_exported, \
; RUN:   -r=device.bc,f_uniform,p -r=device.bc,f_mixed,p \
; RUN:   -r=device.bc,f_cluster_var,p -r=device.bc,f_cluster_unknown,p \
; RUN:   -r=device.bc,f_exported,px 2>&1 | FileCheck %s

; CHECK-DAG: ^[[F_CLUSTER_UNKNOWN:[0-9]+]] = gv: (name: "f_cluster_unknown",
; CHECK-DAG: ^[[F_CLUSTER_VAR:[0-9]+]] = gv: (name: "f_cluster_var",
; CHECK-DAG: ^[[F_UNIFORM:[0-9]+]] = gv: (name: "f_uniform",
; CHECK: = extension: (name: "fn-attrs", version: 1, required: 1, entries: (
; CHECK-SAME: (gv: ^[[F_CLUSTER_UNKNOWN]], module: ^0, ints: (0), strings: ("uniform-work-group-size", "")),
; CHECK-SAME: (gv: ^[[F_CLUSTER_VAR]], module: ^0, ints: (0), strings: ("amdgpu-cluster-dims", "1024,1024,1024")),
; CHECK-SAME: (gv: ^[[F_UNIFORM]], module: ^0, ints: (0), strings: ("amdgpu-cluster-dims", "2,2,1", "uniform-work-group-size", ""))))

; The backend sets the attributes right after the import, so the module that
; -save-temps saves after the import already has them.
; RUN: llvm-lto2 run kernels.bc device.bc -o out -save-temps \
; RUN:   -r=kernels.bc,k_uniform_a,px -r=kernels.bc,k_uniform_b,px \
; RUN:   -r=kernels.bc,k_plain,px -r=kernels.bc,k_cluster_4,px \
; RUN:   -r=kernels.bc,k_no_cluster,px -r=kernels.bc,f_uniform, \
; RUN:   -r=kernels.bc,f_mixed, -r=kernels.bc,f_cluster_var, \
; RUN:   -r=kernels.bc,f_cluster_unknown, -r=kernels.bc,f_exported, \
; RUN:   -r=device.bc,f_uniform,p -r=device.bc,f_mixed,p \
; RUN:   -r=device.bc,f_cluster_var,p -r=device.bc,f_cluster_unknown,p \
; RUN:   -r=device.bc,f_exported,px
; RUN: llvm-dis out.2.3.import.bc -o - | FileCheck %s --check-prefix=IR
; RUN: llvm-dis out.2.5.precodegen.bc -o - | FileCheck %s --check-prefix=IR

; IR: define void @f_uniform() {{.*}}#[[ATTR:[0-9]+]]
; IR: attributes #[[ATTR]] = {
; IR-SAME: "amdgpu-cluster-dims"="2,2,1"
; IR-SAME: "uniform-work-group-size"

;--- kernels.ll
target triple = "amdgcn-amd-amdhsa"

declare void @f_uniform()
declare void @f_mixed()
declare void @f_cluster_var()
declare void @f_cluster_unknown()
declare void @f_exported()

define amdgpu_kernel void @k_uniform_a() #0 {
  call void @f_uniform()
  call void @f_mixed()
  call void @f_cluster_var()
  call void @f_cluster_unknown()
  call void @f_exported()
  ret void
}

define amdgpu_kernel void @k_uniform_b() #0 {
  call void @f_uniform()
  ret void
}

define amdgpu_kernel void @k_plain() {
  call void @f_mixed()
  ret void
}

define amdgpu_kernel void @k_cluster_4() #1 {
  call void @f_cluster_var()
  ret void
}

define amdgpu_kernel void @k_no_cluster() #2 {
  call void @f_cluster_unknown()
  ret void
}

attributes #0 = { "uniform-work-group-size" "amdgpu-cluster-dims"="2,2,1" }
attributes #1 = { "amdgpu-cluster-dims"="4,4,1" }
attributes #2 = { "uniform-work-group-size" "amdgpu-cluster-dims"="0,0,0" }

;--- device.ll
target triple = "amdgcn-amd-amdhsa"

define void @f_uniform() {
  ret void
}

define void @f_mixed() {
  ret void
}

define void @f_cluster_var() {
  ret void
}

define void @f_cluster_unknown() {
  ret void
}

define void @f_exported() {
  ret void
}
