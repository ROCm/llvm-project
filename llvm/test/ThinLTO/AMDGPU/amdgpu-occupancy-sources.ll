; Test the sources of the occupancy attributes:
; - @f_uniform_only has a launch attribute, but no occupancy attribute. It
;   gets the attributes of @k_256, without the default values.
; - @f_exported is visible to regular objects (resolution x). Native code can
;   call it from a kernel with any launch, so it gets the default values.
; - @f_spir gets the merge of @k_256 and of the SPIR kernel @k_spir.
; - The flat work-group size of @k_invalid is not valid. Like the backend, the
;   pass uses the default value for it.
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernels.ll -o kernels.bc
; RUN: opt -thinlto-bc device.ll -o device.bc
; RUN: llvm-lto2 run kernels.bc device.bc -o out -thinlto-distributed-indexes \
; RUN:   -thinlink-pipeline=amdgpu-thinlink-occupancy \
; RUN:   -print-after=amdgpu-thinlink-occupancy \
; RUN:   -r=kernels.bc,k_256,px -r=kernels.bc,k_spir,px \
; RUN:   -r=kernels.bc,k_invalid,px -r=kernels.bc,f_uniform_only, \
; RUN:   -r=kernels.bc,f_exported, -r=kernels.bc,f_spir, \
; RUN:   -r=kernels.bc,f_invalid, -r=device.bc,f_uniform_only,p \
; RUN:   -r=device.bc,f_exported,px -r=device.bc,f_spir,p \
; RUN:   -r=device.bc,f_invalid,p 2>&1 | FileCheck %s

; CHECK-DAG: ^[[F_UNIFORM_ONLY:[0-9]+]] = gv: (name: "f_uniform_only",
; CHECK-DAG: ^[[F_INVALID:[0-9]+]] = gv: (name: "f_invalid",
; CHECK-DAG: ^[[F_SPIR:[0-9]+]] = gv: (name: "f_spir",
; CHECK-DAG: ^[[F_EXPORTED:[0-9]+]] = gv: (name: "f_exported",
; CHECK: = extension: (name: "fn-attrs", version: 1, required: 1, entries: (
; CHECK-SAME: (gv: ^[[F_UNIFORM_ONLY]], module: ^0, ints: (0), strings: ("amdgpu-flat-work-group-size", "1,256", "amdgpu-max-num-workgroups", "4294967295,4294967295,4294967295", "amdgpu-waves-per-eu", "1")),
; CHECK-SAME: (gv: ^[[F_INVALID]], module: ^0, ints: (0), strings: ("amdgpu-flat-work-group-size", "1,1024", "amdgpu-max-num-workgroups", "4294967295,4294967295,4294967295", "amdgpu-waves-per-eu", "1")),
; CHECK-SAME: (gv: ^[[F_SPIR]], module: ^0, ints: (0), strings: ("amdgpu-flat-work-group-size", "1,512", "amdgpu-max-num-workgroups", "4294967295,4294967295,4294967295", "amdgpu-waves-per-eu", "1")),
; CHECK-SAME: (gv: ^[[F_EXPORTED]], module: ^0, ints: (0), strings: ("amdgpu-flat-work-group-size", "1,1024", "amdgpu-max-num-workgroups", "4294967295,4294967295,4294967295", "amdgpu-waves-per-eu", "1"))))

;--- kernels.ll
target triple = "amdgcn-amd-amdhsa"

declare void @f_uniform_only()
declare void @f_exported()
declare void @f_spir()
declare void @f_invalid()

define amdgpu_kernel void @k_256() #0 {
  call void @f_uniform_only()
  call void @f_exported()
  call void @f_spir()
  ret void
}

define spir_kernel void @k_spir() #1 {
  call void @f_spir()
  ret void
}

define amdgpu_kernel void @k_invalid() #2 {
  call void @f_invalid()
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,512" }
attributes #2 = { "amdgpu-flat-work-group-size"="512,256" }

;--- device.ll
target triple = "amdgcn-amd-amdhsa"

define void @f_uniform_only() #0 {
  ret void
}

define void @f_exported() {
  ret void
}

define void @f_spir() {
  ret void
}

define void @f_invalid() {
  ret void
}

attributes #0 = { "uniform-work-group-size" }
