; XFAIL: *
; The thin link does not compute the facts about implicit arguments for now
; (see addImplicitArgFacts() in AMDGPUThinLinkPasses.cpp).

; Test that the thin link gives a kernel the amdgpu-no-* attributes of the
; implicit arguments that it and the functions that it can call do not use.
; @kernel_y calls @device_y in another module, which uses only workitem-id-y.
; @kernel_ext calls @ext, which has no summary and can use all implicit
; arguments.
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernel.ll -o kernel.bc
; RUN: opt -thinlto-bc device.ll -o device.bc

; The index file of kernel.bc holds the attributes of @kernel_y.
; RUN: llvm-lto2 run kernel.bc device.bc -o out -thinlto-distributed-indexes \
; RUN:   -thinlink-pipeline=amdgpu-thinlink-implicit-args \
; RUN:   -r=kernel.bc,kernel_y,px -r=kernel.bc,kernel_ext,px \
; RUN:   -r=kernel.bc,device_y, -r=kernel.bc,ext, -r=device.bc,device_y,px
; RUN: llvm-dis kernel.bc.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX

; INDEX:      ^[[DEVICE_MOD:[0-9]+]] = module: (path: "device.bc",
; INDEX-NEXT: ^[[KERNEL_MOD:[0-9]+]] = module: (path: "kernel.bc",
; INDEX:      ^[[KERNEL_Y:[0-9]+]] = gv: (guid: {{[0-9]+}}, summaries: (function: (module: ^[[KERNEL_MOD]], {{.*}}calls: ((callee: ^[[DEVICE_Y:[0-9]+]]
; INDEX:      = extension: (name: "fn-attrs", version: 1, required: 1, entries: (
; INDEX-SAME: (gv: ^[[KERNEL_Y]], module: ^[[KERNEL_MOD]], ints: (0), strings: ("amdgpu-no-cluster-id-x", "", "amdgpu-no-cluster-id-y", "", "amdgpu-no-cluster-id-z", "", "amdgpu-no-completion-action", "", "amdgpu-no-default-queue", "", "amdgpu-no-dispatch-id", "", "amdgpu-no-dispatch-ptr", "", "amdgpu-no-flat-scratch-init", "", "amdgpu-no-heap-ptr", "", "amdgpu-no-hostcall-ptr", "", "amdgpu-no-implicitarg-ptr", "", "amdgpu-no-lds-kernel-id", "", "amdgpu-no-multigrid-sync-arg", "", "amdgpu-no-queue-ptr", "", "amdgpu-no-workgroup-id-x", "", "amdgpu-no-workgroup-id-y", "", "amdgpu-no-workgroup-id-z", "", "amdgpu-no-workitem-id-x", "", "amdgpu-no-workitem-id-z", "", "amdgpu-no-wwm", "")),
; INDEX-SAME: (gv: ^[[DEVICE_Y]], module: ^[[DEVICE_MOD]], ints: (0), strings: ({{.*}}"amdgpu-no-workitem-id-x", "", "amdgpu-no-workitem-id-z", "", "amdgpu-no-wwm", ""))))

; With no import, the backend of @kernel_y does not have the code of
; @device_y. The kernel gets the attributes from the thin link.
; RUN: llvm-lto2 run kernel.bc device.bc -o out -save-temps \
; RUN:   -import-instr-limit=0 \
; RUN:   -r=kernel.bc,kernel_y,px -r=kernel.bc,kernel_ext,px \
; RUN:   -r=kernel.bc,device_y, -r=kernel.bc,ext, -r=device.bc,device_y,px
; RUN: llvm-dis out.1.5.precodegen.bc -o - | FileCheck %s

; CHECK: define amdgpu_kernel void @kernel_y({{.*}}) {{.*}}#[[ATTR:[0-9]+]]
; CHECK: define amdgpu_kernel void @kernel_ext() {{[^#]*}}{
; CHECK: attributes #[[ATTR]] = { "amdgpu-no-cluster-id-x" {{.*}} "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }

; Without the pass, the kernel gets no attributes.
; RUN: llvm-lto2 run kernel.bc device.bc -o nopass -save-temps \
; RUN:   -import-instr-limit=0 -thinlink-pipeline=amdgpu-thinlink-occupancy \
; RUN:   -r=kernel.bc,kernel_y,px -r=kernel.bc,kernel_ext,px \
; RUN:   -r=kernel.bc,device_y, -r=kernel.bc,ext, -r=device.bc,device_y,px
; RUN: llvm-dis nopass.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=NOPASS

; NOPASS: define amdgpu_kernel void @kernel_y({{.*}}) {{[^#]*}}{
; NOPASS-NOT: amdgpu-no-

;--- kernel.ll
target triple = "amdgcn-amd-amdhsa"

declare void @device_y(ptr addrspace(1))
declare void @ext()

define amdgpu_kernel void @kernel_y(ptr addrspace(1) %out) {
  call void @device_y(ptr addrspace(1) %out)
  ret void
}

define amdgpu_kernel void @kernel_ext() {
  call void @ext()
  ret void
}

;--- device.ll
target triple = "amdgcn-amd-amdhsa"

define void @device_y(ptr addrspace(1) %out) {
  %v = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %v, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.y()
