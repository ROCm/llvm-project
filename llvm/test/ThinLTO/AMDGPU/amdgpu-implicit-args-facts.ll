; Test the facts about the implicit arguments in the summary: the implicit
; arguments that the code of each function uses, without the functions that it
; calls. The attributor finds the same uses. A function without an entry can
; use all implicit arguments.
; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

; CHECK-DAG: ^[[ESCAPE:[0-9]+]] = gv: (name: "escape",
; CHECK-DAG: ^[[GROUP_SIZE:[0-9]+]] = gv: (name: "group_size",
; CHECK-DAG: ^[[NONE:[0-9]+]] = gv: (name: "none",
; CHECK-DAG: ^[[KERNEL_X:[0-9]+]] = gv: (name: "kernel_x",
; CHECK-DAG: ^[[HOSTCALL:[0-9]+]] = gv: (name: "hostcall",
; CHECK-DAG: ^[[SANITIZED:[0-9]+]] = gv: (name: "sanitized",
; CHECK-DAG: ^[[PRIVATE_CAST:[0-9]+]] = gv: (name: "private_cast",
; CHECK-DAG: ^[[CALLER:[0-9]+]] = gv: (name: "caller",
; CHECK-DAG: ^[[DEVICE_X:[0-9]+]] = gv: (name: "device_x",
; CHECK: = extension: (name: "amdgpu-implicit-args", version: 1, required: 0, entries: (
; CHECK-SAME: (gv: ^[[ESCAPE]], module: ^0, strings: ("queue-ptr", "implicitarg-ptr", "multigrid-sync-arg", "hostcall-ptr", "heap-ptr", "default-queue", "completion-action")),
; CHECK-SAME: (gv: ^[[GROUP_SIZE]], module: ^0, strings: ("implicitarg-ptr")),
; CHECK-SAME: (gv: ^[[NONE]], module: ^0),
; CHECK-SAME: (gv: ^[[KERNEL_X]], module: ^0),
; CHECK-SAME: (gv: ^[[HOSTCALL]], module: ^0, strings: ("implicitarg-ptr", "hostcall-ptr")),
; CHECK-SAME: (gv: ^[[SANITIZED]], module: ^0, strings: ("implicitarg-ptr", "hostcall-ptr", "flat-scratch-init")),
; CHECK-SAME: (gv: ^[[PRIVATE_CAST]], module: ^0, strings: ("implicitarg-ptr", "flat-scratch-init")),
; CHECK-SAME: (gv: ^[[CALLER]], module: ^0),
; CHECK-SAME: (gv: ^[[DEVICE_X]], module: ^0, strings: ("workitem-id-x"))))

target triple = "amdgcn-amd-amdhsa"

define void @none() {
  ret void
}

; A kernel always gets the x IDs, so it does not count them.
define void @device_x(ptr addrspace(1) %out) {
  %v = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @kernel_x(ptr addrspace(1) %out) {
  %v = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; The function reads the hostcall pointer through the implicit argument
; pointer.
define void @hostcall(ptr addrspace(1) %out) {
  %p = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %g = getelementptr i8, ptr addrspace(4) %p, i64 80
  %v = load i64, ptr addrspace(4) %g
  store i64 %v, ptr addrspace(1) %out
  ret void
}

; The work-group size is not one of the implicit arguments with an attribute.
define void @group_size(ptr addrspace(1) %out) {
  %p = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %g = getelementptr i8, ptr addrspace(4) %p, i64 12
  %v = load i16, ptr addrspace(4) %g
  store i16 %v, ptr addrspace(1) %out
  ret void
}

; The implicit argument pointer escapes, so the code can read any implicit
; argument.
define void @escape(ptr addrspace(1) %out) {
  %p = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  store ptr addrspace(4) %p, ptr addrspace(1) %out
  ret void
}

; Without aperture registers, the cast needs the queue pointer, which is in the
; implicit arguments in code object version 6.
define void @private_cast(ptr addrspace(1) %out) {
  %a = alloca i32, addrspace(5)
  %f = addrspacecast ptr addrspace(5) %a to ptr
  store ptr %f, ptr addrspace(1) %out
  ret void
}

define void @sanitized() sanitize_address {
  ret void
}

; The calls to other functions are in the call graph of the summary.
define void @caller(ptr addrspace(1) %out) {
  call void @device_x(ptr addrspace(1) %out)
  ret void
}

; The attributor does not track the implicit arguments of a graphics shader.
define amdgpu_ps void @shader() {
  ret void
}

; The attributor does not change an optnone function.
define void @no_opt() noinline optnone {
  ret void
}

; An intrinsic that can call back can reach any function.
define void @statepoint() gc "statepoint-example" {
  %t = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @none, i32 0, i32 0, i32 0, i32 0)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, ptr, i32 immarg, i32 immarg, ...)
