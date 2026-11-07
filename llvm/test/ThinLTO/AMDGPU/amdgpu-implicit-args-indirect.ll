; XFAIL: *
; The thin link does not compute the facts about implicit arguments for now
; (see addImplicitArgFacts() in AMDGPUThinLinkPasses.cpp).

; Test that the thin link follows indirect calls for the amdgpu-no-* attributes.
; @kernel calls a function pointer of the type void(ptr addrspace(1)). @table
; takes the address of @device_y, which uses only workitem-id-y, and of
; @device_none, which uses no implicit argument.
;
; In a closed world, the indirect call reaches only these functions, so
; @kernel gets the attributes of all implicit arguments other than
; workitem-id-y. In an open world, the indirect call can reach code outside the
; summaries, so @kernel gets no attributes. The backend of @kernel does not see
; the targets, so it adds no attributes.
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernel.ll -o kernel.bc
; RUN: opt -thinlto-bc device.ll -o device.bc
; RUN: llvm-lto2 run kernel.bc device.bc -o closed -save-temps \
; RUN:   -thinlink-closed-world -r=kernel.bc,kernel,px \
; RUN:   -r=device.bc,device_y,p -r=device.bc,device_none,p \
; RUN:   -r=device.bc,table,px
; RUN: llvm-dis closed.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=CLOSED
; RUN: llvm-lto2 run kernel.bc device.bc -o open -save-temps \
; RUN:   -r=kernel.bc,kernel,px \
; RUN:   -r=device.bc,device_y,p -r=device.bc,device_none,p \
; RUN:   -r=device.bc,table,px
; RUN: llvm-dis open.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=OPEN

; CLOSED: define amdgpu_kernel void @kernel({{.*}}) {{.*}}#[[ATTR:[0-9]+]]
; CLOSED: attributes #[[ATTR]] = { "amdgpu-no-cluster-id-x" {{.*}} "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }

; OPEN: define amdgpu_kernel void @kernel({{.*}}) {{[^#]*}}{
; OPEN-NOT: amdgpu-no-

; A function in module assembly has no call type IDs, so it can have any call
; type ID. In a closed world, the indirect call can also reach @asm_fn, which
; can use all implicit arguments. So @kernel gets no attributes.
; RUN: opt -thinlto-bc asm.ll -o asm.bc
; RUN: llvm-lto2 run kernel.bc device.bc asm.bc -o asm -save-temps \
; RUN:   -thinlink-closed-world -r=kernel.bc,kernel,px \
; RUN:   -r=device.bc,device_y,p -r=device.bc,device_none,p \
; RUN:   -r=device.bc,table,px -r=asm.bc,asm_fn,p -r=asm.bc,asm_table,px
; RUN: llvm-dis asm.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=ASM

; ASM: define amdgpu_kernel void @kernel({{.*}}) {{[^#]*}}{
; ASM-NOT: amdgpu-no-

;--- kernel.ll
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel(ptr %fp, ptr addrspace(1) %out) {
  call void %fp(ptr addrspace(1) %out)
  ret void
}

;--- device.ll
target triple = "amdgcn-amd-amdhsa"

@table = addrspace(1) global [2 x ptr] [ptr @device_y, ptr @device_none]

define void @device_y(ptr addrspace(1) %out) {
  %y = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %y, ptr addrspace(1) %out
  ret void
}

define void @device_none(ptr addrspace(1) %out) {
  store i32 0, ptr addrspace(1) %out
  ret void
}

;--- asm.ll
target triple = "amdgcn-amd-amdhsa"

module asm "asm_fn:"
module asm "  s_setpc_b64 s[30:31]"

declare void @asm_fn(ptr addrspace(1))

@asm_table = addrspace(1) global [1 x ptr] [ptr @asm_fn]
