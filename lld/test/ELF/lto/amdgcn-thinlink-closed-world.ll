; REQUIRES: amdgpu
;; Test that the ThinLTO thin link assumes a closed world only if the link has
;; no native code. The exported @table takes the address of @target. In a
;; closed world, only @kernel calls @target, so @target gets the occupancy
;; attributes of @kernel. A native object file in the link turns off the closed
;; world, so @target gets the default attributes.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc kernel.ll -o kernel.o
; RUN: opt -thinlto-bc device.ll -o device.o
; RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj native.s \
; RUN:   -o native.o

; RUN: ld.lld -shared kernel.o device.o -o closed.so -plugin-opt=mcpu=gfx900 \
; RUN:   -mllvm -thinlink-closed-world -save-temps
; RUN: llvm-dis device.o.5.precodegen.bc -o - | FileCheck %s --check-prefix=CLOSED

; RUN: ld.lld -shared kernel.o device.o native.o -o open.so \
; RUN:   -plugin-opt=mcpu=gfx900 -mllvm -thinlink-closed-world -save-temps
; RUN: llvm-dis device.o.5.precodegen.bc -o - | FileCheck %s --check-prefix=OPEN

; CLOSED: define hidden void @target() {{.*}}#[[ATTR:[0-9]+]]
; CLOSED: attributes #[[ATTR]] = { {{.*}}"amdgpu-flat-work-group-size"="64,256"

; OPEN: define hidden void @target() {{.*}}#[[ATTR:[0-9]+]]
; OPEN: attributes #[[ATTR]] = { {{.*}}"amdgpu-flat-work-group-size"="1,1024"

;--- kernel.ll
target triple = "amdgcn-amd-amdhsa"

declare hidden void @target()

define amdgpu_kernel void @kernel() #0 {
  call void @target()
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" }

;--- device.ll
target triple = "amdgcn-amd-amdhsa"

@table = addrspace(1) global ptr @target

define hidden void @target() {
  ret void
}

;--- native.s
  .text
