; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-sqtt-function-instrument -S < %s > %t.ll
; RUN: llc -global-isel=0 -mtriple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 < %t.ll | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 < %t.ll | FileCheck %s
; RUN: llc -global-isel=0 -mtriple=amdgpu10.10-amd-amdhsa -mcpu=gfx1010 < %t.ll | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu10.10-amd-amdhsa -mcpu=gfx1010 < %t.ll | FileCheck %s

; The events go through the same pipeline (including the return unification),
; but they are lowered to a comment: no s_ttracedata, no m0 setup, no
; sched_barrier and no event table.

; CHECK-LABEL: divergent_returns:
; CHECK-NOT: s_ttracedata
; CHECK-NOT: m0
; CHECK-NOT: sched_barrier
; CHECK: ; sqtt_event 0
; CHECK: .LBB0_{{[0-9]+}}: ; %UnifiedReturnBlock
; CHECK-NOT: s_ttracedata
; CHECK-NOT: m0
; CHECK-NOT: sched_barrier
; CHECK: ; sqtt_event 1
; CHECK: s_setpc_b64
; CHECK-NOT: .section __sqtt
define void @divergent_returns(ptr addrspace(1) %p) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %c = icmp ult i32 %id, 32
  br i1 %c, label %a, label %b

a:
  store i32 1, ptr addrspace(1) %p
  ret void

b:
  store i32 2, ptr addrspace(1) %p
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { "amdgpu-sqtt-instrument" }
