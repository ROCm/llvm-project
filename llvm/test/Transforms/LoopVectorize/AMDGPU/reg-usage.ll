; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
; RUN:   -force-vector-width=8 -debug-only=loop-vectorize --disable-output \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V8
; RUN: opt -passes=loop-vectorize -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
; RUN:   -force-vector-width=4 -debug-only=loop-vectorize --disable-output \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V4
; RUN: opt -passes=loop-vectorize -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
; RUN:   -force-vector-width=8 -debug-only=loop-vectorize --disable-output \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-I32

target triple = "amdgcn-amd-amdhsa"

define void @i8_add(ptr addrspace(1) nocapture noalias readonly %a,
                      ptr addrspace(1) nocapture noalias readonly %b,
                      ptr addrspace(1) nocapture noalias writeonly %out,
                      i32 noundef %n) {
; CHECK-V8-LABEL: i8_add
; CHECK-V8:      LV(REG): VF = 8
; CHECK-V8-NEXT: LV(REG): Found max usage: 2 item
; CHECK-V8-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 4 registers
; CHECK-V8-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 4 registers
; CHECK-V4-LABEL: i8_add
; CHECK-V4:      LV(REG): VF = 4
; CHECK-V4-NEXT: LV(REG): Found max usage: 2 item
; CHECK-V4-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 4 registers
; CHECK-V4-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 2 registers
; CHECK-I32-NOT: RegisterClass: Generic::VectorRC, 16 registers
entry:
  %cmp.not = icmp eq i32 %n, 0
  br i1 %cmp.not, label %exit, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.body ]
  %gep.a = getelementptr inbounds i8, ptr addrspace(1) %a, i32 %i
  %gep.b = getelementptr inbounds i8, ptr addrspace(1) %b, i32 %i
  %gep.out = getelementptr inbounds i8, ptr addrspace(1) %out, i32 %i
  %va = load i8, ptr addrspace(1) %gep.a, align 1
  %vb = load i8, ptr addrspace(1) %gep.b, align 1
  %sum = add i8 %va, %vb
  store i8 %sum, ptr addrspace(1) %gep.out, align 1
  %i.next = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %i.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

define void @i32_add(ptr addrspace(1) nocapture noalias readonly %a,
                       ptr addrspace(1) nocapture noalias readonly %b,
                       ptr addrspace(1) nocapture noalias writeonly %out,
                       i32 noundef %n) {
; CHECK-I32-LABEL: i32_add
; CHECK-I32:      LV(REG): VF = 8
; CHECK-I32-NEXT: LV(REG): Found max usage: 2 item
; CHECK-I32-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 4 registers
; CHECK-I32-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 2 registers
entry:
  %cmp.not = icmp eq i32 %n, 0
  br i1 %cmp.not, label %exit, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  %gep.b = getelementptr inbounds i32, ptr addrspace(1) %b, i32 %i
  %gep.out = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i
  %va = load i32, ptr addrspace(1) %gep.a, align 4
  %vb = load i32, ptr addrspace(1) %gep.b, align 4
  %sum = add i32 %va, %vb
  store i32 %sum, ptr addrspace(1) %gep.out, align 4
  %i.next = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %i.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}
