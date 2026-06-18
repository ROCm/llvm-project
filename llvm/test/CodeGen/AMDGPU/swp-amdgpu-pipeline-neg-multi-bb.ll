; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-enable-pipeliner -pass-remarks-analysis=pipeliner %s -o /dev/null 2>&1 | FileCheck %s
; Reject a loop whose body lowers to more than one basic block: the runtime-masked
; load is scalarized into cond.load blocks, so the pipeliner bails (single-block only).
; CHECK: Not a single basic block

define amdgpu_kernel void @swp_amdgpu_pipeline_neg_multi_bb(ptr addrspace(1) %p, i32 inreg %n) {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi <8 x half> [ zeroinitializer, %entry ], [ %v, %loop ]
  %m = icmp slt i32 %i, %n
  %mv = insertelement <8 x i1> poison, i1 %m, i64 0
  %mask = shufflevector <8 x i1> %mv, <8 x i1> poison, <8 x i32> zeroinitializer
  %gep = getelementptr half, ptr addrspace(1) %p, i32 %i
  %v = tail call <8 x half> @llvm.masked.load.v8f16.p1(ptr addrspace(1) %gep, i32 16, <8 x i1> %mask, <8 x half> %acc)
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:                                             ; preds = %loop
  store <8 x half> %v, ptr addrspace(1) %p
  ret void
}

declare <8 x half> @llvm.masked.load.v8f16.p1(ptr addrspace(1) nocapture, i32 immarg, <8 x i1>, <8 x half>)
