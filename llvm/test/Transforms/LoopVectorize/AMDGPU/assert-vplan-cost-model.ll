; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=loop-vectorize -amdgpu-coerce-illegal-types=1 < %s -S -o - | FileCheck %s

; REQUIRES: asserts

target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite)
define protected amdgpu_kernel void @func_int8(ptr addrspace(1) %p_a_grid.coerce, ptr addrspace(1) %p_b_grid.coerce, ptr addrspace(1) %p_c_grid.coerce, i32 %m, i32 %n, i32 %k, i1 %c, i32 %add, i32 %add12) {
; CHECK-LABEL: define protected amdgpu_kernel void @func_int8(
; CHECK-SAME: ptr addrspace(1) [[P_A_GRID_COERCE:%.*]], ptr addrspace(1) [[P_B_GRID_COERCE:%.*]], ptr addrspace(1) [[P_C_GRID_COERCE:%.*]], i32 [[M:%.*]], i32 [[N:%.*]], i32 [[K:%.*]], i1 [[C:%.*]], i32 [[ADD:%.*]], i32 [[ADD12:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    br i1 [[C]], label %[[FOR_COND_PREHEADER:.*]], label %[[IF_END:.*]]
; CHECK:       [[FOR_COND_PREHEADER]]:
; CHECK-NEXT:    [[CMP1444:%.*]] = icmp sgt i32 [[K]], 0
; CHECK-NEXT:    br i1 [[CMP1444]], label %[[FOR_BODY_LR_PH:.*]], label %[[FOR_COND_CLEANUP:.*]]
; CHECK:       [[FOR_BODY_LR_PH]]:
; CHECK-NEXT:    [[MUL15:%.*]] = mul nsw i32 [[ADD]], [[K]]
; CHECK-NEXT:    [[MUL17:%.*]] = mul nsw i32 [[ADD12]], [[K]]
; CHECK-NEXT:    br label %[[FOR_BODY:.*]]
; CHECK:       [[FOR_COND_CLEANUP_LOOPEXIT:.*]]:
; CHECK-NEXT:    [[ADD24_LCSSA:%.*]] = phi i32 [ [[ADD24:%.*]], %[[FOR_BODY]] ]
; CHECK-NEXT:    [[TMP15:%.*]] = trunc i32 [[ADD24_LCSSA]] to i8
; CHECK-NEXT:    br label %[[FOR_COND_CLEANUP]]
; CHECK:       [[FOR_COND_CLEANUP]]:
; CHECK-NEXT:    [[V_ACC_0_LCSSA:%.*]] = phi i8 [ 0, %[[FOR_COND_PREHEADER]] ], [ [[TMP15]], %[[FOR_COND_CLEANUP_LOOPEXIT]] ]
; CHECK-NEXT:    [[MUL25:%.*]] = mul nsw i32 [[ADD]], [[N]]
; CHECK-NEXT:    [[ADD26:%.*]] = add nsw i32 [[ADD12]], [[MUL25]]
; CHECK-NEXT:    [[IDXPROM27:%.*]] = sext i32 [[ADD26]] to i64
; CHECK-NEXT:    [[ARRAYIDX28:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P_C_GRID_COERCE]], i64 [[IDXPROM27]]
; CHECK-NEXT:    store i8 [[V_ACC_0_LCSSA]], ptr addrspace(1) [[ARRAYIDX28]], align 1
; CHECK-NEXT:    br label %[[IF_END]]
; CHECK:       [[FOR_BODY]]:
; CHECK-NEXT:    [[K_IDX_046:%.*]] = phi i32 [ 0, %[[FOR_BODY_LR_PH]] ], [ [[INC:%.*]], %[[FOR_BODY]] ]
; CHECK-NEXT:    [[V_ACC_045:%.*]] = phi i32 [ 0, %[[FOR_BODY_LR_PH]] ], [ [[ADD24]], %[[FOR_BODY]] ]
; CHECK-NEXT:    [[ADD16:%.*]] = add nsw i32 [[K_IDX_046]], [[MUL15]]
; CHECK-NEXT:    [[ADD18:%.*]] = add nsw i32 [[K_IDX_046]], [[MUL17]]
; CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 [[ADD16]] to i64
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P_A_GRID_COERCE]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[ARRAYIDX_VAL:%.*]] = load i8, ptr addrspace(1) [[ARRAYIDX]], align 1
; CHECK-NEXT:    [[IDXPROM19:%.*]] = sext i32 [[ADD18]] to i64
; CHECK-NEXT:    [[ARRAYIDX20:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P_B_GRID_COERCE]], i64 [[IDXPROM19]]
; CHECK-NEXT:    [[ARRAYIDX20_VAL:%.*]] = load i8, ptr addrspace(1) [[ARRAYIDX20]], align 1
; CHECK-NEXT:    [[CONV_I47:%.*]] = zext i8 [[ARRAYIDX_VAL]] to i32
; CHECK-NEXT:    [[CONV_I4248:%.*]] = zext i8 [[ARRAYIDX20_VAL]] to i32
; CHECK-NEXT:    [[MUL23:%.*]] = mul nuw nsw i32 [[CONV_I4248]], [[CONV_I47]]
; CHECK-NEXT:    [[ADD24]] = add i32 [[MUL23]], [[V_ACC_045]]
; CHECK-NEXT:    [[INC]] = add nuw nsw i32 [[K_IDX_046]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i32 [[INC]], [[K]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label %[[FOR_COND_CLEANUP_LOOPEXIT]], label %[[FOR_BODY]]
; CHECK:       [[IF_END]]:
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  %cmp1444 = icmp sgt i32 %k, 0
  br i1 %cmp1444, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %mul15 = mul nsw i32 %add, %k
  %mul17 = mul nsw i32 %add12, %k
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add24.lcssa = phi i32 [ %add24, %for.body ]
  %17 = trunc i32 %add24.lcssa to i8
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %for.cond.preheader
  %v_acc.0.lcssa = phi i8 [ 0, %for.cond.preheader ], [ %17, %for.cond.cleanup.loopexit ]
  %mul25 = mul nsw i32 %add, %n
  %add26 = add nsw i32 %add12, %mul25
  %idxprom27 = sext i32 %add26 to i64
  %arrayidx28 = getelementptr inbounds i8, ptr addrspace(1) %p_c_grid.coerce, i64 %idxprom27
  store i8 %v_acc.0.lcssa, ptr addrspace(1) %arrayidx28, align 1
  br label %if.end

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %k_idx.046 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %v_acc.045 = phi i32 [ 0, %for.body.lr.ph ], [ %add24, %for.body ]
  %add16 = add nsw i32 %k_idx.046, %mul15
  %add18 = add nsw i32 %k_idx.046, %mul17
  %idxprom = sext i32 %add16 to i64
  %arrayidx = getelementptr inbounds i8, ptr addrspace(1) %p_a_grid.coerce, i64 %idxprom
  %arrayidx.val = load i8, ptr addrspace(1) %arrayidx, align 1
  %idxprom19 = sext i32 %add18 to i64
  %arrayidx20 = getelementptr inbounds i8, ptr addrspace(1) %p_b_grid.coerce, i64 %idxprom19
  %arrayidx20.val = load i8, ptr addrspace(1) %arrayidx20, align 1
  %conv.i47 = zext i8 %arrayidx.val to i32
  %conv.i4248 = zext i8 %arrayidx20.val to i32
  %mul23 = mul nuw nsw i32 %conv.i4248, %conv.i47
  %add24 = add i32 %mul23, %v_acc.045
  %inc = add nuw nsw i32 %k_idx.046, 1
  %exitcond.not = icmp eq i32 %inc, %k
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body

if.end:                                           ; preds = %for.cond.cleanup, %entry
  ret void
}
