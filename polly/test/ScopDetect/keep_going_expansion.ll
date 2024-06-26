; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-detect-track-failures -polly-detect-keep-going '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @a(i32 %n, ptr noalias %A, ptr noalias %B) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.cond2.preheader:                              ; preds = %for.body
  br label %for.body4

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %j.02 = trunc i64 %indvar to i32
  %arrayidx = getelementptr i32, ptr %B, i64 %indvar
  store i32 %j.02, ptr %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond3 = icmp ne i64 %indvar.next, 32
  br i1 %exitcond3, label %for.body, label %for.cond2.preheader

for.body4:                                        ; preds = %for.cond2.preheader, %for.body4
  %0 = phi i32 [ 0, %for.cond2.preheader ], [ %1, %for.body4 ]
  %mul = mul i32 %n, %0
  %idxprom5 = sext i32 %mul to i64
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i64 %idxprom5
  store i32 %0, ptr %arrayidx6, align 4
  %1 = add nsw i32 %0, 1
  %exitcond = icmp ne i32 %1, 32
  br i1 %exitcond, label %for.body4, label %for.end9

for.end9:                                         ; preds = %for.body4
  %idxprom10 = sext i32 %n to i64
  %arrayidx11 = getelementptr inbounds i32, ptr %A, i64 %idxprom10
  %2 = load i32, ptr %arrayidx11, align 4
  %idxprom12 = sext i32 %n to i64
  %arrayidx13 = getelementptr inbounds i32, ptr %B, i64 %idxprom12
  %3 = load i32, ptr %arrayidx13, align 4
  %add = add nsw i32 %3, %2
  ret i32 %add
}

; CHECK: Valid Region for Scop: for.body => for.body4
