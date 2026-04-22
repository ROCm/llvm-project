; RUN: llc -amdgpu-late-wave-transform=1 -mtriple=amdgcn -mcpu=verde < %s | FileCheck %s
; RUN: llc -amdgpu-late-wave-transform=1 -mtriple=amdgcn -mcpu=tonga < %s | FileCheck %s

; CHECK-LABEL: {{^}}else_no_execfix:
; CHECK: s_cbranch_execz [[FLOW_BB:.LBB[0-9]+_[0-9]+]]
; CHECK: [[FLOW_BB]]:
; CHECK:	s_xor_b64 [[ACC:s\[[0-9]+:[0-9]+\]]], exec, [[ELSE_COND:s\[[0-9]+:[0-9]+\]]]
; CHECK:	s_and_b64 [[ACC]], [[ACC]], exec
; CHECK:	s_mov_b64 exec, [[ELSE_COND]]
define amdgpu_ps float @else_no_execfix(i32 %z, float %v) #0 {
main_body:
  %cc = icmp sgt i32 %z, 5
  br i1 %cc, label %if, label %else

if:
  %v.if = fmul float %v, 2.0
  br label %end

else:
  %v.else = fmul float %v, 3.0
  br label %end

end:
  %r = phi float [ %v.if, %if ], [ %v.else, %else ]
  ret float %r
}

; CHECK-LABEL: {{^}}else_execfix_leave_wqm:
; CHECK: ; %bb.0:
; CHECK-NEXT: s_mov_b64 [[INIT_EXEC:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: v_cmp_gt_i32_e64 [[ELSE_COND:s\[[0-9]+:[0-9]+\]]], 6, v0
; CHECK: s_xor_b64 [[TMP:s\[[0-9]+:[0-9]+\]]], [[ELSE_COND]], exec
; CHECK: s_xor_b64 [[ELSE:s\[[0-9]+:[0-9]+\]]], exec, [[TMP]]
; CHECK: s_cbranch_execz [[FLOW_BB:.LBB[0-9]+_[0-9]+]]
; CHECK: ; %if
; CHECK: s_and_b64 exec, exec, [[INIT_EXEC]]
; CHECK: [[FLOW_BB]]:
; CHECK-NEXT: s_or_b64 exec, exec, [[ELSE]]
; CHECK-NEXT: s_xor_b64 [[AND_INIT:s\[[0-9]+:[0-9]+\]]], exec, [[ELSE_COND]]
; CHECK-NEXT: s_and_b64 [[AND_INIT]], [[AND_INIT]], exec
; CHECK: s_mov_b64 exec, [[ELSE_COND]]
define amdgpu_ps void @else_execfix_leave_wqm(i32 %z, float %v) #0 {
main_body:
  %cc = icmp sgt i32 %z, 5
  br i1 %cc, label %if, label %else

if:
  %v.if = fmul float %v, 2.0
  br label %end

else:
  %c = fmul float %v, 3.0
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c, <8 x i32> poison, <4 x i32> poison, i1 0, i32 0, i32 0)
  %v.else = extractelement <4 x float> %tex, i32 0
  br label %end

end:
  %r = phi float [ %v.if, %if ], [ %v.else, %else ]
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %r, ptr addrspace(8) poison, i32 0, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind writeonly }
attributes #2 = { nounwind readonly }
