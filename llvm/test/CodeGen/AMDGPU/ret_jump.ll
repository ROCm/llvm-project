; RUN: llc -amdgpu-late-wave-transform=1 -mtriple=amdgcn -mcpu=tahiti -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -amdgpu-late-wave-transform=1 -mtriple=amdgcn -mcpu=tonga -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -check-prefix=GCN %s

; This should end with an no-op sequence of exec mask manipulations
; Mask should be in original state after executed unreachable block


; GCN-LABEL: {{^}}uniform_br_trivial_ret_divergent_br_trivial_unreachable:
; GCN: s_cbranch_scc1 [[RET_BB:.LBB[0-9]+_[0-9]+]]

; GCN: ; %else

; GCN: v_cmp_gt_f32_e64 [[SAVE:s\[[0-9]+:[0-9]+\]]]
; GCN: s_xor_b64 exec, [[SAVE]], exec
; GCN: s_cbranch_execz [[RET_BB]]

; GCN: .LBB{{[0-9]+}}_{{[0-9]+}}: ; %unreachable.bb
; GCN: ; divergent unreachable

; GCN: [[RET_BB]]: ; %UnifiedReturnBlock
; GCN: s_or_b64 exec, exec, [[SAVE]]

; GCN: ; return
; GCN: .Lfunc_end0
define amdgpu_ps <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @uniform_br_trivial_ret_divergent_br_trivial_unreachable(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg1, ptr addrspace(4) inreg %arg2, ptr addrspace(4) inreg %arg3, float inreg %arg4, i32 inreg %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <3 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, float %arg13, float %arg14, float %arg15, float %arg16, i32 inreg %arg17, i32 %arg18, i32 %arg19, float %arg20, i32 %arg21) #0 {
entry:
  %i.i = extractelement <2 x i32> %arg7, i32 0
  %j.i = extractelement <2 x i32> %arg7, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 1, i32 0, i32 %arg5) #2
  %p2 = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 1, i32 0, i32 %arg5) #2
  %p87 = fmul float %p2, %p2
  %p88 = fadd float %p87, %p87
  %p93 = fadd float %p88, %p88
  %p97 = fmul float %p93, %p93
  %p102 = fsub float %p97, %p97
  %p104 = fmul float %p102, %p102
  %p106 = fadd float 0.000000e+00, %p104
  %p108 = fadd float %p106, %p106
  %uniform.cond = icmp slt i32 %arg17, 0
  br i1 %uniform.cond, label %ret.bb, label %else

else:                                             ; preds = %main_body
  %p124 = fmul float %p108, %p108
  %p125 = fsub float %p124, %p124
  %divergent.cond = fcmp olt float %p125, 0.000000e+00
  br i1 %divergent.cond, label %ret.bb, label %unreachable.bb

unreachable.bb:                                           ; preds = %else
  unreachable

ret.bb:                                          ; preds = %else, %main_body
  ret <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> poison
}

; GCN-LABEL: {{^}}uniform_br_nontrivial_ret_divergent_br_nontrivial_unreachable:
; GCN: s_cbranch_scc{{(0|1)}} [[RET_BB2:.LBB[0-9]+_[0-9]+]]

; GCN: v_cmp_gt_f32_e32 vcc
; GCN: s_xor_b64 [[SAVE1:s\[[0-9]+:[0-9]+\]]], vcc, exec
; GCN: s_mov_b64 [[SAVE2:s\[[0-9]+:[0-9]+\]]], [[SAVE1]]
; GCN: s_mov_b64 exec, vcc
; GCN: s_cbranch_execz

; GCN: [[RET_BB2]]: ; %ret.bb
; GCN: v_mov_b32_e32 v0, 11
; GCN: {{buffer|flat}}_store_dword

; GCN: s_or_b64 exec, exec, [[SAVE2]]
; GCN: s_xor_b64 [[SAVE3:s\[[0-9]+:[0-9]+\]]], exec, [[SAVE1]]
; GCN: s_mov_b64 exec, [[SAVE1]]
; GCN: s_cbranch_execz [[UNIFIED_RET:.LBB[0-9]+_[0-9]+]]

; GCN: .LBB{{[0-9]+}}_{{[0-9]+}}: ; %unreachable.bb
; GCN: ds_write_b32
; GCN: ; divergent unreachable

; GCN: [[UNIFIED_RET]]: ; %UnifiedReturnBlock
; GCN: s_or_b64 exec, exec, [[SAVE3]]
; GCN: s_waitcnt
define amdgpu_ps <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @uniform_br_nontrivial_ret_divergent_br_nontrivial_unreachable(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg1, ptr addrspace(4) inreg %arg2, ptr addrspace(4) inreg %arg3, float inreg %arg4, i32 inreg %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <3 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, i32 inreg %arg18, i32 %arg19, float %arg20, i32 %arg21) #0 {
main_body:
  %i.i = extractelement <2 x i32> %arg7, i32 0
  %j.i = extractelement <2 x i32> %arg7, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 1, i32 0, i32 %arg5) #2
  %p2 = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 1, i32 0, i32 %arg5) #2
  %p87 = fmul float %p2, %p2
  %p88 = fadd float %p87, %p87
  %p93 = fadd float %p88, %p88
  %p97 = fmul float %p93, %p93
  %p102 = fsub float %p97, %p97
  %p104 = fmul float %p102, %p102
  %p106 = fadd float 0.000000e+00, %p104
  %p108 = fadd float %p106, %p106
  %uniform.cond = icmp slt i32 %arg18, 0
  br i1 %uniform.cond, label %ret.bb, label %else

else:                                             ; preds = %main_body
  %p124 = fmul float %p108, %p108
  %p125 = fsub float %p124, %p124
  %divergent.cond = fcmp olt float %p125, 0.000000e+00
  br i1 %divergent.cond, label %ret.bb, label %unreachable.bb

unreachable.bb:                                           ; preds = %else
  store volatile i32 8, ptr addrspace(3) poison
  unreachable

ret.bb:                                          ; preds = %else, %main_body
  store volatile i32 11, ptr addrspace(1) poison
  ret <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> poison
}

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.fabs.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.floor.f32(float) #1

attributes #0 = { "InitialPSInputAddr"="36983" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
