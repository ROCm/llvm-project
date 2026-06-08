; RUN: llc -amdgpu-late-wave-transform=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+wavefrontsize64 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -O0 -amdgpu-late-wave-transform=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+wavefrontsize64 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck -check-prefixes=OPT-O0 %s

; Test LowerBRCOND behaviour in the wave-transform flow (isWaveCFG == false).
; CHECK  pipeline:  default optimizations      -- negations appear as SETCC wrappers
; OPT-O0 pipeline: -O0 (no DAGCombine/IR opts) -- negations survive as XOR wrappers

declare i32 @llvm.amdgcn.workitem.id.x() #0
attributes #0 = { nounwind readnone }

define amdgpu_kernel void @setcc_negation_phi_i1(i32 %arg) {
; CHECK-LABEL: name: setcc_negation_phi_i1
; CHECK:       bb.2.bb3:
; CHECK:         [[CMP1:%[0-9]+]]:sreg_64_xexec = V_CMP_NE_U32_e64 [[PHI:%[0-9]+]], 0, implicit $exec
; CHECK-NEXT:    [[CNDMASK:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, [[CMP1]], implicit $exec
; CHECK-NEXT:    [[ONE:%[0-9]+]]:sreg_32 = S_MOV_B32 1
; CHECK-NEXT:    [[CMP2:%[0-9]+]]:sreg_64 = V_CMP_NE_U32_e64 killed [[CNDMASK]], killed [[ONE]], implicit $exec
; CHECK-NEXT:    SI_BRCOND %bb.4, killed [[CMP2]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; CHECK-NEXT:    S_BRANCH %bb.3
;
; OPT-O0 -- same function, but negation appears as S_XOR_B64 instead of SETCC.
; OPT-O0-LABEL: name: setcc_negation_phi_i1
; OPT-O0:       bb.2.bb3:
; OPT-O0:         [[CMP:%[0-9]+]]:sreg_64 = V_CMP_NE_U32_e64 [[PHI:%[0-9]+]], 0, implicit $exec
; OPT-O0-NEXT:    [[NEG1:%[0-9]+]]:sreg_64 = S_MOV_B64 -1
; OPT-O0-NEXT:    [[XOR:%[0-9]+]]:sreg_64 = S_XOR_B64 [[CMP]], killed [[NEG1:%[0-9]+]], implicit-def dead $scc
; OPT-O0-NEXT:    SI_BRCOND %bb.4, killed [[XOR]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; OPT-O0-NEXT:    S_BRANCH %bb.3
bb:
  %tidig = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = trunc i32 %tidig to i1
  br i1 %cmp, label %bb2, label %bb3

bb2:
  br label %bb3

bb3:
  %tmp = phi i1 [ true, %bb2 ], [ false, %bb ]
  br i1 %tmp, label %bb4, label %bb6

bb4:
  %val = load volatile i32, ptr addrspace(1) poison
  br label %bb6

bb6:
  ret void
}

define amdgpu_kernel void @xor_negation_icmp(i32 %N, ptr addrspace(1) %p) {
; CHECK-LABEL: name: xor_negation_icmp
; CHECK:       S_ENDPGM 0
;
; OPT-O0 -- the XOR negation survives.
; OPT-O0-LABEL: name: xor_negation_icmp
; OPT-O0:       bb.0.entry:
; OPT-O0:         [[CMP:%[0-9]+]]:sreg_64 = V_CMP_LT_I32_e64 killed [[LHS:%[0-9]+]], killed [[RHS:%[0-9]+]], implicit $exec
; OPT-O0-NEXT:    [[NEG1:%[0-9]+]]:sreg_64 = S_MOV_B64 -1
; OPT-O0-NEXT:    [[XOR:%[0-9]+]]:sreg_64 = S_XOR_B64 killed [[CMP]], killed [[NEG1]], implicit-def dead $scc
; OPT-O0-NEXT:    SI_BRCOND %bb.2, killed [[XOR]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; OPT-O0-NEXT:    S_BRANCH %bb.1
entry:
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp slt i32 %id.x, 1
  br i1 %cmp, label %if.then, label %exit

if.then:
  %idx.ext = zext i32 %N to i64
  %add.ptr = getelementptr i8, ptr addrspace(1) %p, i64 %idx.ext
  ret void

exit:
  ret void
}

define amdgpu_kernel void @plain_divergent_branch(ptr addrspace(1) %out, i32 %arg) {
; CHECK-LABEL: name: plain_divergent_branch
; CHECK:       bb.0.entry:
; CHECK:         [[CMP:%[0-9]+]]:sreg_64 = V_CMP_LE_U32_e64 killed [[LHS:%[0-9]+]], killed [[RHS:%[0-9]+]], implicit $exec
; CHECK:         SI_BRCOND %bb.2, killed [[CMP]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; CHECK-NEXT:    S_BRANCH %bb.1
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp ugt i32 %id, %arg
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 1, ptr addrspace(1) %out
  br label %if.end

if.end:
  ret void
}

define amdgpu_kernel void @explicit_i1_not(ptr addrspace(1) %out) {
; CHECK-LABEL: name: explicit_i1_not
; CHECK:       bb.0.entry:
; CHECK:         [[CMP:%[0-9]+]]:sreg_64 = V_CMP_EQ_U32_e64 killed [[LHS:%[0-9]+]], killed [[RHS:%[0-9]+]], implicit $exec
; CHECK:         SI_BRCOND %bb.2, killed [[CMP]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; CHECK-NEXT:    S_BRANCH %bb.1
;
; OPT-O0 -- double S_XOR_B64 (double negation).
; OPT-O0-LABEL: name: explicit_i1_not
; OPT-O0:       bb.0.entry:
; OPT-O0:         [[CMP:%[0-9]+]]:sreg_64 = V_CMP_EQ_U32_e64 killed [[LHS:%[0-9]+]], killed [[RHS:%[0-9]+]], implicit $exec
; OPT-O0-NEXT:    [[NEG1:%[0-9]+]]:sreg_64 = S_MOV_B64 -1
; OPT-O0-NEXT:    [[XOR1:%[0-9]+]]:sreg_64 = S_XOR_B64 killed [[CMP]], [[NEG1:%[0-9]+]], implicit-def dead $scc
; OPT-O0-NEXT:    [[XOR2:%[0-9]+]]:sreg_64 = S_XOR_B64 killed [[XOR1]], [[NEG1]], implicit-def dead $scc
; OPT-O0-NEXT:    SI_BRCOND %bb.2, killed [[XOR2]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; OPT-O0-NEXT:    S_BRANCH %bb.1
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp eq i32 %id, 0
  %neg = xor i1 %cmp, true
  br i1 %neg, label %if.then, label %if.end

if.then:
  store i32 42, ptr addrspace(1) %out
  br label %if.end

if.end:
  ret void
}

define amdgpu_kernel void @uniform_branch(ptr addrspace(1) %out, i32 inreg %flag) {
; CHECK-LABEL: name: uniform_branch
; CHECK:       bb.0.entry:
; CHECK:         S_CMP_LG_U32 killed [[LHS:%[0-9]+]], killed [[RHS:%[0-9]+]], implicit-def $scc
; CHECK:         S_CBRANCH_SCC1 %bb.2, implicit $scc
; CHECK-NEXT:    S_BRANCH %bb.1
;
; OPT-O0 -- still scalar, no SI_BRCOND.
; OPT-O0-LABEL: name: uniform_branch
; OPT-O0:       bb.0.entry:
; OPT-O0:         S_CMP_EQ_U32 {{%[0-9]+}}, killed [[ZERO:%[0-9]+]], implicit-def $scc
; OPT-O0:         S_CBRANCH_VCCNZ %bb.2, implicit $vcc
; OPT-O0-NEXT:    S_BRANCH %bb.1
entry:
  %cmp = icmp eq i32 %flag, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 1, ptr addrspace(1) %out
  br label %if.end

if.end:
  ret void
}

define amdgpu_kernel void @nested_phi_i1_diamond(ptr addrspace(1) %out) {
; CHECK-LABEL: name: nested_phi_i1_diamond
;
; CHECK:       bb.2.merge1:
; CHECK:         [[CMP1:%[0-9]+]]:sreg_64_xexec = V_CMP_NE_U32_e64 [[PHI1:%[0-9]+]], 0, implicit $exec
; CHECK-NEXT:    [[CNDMASK1:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, [[CMP1]], implicit $exec
; CHECK-NEXT:    [[ONE1:%[0-9]+]]:sreg_32 = S_MOV_B32 1
; CHECK-NEXT:    [[CMP2:%[0-9]+]]:sreg_64 = V_CMP_NE_U32_e64 killed [[CNDMASK1]], killed [[ONE1]], implicit $exec
; CHECK-NEXT:    SI_BRCOND %bb.4, killed [[CMP2]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; CHECK-NEXT:    S_BRANCH %bb.3
;
; CHECK:       bb.7.merge3:
; CHECK:         [[CMP3:%[0-9]+]]:sreg_64_xexec = V_CMP_NE_U32_e64 [[PHI2:%[0-9]+]], 0, implicit $exec
; CHECK-NEXT:    [[CNDMASK2:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, [[CMP3]], implicit $exec
; CHECK-NEXT:    [[ONE2:%[0-9]+]]:sreg_32 = S_MOV_B32 1
; CHECK-NEXT:    [[CMP4:%[0-9]+]]:sreg_64 = V_CMP_NE_U32_e64 killed [[CNDMASK2]], killed [[ONE2]], implicit $exec
; CHECK-NEXT:    SI_BRCOND %bb.9, killed [[CMP4]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; CHECK-NEXT:    S_BRANCH %bb.8
;
; OPT-O0 -- both merges show S_XOR_B64 instead.
; OPT-O0-LABEL: name: nested_phi_i1_diamond
; OPT-O0:       bb.2.merge1:
; OPT-O0:         [[CMP1:%[0-9]+]]:sreg_64 = V_CMP_NE_U32_e64 [[PHI1:%[0-9]+]], 0, implicit $exec
; OPT-O0-NEXT:    [[NEG1:%[0-9]+]]:sreg_64 = S_MOV_B64 -1
; OPT-O0-NEXT:    [[XOR1:%[0-9]+]]:sreg_64 = S_XOR_B64 [[CMP1]], killed [[NEG1:%[0-9]+]], implicit-def dead $scc
; OPT-O0-NEXT:    SI_BRCOND %bb.4, killed [[XOR1]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; OPT-O0-NEXT:    S_BRANCH %bb.3
;
; OPT-O0:       bb.7.merge3:
; OPT-O0:         [[CMP2:%[0-9]+]]:sreg_64 = V_CMP_NE_U32_e64 [[PHI2:%[0-9]+]], 0, implicit $exec
; OPT-O0-NEXT:    [[NEG2:%[0-9]+]]:sreg_64 = S_MOV_B64 -1
; OPT-O0-NEXT:    [[XOR2:%[0-9]+]]:sreg_64 = S_XOR_B64 [[CMP2]], killed [[NEG2:%[0-9]+]], implicit-def dead $scc
; OPT-O0-NEXT:    SI_BRCOND %bb.9, killed [[XOR2]], implicit-def dead $exec, implicit-def dead $vcc, implicit $exec
; OPT-O0-NEXT:    S_BRANCH %bb.8
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %c1 = trunc i32 %id to i1
  br i1 %c1, label %left, label %merge1

left:
  br label %merge1

merge1:
  %p1 = phi i1 [ true, %left ], [ false, %entry ]
  br i1 %p1, label %inner.left, label %inner.right

inner.left:
  store i32 1, ptr addrspace(1) %out
  br label %merge2

inner.right:
  store i32 2, ptr addrspace(1) %out
  br label %merge2

merge2:
  %id2 = add i32 %id, 1
  %c2 = trunc i32 %id2 to i1
  br i1 %c2, label %left2, label %merge3

left2:
  br label %merge3

merge3:
  %p2 = phi i1 [ true, %left2 ], [ false, %merge2 ]
  br i1 %p2, label %final.then, label %final.end

final.then:
  store i32 3, ptr addrspace(1) %out
  br label %final.end

final.end:
  ret void
}
