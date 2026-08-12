; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; A cycle with a divergent exit that is left through more than one edge into the
; same exit block. Threads leave on different iterations and along different
; edges, so a phi in that exit block is divergent even though both of its
; incoming values are constants defined outside the cycle.

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #0

; Two edges leave the cycle {header, body} into %exit: the uniform branch in
; %header and the divergent branch in %body. %acc.same.block shares the exit
; block and the same two edges, but has a single reaching value, so the per-phi
; escape still applies once the edge count has been established for the block.

; CHECK-LABEL: UniformityInfo for function 'two_exit_edges':
; CHECK: DIVERGENT:   %acc.two.edges = phi i32 [ 1, %body ], [ 0, %header ]
; CHECK-NOT: DIVERGENT:   %acc.same.block
define amdgpu_kernel void @two_exit_edges(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %body, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %body ]
  %uni.cond = icmp slt i32 %ctr, 1
  br i1 %uni.cond, label %body, label %exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %body, %header
  %acc.two.edges = phi i32 [ 1, %body ], [ 0, %header ]
  %acc.same.block = phi i32 [ 5, %body ], [ 5, %header ]
  store i32 %acc.two.edges, ptr addrspace(1) %out
  store i32 %acc.same.block, ptr addrspace(1) %out
  ret void
}

; The same two-edge inner cycle, nested in an outer cycle. %inner.exit is inside
; the outer cycle, so the property has to hold per outer iteration.

; CHECK-LABEL: UniformityInfo for function 'nested_two_exit_edges':
; CHECK: DIVERGENT:   %acc.nested = phi i32 [ 1, %inner.body ], [ 0, %inner.header ]
define amdgpu_kernel void @nested_two_exit_edges(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %outer.header

outer.header:                                     ; preds = %outer.latch, %entry
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:                                     ; preds = %inner.body, %outer.header
  %ctr = phi i32 [ 0, %outer.header ], [ 1, %inner.body ]
  %uni.cond = icmp slt i32 %ctr, 1
  br i1 %uni.cond, label %inner.body, label %inner.exit

inner.body:                                       ; preds = %inner.header
  %div.cond = icmp eq i32 %tid, %i
  br i1 %div.cond, label %inner.exit, label %inner.header

inner.exit:                                       ; preds = %inner.body, %inner.header
  %acc.nested = phi i32 [ 1, %inner.body ], [ 0, %inner.header ]
  store i32 %acc.nested, ptr addrspace(1) %out
  br label %outer.latch

outer.latch:                                      ; preds = %inner.exit
  %i.next = add i32 %i, 1
  %outer.cond = icmp slt i32 %i.next, %n
  br i1 %outer.cond, label %outer.header, label %ret.bb

ret.bb:                                           ; preds = %outer.latch
  ret void
}

; Only one edge leaves the cycle {header} into %exit; the second predecessor of
; %exit is outside the cycle. Every thread that leaves the cycle does so along
; the same edge, so the phi stays uniform.

; CHECK-LABEL: UniformityInfo for function 'one_exit_edge':
; CHECK-NOT: DIVERGENT:   %acc.one.edge
define amdgpu_kernel void @one_exit_edge(ptr addrspace(1) %out, i1 %uni) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp eq i32 %tid, 0
  br i1 %uni, label %header, label %exit

header:                                           ; preds = %header, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %header ]
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %header, %entry
  %acc.one.edge = phi i32 [ 1, %header ], [ 0, %entry ]
  store i32 %acc.one.edge, ptr addrspace(1) %out
  ret void
}

; Two exit edges as in @two_exit_edges, but the phi selects the same value along
; both of them, so it remains uniform.

; CHECK-LABEL: UniformityInfo for function 'two_exit_edges_same_value':
; CHECK-NOT: DIVERGENT:   %acc.same
define amdgpu_kernel void @two_exit_edges_same_value(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %body, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %body ]
  %uni.cond = icmp slt i32 %ctr, 1
  br i1 %uni.cond, label %body, label %exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %body, %header
  %acc.same = phi i32 [ 7, %body ], [ 7, %header ]
  store i32 %acc.same, ptr addrspace(1) %out
  ret void
}

; As @two_exit_edges_same_value, but %exit has a third predecessor from outside
; the cycle carrying a different value. Only the values on the two exit edges
; decide whether leaving the cycle is observable, and those agree, so the phi
; stays uniform. Whether reaching %exit from %entry rather than from the cycle
; is divergent is a join question, answered by taintAndPushPhiNodes.

; CHECK-LABEL: UniformityInfo for function 'same_value_on_exit_edges':
; CHECK-NOT: DIVERGENT:   %acc.outside
define amdgpu_kernel void @same_value_on_exit_edges(ptr addrspace(1) %out, i1 %uni) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br i1 %uni, label %header, label %exit

header:                                           ; preds = %body, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %body ]
  %uni.cond = icmp slt i32 %ctr, 1
  br i1 %uni.cond, label %body, label %exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %body, %header, %entry
  %acc.outside = phi i32 [ 7, %body ], [ 7, %header ], [ 99, %entry ]
  store i32 %acc.outside, ptr addrspace(1) %out
  ret void
}

; Two blocks of the cycle branch to %exit, but the %header edge is statically
; dead. AMDGPUUnifyDivergentExitNodes creates exactly this shape when it wires a
; non-returning loop to the dummy return block. Only one edge can be taken, so
; the phi has a single reaching value and stays uniform.

; CHECK-LABEL: UniformityInfo for function 'dead_exit_edge':
; CHECK-NOT: DIVERGENT:   %acc.dead
define amdgpu_kernel void @dead_exit_edge(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %body, %entry
  br i1 true, label %body, label %exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %body, %header
  %acc.dead = phi i32 [ 1, %body ], [ 0, %header ]
  store i32 %acc.dead, ptr addrspace(1) %out
  ret void
}

; As @dead_exit_edge, but the statically dead edge leaves a switch on a constant
; rather than a conditional branch. The case value matches, so the default edge
; into %exit cannot be taken.

; CHECK-LABEL: UniformityInfo for function 'dead_exit_edge_switch':
; CHECK-NOT: DIVERGENT:   %acc.dead.sw
define amdgpu_kernel void @dead_exit_edge_switch(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %body, %entry
  switch i32 1, label %exit [ i32 1, label %body ]

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %body, %header
  %acc.dead.sw = phi i32 [ 1, %body ], [ 0, %header ]
  store i32 %acc.dead.sw, ptr addrspace(1) %out
  ret void
}

; %header reaches %exit through two switch cases, but both leave the cycle along
; the same edge, so they count once. The remaining predecessor of %exit is
; outside the cycle, so only one edge leaves the cycle and %acc.dedup is uniform
; despite having more than one reaching value.

; CHECK-LABEL: UniformityInfo for function 'duplicate_exit_edges':
; CHECK-NOT: DIVERGENT:   %acc.dedup
define amdgpu_kernel void @duplicate_exit_edges(ptr addrspace(1) %out, i32 %n, i1 %uni) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br i1 %uni, label %header, label %exit

header:                                           ; preds = %body, %entry
  switch i32 %n, label %body [ i32 0, label %exit
                               i32 1, label %exit ]

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %div.exit, label %header

div.exit:                                         ; preds = %body
  br label %ret.bb

exit:                                             ; preds = %header, %header, %entry
  %acc.dedup = phi i32 [ 4, %header ], [ 4, %header ], [ 9, %entry ]
  store i32 %acc.dedup, ptr addrspace(1) %out
  br label %ret.bb

ret.bb:                                           ; preds = %exit, %div.exit
  ret void
}

; The cycle is divergently exited into %div.exit, but %uni.exit is reached only
; by the two uniform branches in %header and %latch. Threads that are still in
; the cycle are converged, so they all leave through the same one of those two
; edges and %acc.uni is really uniform.
;
; FIXME: The rule counts exit edges without asking whether the branch that takes
; them is divergent, so this is conservatively reported as divergent. Answering
; that question needs the divergence of each exit terminator, which is only
; known once the worklist has settled.

; CHECK-LABEL: UniformityInfo for function 'two_uniform_exit_edges':
; CHECK: DIVERGENT:   %acc.uni = phi i32 [ 1, %header ], [ 0, %latch ]
define amdgpu_kernel void @two_uniform_exit_edges(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %latch, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %latch ]
  %uni.head = icmp slt i32 %ctr, 1
  br i1 %uni.head, label %body, label %uni.exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %div.exit, label %latch

latch:                                            ; preds = %body
  %uni.latch = icmp slt i32 %n, 1
  br i1 %uni.latch, label %header, label %uni.exit

div.exit:                                         ; preds = %body
  br label %done

uni.exit:                                         ; preds = %latch, %header
  %acc.uni = phi i32 [ 1, %header ], [ 0, %latch ]
  store i32 %acc.uni, ptr addrspace(1) %out
  br label %done

done:                                             ; preds = %uni.exit, %div.exit
  ret void
}

; An undef input agrees with every other input, so a PHI whose only other
; reaching value is a constant has a single reaching value and stays uniform.
; This matches isConstantOrUndefValuePhi, which taintAndPushPhiNodes already
; applies to the same merge at a plain join block.
; CHECK-LABEL: for function 'undef_on_exit_edge':
; CHECK-NOT: DIVERGENT: %acc.undef
define amdgpu_kernel void @undef_on_exit_edge(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %body, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %body ]
  %uni.cond = icmp slt i32 %ctr, 1
  br i1 %uni.cond, label %body, label %exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %body, %header
  %acc.undef = phi i32 [ undef, %body ], [ 0, %header ]
  store i32 %acc.undef, ptr addrspace(1) %out
  ret void
}

; Same, for poison: PoisonValue derives from UndefValue, so getPhiInputs maps
; it to the null value reference too.
; CHECK-LABEL: for function 'poison_on_exit_edge':
; CHECK-NOT: DIVERGENT: %acc.poison
define amdgpu_kernel void @poison_on_exit_edge(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %body, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %body ]
  %uni.cond = icmp slt i32 %ctr, 1
  br i1 %uni.cond, label %body, label %exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %header

exit:                                             ; preds = %body, %header
  %acc.poison = phi i32 [ poison, %body ], [ 0, %header ]
  store i32 %acc.poison, ptr addrspace(1) %out
  ret void
}

; An undef input must not mask a genuine disagreement between two live
; values: with three exit edges carrying undef, 1 and 2, the PHI is divergent.
; CHECK-LABEL: for function 'undef_does_not_mask_disagreement':
; CHECK: DIVERGENT: %acc.mixed
define amdgpu_kernel void @undef_does_not_mask_disagreement(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:                                           ; preds = %body2, %entry
  %ctr = phi i32 [ 0, %entry ], [ 1, %body2 ]
  %uni.cond = icmp slt i32 %ctr, 1
  br i1 %uni.cond, label %body, label %exit

body:                                             ; preds = %header
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %body2

body2:                                            ; preds = %body
  %div.cond2 = icmp eq i32 %tid, 1
  br i1 %div.cond2, label %exit, label %header

exit:                                             ; preds = %body2, %body, %header
  %acc.mixed = phi i32 [ undef, %header ], [ 1, %body ], [ 2, %body2 ]
  store i32 %acc.mixed, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone speculatable }
