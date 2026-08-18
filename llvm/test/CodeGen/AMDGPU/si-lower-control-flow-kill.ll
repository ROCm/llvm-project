; RUN: llc -amdgpu-late-wave-transform=1 -mtriple=amdgpu6.00 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}if_with_kill:
; GCN: v_cmp_ne_u32_e32 vcc, 32, v0
; GCN: s_xor_b64 [[COND:s\[[0-9]+:[0-9]+\]]], vcc, exec
; GCN: s_mov_b64 exec, [[COND]]
define amdgpu_ps void @if_with_kill(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  tail call void @llvm.amdgcn.kill(i1 false)
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}if_with_loop_kill_after:
; GCN: v_cmp_ne_u32_e32 vcc, 32, v0
; GCN: s_xor_b64 [[COND:s\[[0-9]+:[0-9]+\]]], vcc, exec
; GCN: s_mov_b64 exec, [[COND]]
define amdgpu_ps void @if_with_loop_kill_after(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  %sub = sub i32 %arg, 1
  br label %loop

loop:
  %ind = phi i32 [%sub, %then], [%dec, %loop]
  %dec = sub i32 %ind, 1
  %cc = icmp ne i32 %ind, 0
  br i1 %cc, label %loop, label %break

break:
  tail call void @llvm.amdgcn.kill(i1 false)
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}if_with_kill_inside_loop:
; GCN: v_cmp_ne_u32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], 32, v0
; GCN: s_xor_b64 [[COND:s\[[0-9]+:[0-9]+\]]], [[CMP]], exec
; GCN: s_mov_b64 exec, [[COND]]
define amdgpu_ps void @if_with_kill_inside_loop(i32 %arg) {
.entry:
  %cmp = icmp eq i32 %arg, 32
  br i1 %cmp, label %then, label %endif

then:
  %sub = sub i32 %arg, 1
  br label %loop

loop:
  %ind = phi i32 [%sub, %then], [%dec, %loop]
  %dec = sub i32 %ind, 1
  %cc = icmp ne i32 %ind, 0
  tail call void @llvm.amdgcn.kill(i1 false)
  br i1 %cc, label %loop, label %break

break:
  br label %endif

endif:
  ret void
}

declare void @llvm.amdgcn.kill(i1) #0

attributes #0 = { nounwind }
