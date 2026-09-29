; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes=instcombine -S < %s | FileCheck %s

; cvt.pk.{fp8,bf8}(a, b, old, word_sel) passes the unselected 16-bit half of
; `old` through. When every user overwrites exactly that half (it takes the
; result as `old` with the opposite word_sel), `old` is dead -> poison.

declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1 immarg)
declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1 immarg)

; CHECK-LABEL: @lo_then_hi(
; CHECK: %lo = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 poison, i1 false)
; CHECK: %hi = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %c, float %d, i32 %lo, i1 true)
define i32 @lo_then_hi(float %a, float %b, float %c, float %d) {
  %lo = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 0, i1 false)
  %hi = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %c, float %d, i32 %lo, i1 true)
  ret i32 %hi
}

; CHECK-LABEL: @hi_then_lo_mixed(
; CHECK: %hi = call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %a, float %b, i32 poison, i1 true)
define i32 @hi_then_lo_mixed(float %a, float %b, float %c, float %d, i32 %old) {
  %hi = call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %a, float %b, i32 %old, i1 true)
  %lo = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %c, float %d, i32 %hi, i1 false)
  ret i32 %lo
}

; Same word_sel: the old half survives.
; CHECK-LABEL: @same_half(
; CHECK: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 7, i1 false)
define i32 @same_half(float %a, float %b, float %c, float %d) {
  %x = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 7, i1 false)
  %y = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %c, float %d, i32 %x, i1 false)
  ret i32 %y
}

; The low-half result is also used directly: its high half is observable.
; CHECK-LABEL: @extra_use(
; CHECK: %lo = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 7, i1 false)
define i32 @extra_use(float %a, float %b, float %c, float %d, ptr %p) {
  %lo = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 7, i1 false)
  store i32 %lo, ptr %p
  %hi = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %c, float %d, i32 %lo, i1 true)
  ret i32 %hi
}

; Result used only directly (no second convert): keep `old`.
; CHECK-LABEL: @no_overwrite(
; CHECK: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 7, i1 false)
define i32 @no_overwrite(float %a, float %b) {
  %lo = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 7, i1 false)
  ret i32 %lo
}
