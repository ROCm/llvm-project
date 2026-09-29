; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s

; A poison `old` operand needs no materialization (no v_mov before the
; low-half convert).

declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1 immarg)

; CHECK-LABEL: pack4:
; CHECK-NOT:   v_mov_b32
; CHECK:       v_cvt_pk_fp8_f32 [[R:v[0-9]+]], v0, v1
; CHECK-NEXT:  v_cvt_pk_fp8_f32 [[R]], v2, v3 op_sel:[0,0,1]
define i32 @pack4(float %a, float %b, float %c, float %d) {
  %lo = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 poison, i1 false)
  %hi = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %c, float %d, i32 %lo, i1 true)
  ret i32 %hi
}
