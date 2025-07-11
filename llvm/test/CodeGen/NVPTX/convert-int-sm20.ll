; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

;; Integer conversions happen inplicitly by loading/storing the proper types

; i16

define i16 @cvt_i16_i32(i32 %x) {
; CHECK-LABEL: cvt_i16_i32(
; CHECK:       {
; CHECK-NEXT:    .reg .b32 %r<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b32 %r1, [cvt_i16_i32_param_0];
; CHECK-NEXT:    st.param.b32 [func_retval0], %r1;
; CHECK-NEXT:    ret;
  %a = trunc i32 %x to i16
  ret i16 %a
}

define i16 @cvt_i16_i64(i64 %x) {
; CHECK-LABEL: cvt_i16_i64(
; CHECK:       {
; CHECK-NEXT:    .reg .b32 %r<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b16 %r1, [cvt_i16_i64_param_0];
; CHECK-NEXT:    st.param.b32 [func_retval0], %r1;
; CHECK-NEXT:    ret;
  %a = trunc i64 %x to i16
  ret i16 %a
}

; i32

define i32 @cvt_i32_i16(i16 %x) {
; CHECK-LABEL: cvt_i32_i16(
; CHECK:       {
; CHECK-NEXT:    .reg .b32 %r<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b16 %r1, [cvt_i32_i16_param_0];
; CHECK-NEXT:    st.param.b32 [func_retval0], %r1;
; CHECK-NEXT:    ret;
  %a = zext i16 %x to i32
  ret i32 %a
}

define i32 @cvt_i32_i64(i64 %x) {
; CHECK-LABEL: cvt_i32_i64(
; CHECK:       {
; CHECK-NEXT:    .reg .b64 %rd<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b64 %rd1, [cvt_i32_i64_param_0];
; CHECK-NEXT:    st.param.b32 [func_retval0], %rd1;
; CHECK-NEXT:    ret;
  %a = trunc i64 %x to i32
  ret i32 %a
}

; i64

define i64 @cvt_i64_i16(i16 %x) {
; CHECK-LABEL: cvt_i64_i16(
; CHECK:       {
; CHECK-NEXT:    .reg .b64 %rd<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b16 %rd1, [cvt_i64_i16_param_0];
; CHECK-NEXT:    st.param.b64 [func_retval0], %rd1;
; CHECK-NEXT:    ret;
  %a = zext i16 %x to i64
  ret i64 %a
}

define i64 @cvt_i64_i32(i32 %x) {
; CHECK-LABEL: cvt_i64_i32(
; CHECK:       {
; CHECK-NEXT:    .reg .b64 %rd<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b32 %rd1, [cvt_i64_i32_param_0];
; CHECK-NEXT:    st.param.b64 [func_retval0], %rd1;
; CHECK-NEXT:    ret;
  %a = zext i32 %x to i64
  ret i64 %a
}
