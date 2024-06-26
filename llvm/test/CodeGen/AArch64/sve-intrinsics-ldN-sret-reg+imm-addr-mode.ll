; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=aarch64--linux-gnu -mattr=sve < %s | FileCheck %s
; RUN: llc -mtriple=aarch64--linux-gnu -mattr=sme -force-streaming < %s | FileCheck %s

; NOTE: invalid, upper and lower bound immediate values of the regimm
; addressing mode are checked only for the byte version of each
; instruction (`ld<N>b`), as the code for detecting the immediate is
; common to all instructions, and varies only for the number of
; elements of the structure store, which is <N> = 2, 3, 4.

; ld2b
define { <vscale x 16 x i8>, <vscale x 16 x i8> } @ld2.nxv32i8(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv32i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2b { z0.b, z1.b }, p0/z, [x0, #2, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 2
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @ld2.nxv32i8_lower_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv32i8_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2b { z0.b, z1.b }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 -16
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @ld2.nxv32i8_upper_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv32i8_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2b { z0.b, z1.b }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 14
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @ld2.nxv32i8_not_multiple_of_2(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv32i8_not_multiple_of_2:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #3
; CHECK-NEXT:    ld2b { z0.b, z1.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 3
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @ld2.nxv32i8_outside_lower_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv32i8_outside_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #-18
; CHECK-NEXT:    ld2b { z0.b, z1.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 -18
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8> } @ld2.nxv32i8_outside_upper_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv32i8_outside_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #16
; CHECK-NEXT:    ld2b { z0.b, z1.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 16
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

; ld2h
define { <vscale x 8 x i16>, <vscale x 8 x i16> } @ld2.nxv16i16(<vscale x 8 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv16i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2h { z0.h, z1.h }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x i16>, ptr %addr, i64 14
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.ld2.sret.nxv8i16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 8 x half>, <vscale x 8 x half> } @ld2.nxv16f16(<vscale x 8 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2h { z0.h, z1.h }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x half>, ptr %addr, i64 -16
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x half>, <vscale x 8 x half> } @llvm.aarch64.sve.ld2.sret.nxv8f16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x half>, <vscale x 8 x half> } %res
}

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @ld2.nxv16bf16(<vscale x 8 x i1> %Pg, ptr %addr) #0 {
; CHECK-LABEL: ld2.nxv16bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2h { z0.h, z1.h }, p0/z, [x0, #12, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x bfloat>, ptr %addr, i64 12
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.ld2.sret.nxv8bf16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %res
}

; ld2w
define { <vscale x 4 x i32>, <vscale x 4 x i32> } @ld2.nxv8i32(<vscale x 4 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv8i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2w { z0.s, z1.s }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 4 x i32>, ptr %addr, i64 14
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.ld2.sret.nxv4i32(<vscale x 4 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 4 x float>, <vscale x 4 x float> } @ld2.nxv8f32(<vscale x 4 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2w { z0.s, z1.s }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 4 x float>, ptr %addr, i64 -16
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 4 x float>, <vscale x 4 x float> } @llvm.aarch64.sve.ld2.sret.nxv4f32(<vscale x 4 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 4 x float>, <vscale x 4 x float> } %res
}

; ld2d
define { <vscale x 2 x i64>, <vscale x 2 x i64> } @ld2.nxv4i64(<vscale x 2 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2d { z0.d, z1.d }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 2 x i64>, ptr %addr, i64 14
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld2.sret.nxv2i64(<vscale x 2 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

define { <vscale x 2 x double>, <vscale x 2 x double> } @ld2.nxv4f64(<vscale x 2 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld2.nxv4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld2d { z0.d, z1.d }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 2 x double>, ptr %addr, i64 -16
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld2.sret.nxv2f64(<vscale x 2 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 2 x double>, <vscale x 2 x double> } %res
}

; ld3b
define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld3.nxv48i8(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv48i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3b { z0.b - z2.b }, p0/z, [x0, #3, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 3
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld3.nxv48i8_lower_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv48i8_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3b { z0.b - z2.b }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 -24
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld3.nxv48i8_upper_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv48i8_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3b { z0.b - z2.b }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 21
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld3.nxv48i8_not_multiple_of_3_01(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv48i8_not_multiple_of_3_01:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #4
; CHECK-NEXT:    ld3b { z0.b - z2.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 4
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld3.nxv48i8_not_multiple_of_3_02(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv48i8_not_multiple_of_3_02:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #5
; CHECK-NEXT:    ld3b { z0.b - z2.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 5
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld3.nxv48i8_outside_lower_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv48i8_outside_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #-27
; CHECK-NEXT:    ld3b { z0.b - z2.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 -27
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld3.nxv48i8_outside_upper_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv48i8_outside_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #24
; CHECK-NEXT:    ld3b { z0.b - z2.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 24
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

; ld3h
define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @ld3.nxv24i16(<vscale x 8 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv24i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3h { z0.h - z2.h }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x i16>, ptr %addr, i64 21
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.ld3.sret.nxv8i16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } @ld3.nxv24f16(<vscale x 8 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv24f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3h { z0.h - z2.h }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x half>, ptr %addr, i64 21
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } @llvm.aarch64.sve.ld3.sret.nxv8f16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } %res
}

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @ld3.nxv24bf16(<vscale x 8 x i1> %Pg, ptr %addr) #0 {
; CHECK-LABEL: ld3.nxv24bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3h { z0.h - z2.h }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x bfloat>, ptr %addr, i64 -24
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.ld3.sret.nxv8bf16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %res
}

; ld3w
define { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @ld3.nxv12i32(<vscale x 4 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv12i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3w { z0.s - z2.s }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 4 x i32>, ptr %addr, i64 21
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.ld3.sret.nxv4i32(<vscale x 4 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @ld3.nxv12f32(<vscale x 4 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv12f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3w { z0.s - z2.s }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 4 x float>, ptr %addr, i64 -24
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @llvm.aarch64.sve.ld3.sret.nxv4f32(<vscale x 4 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } %res
}

; ld3d
define { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @ld3.nxv6i64(<vscale x 2 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv6i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3d { z0.d - z2.d }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 2 x i64>, ptr %addr, i64 21
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld3.sret.nxv2i64(<vscale x 2 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

define { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } @ld3.nxv6f64(<vscale x 2 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld3.nxv6f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld3d { z0.d - z2.d }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 2 x double>, ptr %addr, i64 -24
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld3.sret.nxv2f64(<vscale x 2 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } %res
}

; ; ld4b
define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, #4, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 4
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8_lower_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 -32
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8_upper_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 28
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8_not_multiple_of_4_01(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8_not_multiple_of_4_01:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #5
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 5
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8_not_multiple_of_4_02(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8_not_multiple_of_4_02:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #6
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 6
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8_not_multiple_of_4_03(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8_not_multiple_of_4_03:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #7
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 7
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8_outside_lower_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8_outside_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #1
; CHECK-NEXT:    mov x9, #-576
; CHECK-NEXT:    lsr x8, x8, #4
; CHECK-NEXT:    mul x8, x8, x9
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
; FIXME: optimize OFFSET computation so that xOFFSET = (mul (RDVL #4) #9)
; xM = -9 * 2^6
; xP = RDVL * 2^-4
; xOFFSET = RDVL * 2^-4 * -9 * 2^6 = RDVL * -36
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 -36
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

define { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @ld4.nxv64i8_outside_upper_bound(<vscale x 16 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv64i8_outside_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #1
; CHECK-NEXT:    mov w9, #512
; CHECK-NEXT:    lsr x8, x8, #4
; CHECK-NEXT:    mul x8, x8, x9
; CHECK-NEXT:    ld4b { z0.b - z3.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
; FIXME: optimize OFFSET computation so that xOFFSET = (mul (RDVL #16) #2)
; xM = 2^9
; xP = RDVL * 2^-4
; xOFFSET = RDVL * 2^-4 * 2^9 = RDVL * 32
  %base = getelementptr <vscale x 16 x i8>, ptr %addr, i64 32
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } %res
}

; ld4h
define { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @ld4.nxv32i16(<vscale x 8 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv32i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4h { z0.h - z3.h }, p0/z, [x0, #8, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x i16>, ptr %addr, i64 8
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.ld4.sret.nxv8i16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } %res
}

define { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } @ld4.nxv32f16(<vscale x 8 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv32f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4h { z0.h - z3.h }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x half>, ptr %addr, i64 28
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } @llvm.aarch64.sve.ld4.sret.nxv8f16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } %res
}

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @ld4.nxv32bf16(<vscale x 8 x i1> %Pg, ptr %addr) #0 {
; CHECK-LABEL: ld4.nxv32bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4h { z0.h - z3.h }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x bfloat>, ptr %addr, i64 -32
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.ld4.sret.nxv8bf16(<vscale x 8 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %res
}

; ld4w
define { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @ld4.nxv16i32(<vscale x 4 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv16i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4w { z0.s - z3.s }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 4 x i32>, ptr %addr, i64 28
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.ld4.sret.nxv4i32(<vscale x 4 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

define { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @ld4.nxv16f32(<vscale x 4 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv16f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4w { z0.s - z3.s }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 4 x float>, ptr %addr, i64 -32
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @llvm.aarch64.sve.ld4.sret.nxv4f32(<vscale x 4 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } %res
}

; ld4d
define { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @ld4.nxv8i64(<vscale x 2 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv8i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4d { z0.d - z3.d }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 2 x i64>, ptr %addr, i64 28
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld4.sret.nxv2i64(<vscale x 2 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } %res
}

define { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } @ld4.nxv8f64(<vscale x 2 x i1> %Pg, ptr %addr) {
; CHECK-LABEL: ld4.nxv8f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ld4d { z0.d - z3.d }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 2 x double>, ptr %addr, i64 -32
  %base_ptr = bitcast ptr %base to ptr
  %res = call { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld4.sret.nxv2f64(<vscale x 2 x i1> %Pg, ptr %base_ptr)
  ret { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } %res
}

declare { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld2.sret.nxv16i8(<vscale x 16 x i1>, ptr)
declare { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.ld2.sret.nxv8i16(<vscale x 8 x i1>, ptr)
declare { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.ld2.sret.nxv4i32(<vscale x 4 x i1>, ptr)
declare { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld2.sret.nxv2i64(<vscale x 2 x i1>, ptr)
declare { <vscale x 8 x half>, <vscale x 8 x half> } @llvm.aarch64.sve.ld2.sret.nxv8f16(<vscale x 8 x i1>, ptr)
declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.ld2.sret.nxv8bf16(<vscale x 8 x i1>, ptr)
declare { <vscale x 4 x float>, <vscale x 4 x float> } @llvm.aarch64.sve.ld2.sret.nxv4f32(<vscale x 4 x i1>, ptr)
declare { <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld2.sret.nxv2f64(<vscale x 2 x i1>, ptr)

declare { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld3.sret.nxv16i8(<vscale x 16 x i1>, ptr)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.ld3.sret.nxv8i16(<vscale x 8 x i1>, ptr)
declare { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.ld3.sret.nxv4i32(<vscale x 4 x i1>, ptr)
declare { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld3.sret.nxv2i64(<vscale x 2 x i1>, ptr)
declare { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } @llvm.aarch64.sve.ld3.sret.nxv8f16(<vscale x 8 x i1>, ptr)
declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.ld3.sret.nxv8bf16(<vscale x 8 x i1>, ptr)
declare { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @llvm.aarch64.sve.ld3.sret.nxv4f32(<vscale x 4 x i1>, ptr)
declare { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld3.sret.nxv2f64(<vscale x 2 x i1>, ptr)

declare { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.ld4.sret.nxv16i8(<vscale x 16 x i1>, ptr)
declare { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.ld4.sret.nxv8i16(<vscale x 8 x i1>, ptr)
declare { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.ld4.sret.nxv4i32(<vscale x 4 x i1>, ptr)
declare { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.ld4.sret.nxv2i64(<vscale x 2 x i1>, ptr)
declare { <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half>, <vscale x 8 x half> } @llvm.aarch64.sve.ld4.sret.nxv8f16(<vscale x 8 x i1>, ptr)
declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.ld4.sret.nxv8bf16(<vscale x 8 x i1>, ptr)
declare { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @llvm.aarch64.sve.ld4.sret.nxv4f32(<vscale x 4 x i1>, ptr)
declare { <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld4.sret.nxv2f64(<vscale x 2 x i1>, ptr)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+bf16" }
