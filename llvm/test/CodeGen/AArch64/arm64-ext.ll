; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc < %s -mtriple=arm64-eabi -global-isel=0 | FileCheck %s --check-prefixes=CHECK,CHECK-SD
; RUN: llc < %s -mtriple=arm64-eabi -global-isel=1 | FileCheck %s --check-prefixes=CHECK,CHECK-GI

define <8 x i8> @test_vextd(<8 x i8> %tmp1, <8 x i8> %tmp2) {
; CHECK-LABEL: test_vextd:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #3
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x i8> %tmp3
}

define <8 x i8> @test_vextRd(<8 x i8> %tmp1, <8 x i8> %tmp2) {
; CHECK-LABEL: test_vextRd:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v1.8b, v0.8b, #5
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4>
  ret <8 x i8> %tmp3
}

define <16 x i8> @test_vextq(<16 x i8> %tmp1, <16 x i8> %tmp2) {
; CHECK-LABEL: test_vextq:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #3
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
  ret <16 x i8> %tmp3
}

define <16 x i8> @test_vextRq(<16 x i8> %tmp1, <16 x i8> %tmp2) {
; CHECK-LABEL: test_vextRq:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v1.16b, v0.16b, #7
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6>
  ret <16 x i8> %tmp3
}

define <4 x i16> @test_vextd16(<4 x i16> %tmp1, <4 x i16> %tmp2) {
; CHECK-LABEL: test_vextd16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #6
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i16> %tmp3
}

define <4 x i32> @test_vextq32(<4 x i32> %tmp1, <4 x i32> %tmp2) {
; CHECK-LABEL: test_vextq32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #12
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i32> %tmp3
}

; Undef shuffle indices should not prevent matching to VEXT:

define <8 x i8> @test_vextd_undef(<8 x i8> %tmp1, <8 x i8> %tmp2) {
; CHECK-LABEL: test_vextd_undef:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #3
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 3, i32 undef, i32 undef, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x i8> %tmp3
}

define <8 x i8> @test_vextd_undef2(<8 x i8> %tmp1, <8 x i8> %tmp2) {
; CHECK-LABEL: test_vextd_undef2:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v0.8b, #6
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 2, i32 3, i32 4, i32 5>
  ret <8 x i8> %tmp3
}

define <16 x i8> @test_vextRq_undef(<16 x i8> %tmp1, <16 x i8> %tmp2) {
; CHECK-LABEL: test_vextRq_undef:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v1.16b, v0.16b, #7
; CHECK-NEXT:    ret
  %tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 23, i32 24, i32 25, i32 26, i32 undef, i32 undef, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 undef, i32 6>
  ret <16 x i8> %tmp3
}

define <8 x i16> @test_vextRq_undef2(<8 x i16> %tmp1) nounwind {
; CHECK-LABEL: test_vextRq_undef2:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v0.16b, #10
; CHECK-NEXT:    ret
  %vext = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 1, i32 2, i32 3, i32 4>
  ret <8 x i16> %vext;
}

; Tests for ReconstructShuffle function. Indices have to be carefully
; chosen to reach lowering phase as a BUILD_VECTOR.

; An undef in the shuffle list should still be optimizable
define <4 x i16> @test_undef(<8 x i16> %tmp1, <8 x i16> %tmp2) {
; CHECK-SD-LABEL: test_undef:
; CHECK-SD:       // %bb.0:
; CHECK-SD-NEXT:    ext v0.16b, v0.16b, v0.16b, #8
; CHECK-SD-NEXT:    zip1 v0.4h, v0.4h, v1.4h
; CHECK-SD-NEXT:    ret
;
; CHECK-GI-LABEL: test_undef:
; CHECK-GI:       // %bb.0:
; CHECK-GI-NEXT:    adrp x8, .LCPI10_0
; CHECK-GI-NEXT:    // kill: def $q0 killed $q0 killed $q0_q1 def $q0_q1
; CHECK-GI-NEXT:    ldr q2, [x8, :lo12:.LCPI10_0]
; CHECK-GI-NEXT:    // kill: def $q1 killed $q1 killed $q0_q1 def $q0_q1
; CHECK-GI-NEXT:    tbl v0.16b, { v0.16b, v1.16b }, v2.16b
; CHECK-GI-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-GI-NEXT:    ret
  %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <4 x i32> <i32 undef, i32 8, i32 5, i32 9>
  ret <4 x i16> %tmp3
}

define <2 x i64> @test_v2s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_v2s64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v1.16b, v0.16b, #8
; CHECK-NEXT:    ret
  %s = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 0>
  ret <2 x i64> %s
}

define <2 x ptr> @test_v2p0(<2 x ptr> %a, <2 x ptr> %b) {
; CHECK-LABEL: test_v2p0:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v1.16b, v0.16b, #8
; CHECK-NEXT:    ret
  %s = shufflevector <2 x ptr> %a, <2 x ptr> %b, <2 x i32> <i32 3, i32 0>
  ret <2 x ptr> %s
}
