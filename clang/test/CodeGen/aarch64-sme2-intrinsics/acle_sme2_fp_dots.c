// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64 -target-feature +bf16 -target-feature +sme -target-feature +sme2 -disable-O0-optnone -Werror -Wall -emit-llvm -o - %s | opt -S -p mem2reg,instcombine,tailcallelim | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +bf16 -target-feature +sme -target-feature +sme2 -disable-O0-optnone -Werror -Wall -emit-llvm -o - -x c++ %s | opt -S -p mem2reg,instcombine,tailcallelim | FileCheck %s -check-prefix=CPP-CHECK
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64 -target-feature +bf16 -target-feature +sme -target-feature +sme2 -disable-O0-optnone -Werror -Wall -emit-llvm -o - %s | opt -S -p mem2reg,instcombine,tailcallelim | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64 -target-feature +bf16 -target-feature +sme -target-feature +sme2 -disable-O0-optnone -Werror -Wall -emit-llvm -o - -x c++ %s | opt -S -p mem2reg,instcombine,tailcallelim | FileCheck %s -check-prefix=CPP-CHECK
// RUN: %clang_cc1 -triple aarch64 -target-feature +bf16 -target-feature +sme -target-feature +sme2 -S -disable-O0-optnone -Werror -Wall -o /dev/null %s
#include <arm_sme.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED,A5) A1##A3##A5
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4,A5) A1##A2##A3##A4##A5
#endif

//
// Multi, multi (half)
// CHECK-LABEL: @test_svdot_multi_za32_vg1x2_f16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZM:%.*]], i64 0)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZM]], i64 8)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x2.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z31test_svdot_multi_za32_vg1x2_f16j13svfloat16x2_tS_(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZM:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZM]], i64 8)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x2.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_multi_za32_vg1x2_f16(uint32_t slice_base, svfloat16x2_t zn, svfloat16x2_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_za32,,,_f16,_vg1x2)(slice_base, zn, zm);
}

// CHECK-LABEL: @test_svdot_multi_za32_vg1x4_f16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 16)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 24)
// CHECK-NEXT:    [[TMP4:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM:%.*]], i64 0)
// CHECK-NEXT:    [[TMP5:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM]], i64 8)
// CHECK-NEXT:    [[TMP6:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM]], i64 16)
// CHECK-NEXT:    [[TMP7:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM]], i64 24)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x4.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]], <vscale x 8 x half> [[TMP4]], <vscale x 8 x half> [[TMP5]], <vscale x 8 x half> [[TMP6]], <vscale x 8 x half> [[TMP7]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z31test_svdot_multi_za32_vg1x4_f16j13svfloat16x4_tS_(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 16)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 24)
// CPP-CHECK-NEXT:    [[TMP4:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP5:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM]], i64 8)
// CPP-CHECK-NEXT:    [[TMP6:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM]], i64 16)
// CPP-CHECK-NEXT:    [[TMP7:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZM]], i64 24)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x4.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]], <vscale x 8 x half> [[TMP4]], <vscale x 8 x half> [[TMP5]], <vscale x 8 x half> [[TMP6]], <vscale x 8 x half> [[TMP7]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_multi_za32_vg1x4_f16(uint32_t slice_base, svfloat16x4_t zn, svfloat16x4_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_za32,,,_f16,_vg1x4)(slice_base, zn, zm);
}


//
// Multi, single (half)
// CHECK-LABEL: @test_svdot_single_za32_vg1x2_f16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN]], i64 8)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x2.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z32test_svdot_single_za32_vg1x2_f16j13svfloat16x2_tu13__SVFloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x2.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[ZM:%.*]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_single_za32_vg1x2_f16(uint32_t slice_base, svfloat16x2_t zn, svfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_single_za32,,_f16,,_vg1x2)(slice_base, zn, zm);
}

// CHECK-LABEL: @test_svdot_single_za32_vg1x4_f16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 16)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 24)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x4.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]], <vscale x 8 x half> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z32test_svdot_single_za32_vg1x4_f16j13svfloat16x4_tu13__SVFloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 16)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 24)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x4.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]], <vscale x 8 x half> [[ZM:%.*]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_single_za32_vg1x4_f16(uint32_t slice_base, svfloat16x4_t zn, svfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_single_za32,,_f16,,_vg1x4)(slice_base, zn, zm);
}


//
// Multi, indexed (half)
// CHECK-LABEL: @test_svdot_lane_za32_vg1x2_f16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN]], i64 8)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x2.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[ZM:%.*]], i32 3)
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z30test_svdot_lane_za32_vg1x2_f16j13svfloat16x2_tu13__SVFloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv16f16(<vscale x 16 x half> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x2.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[ZM:%.*]], i32 3)
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_lane_za32_vg1x2_f16(uint32_t slice_base, svfloat16x2_t zn, svfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_lane_za32,,,_f16,_vg1x2)(slice_base, zn, zm, 3);
}

// CHECK-LABEL: @test_svdot_lane_za32_vg1x4_f16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 16)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 24)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x4.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]], <vscale x 8 x half> [[ZM:%.*]], i32 3)
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z30test_svdot_lane_za32_vg1x4_f16j13svfloat16x4_tu13__SVFloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 16)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x half> @llvm.vector.extract.nxv8f16.nxv32f16(<vscale x 32 x half> [[ZN]], i64 24)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x4.nxv8f16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x half> [[TMP0]], <vscale x 8 x half> [[TMP1]], <vscale x 8 x half> [[TMP2]], <vscale x 8 x half> [[TMP3]], <vscale x 8 x half> [[ZM:%.*]], i32 3)
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_lane_za32_vg1x4_f16(uint32_t slice_base, svfloat16x4_t zn, svfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_lane_za32,,,_f16,_vg1x4)(slice_base, zn, zm, 3);
}


//
// Multi, multi (bfloat)
// CHECK-LABEL: @test_svdot_multi_za32_vg1x2_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZM:%.*]], i64 0)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZM]], i64 8)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x2.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z32test_svdot_multi_za32_vg1x2_bf16j14svbfloat16x2_tS_(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZM:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZM]], i64 8)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x2.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_multi_za32_vg1x2_bf16(uint32_t slice_base, svbfloat16x2_t zn, svbfloat16x2_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_za32,,,_bf16,_vg1x2)(slice_base, zn, zm);
}

// CHECK-LABEL: @test_svdot_multi_za32_vg1x4_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 16)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 24)
// CHECK-NEXT:    [[TMP4:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM:%.*]], i64 0)
// CHECK-NEXT:    [[TMP5:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM]], i64 8)
// CHECK-NEXT:    [[TMP6:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM]], i64 16)
// CHECK-NEXT:    [[TMP7:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM]], i64 24)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x4.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]], <vscale x 8 x bfloat> [[TMP4]], <vscale x 8 x bfloat> [[TMP5]], <vscale x 8 x bfloat> [[TMP6]], <vscale x 8 x bfloat> [[TMP7]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z32test_svdot_multi_za32_vg1x4_bf16j14svbfloat16x4_tS_(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 16)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 24)
// CPP-CHECK-NEXT:    [[TMP4:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP5:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM]], i64 8)
// CPP-CHECK-NEXT:    [[TMP6:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM]], i64 16)
// CPP-CHECK-NEXT:    [[TMP7:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZM]], i64 24)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.za32.vg1x4.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]], <vscale x 8 x bfloat> [[TMP4]], <vscale x 8 x bfloat> [[TMP5]], <vscale x 8 x bfloat> [[TMP6]], <vscale x 8 x bfloat> [[TMP7]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_multi_za32_vg1x4_bf16(uint32_t slice_base, svbfloat16x4_t zn, svbfloat16x4_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_za32,,,_bf16,_vg1x4)(slice_base, zn, zm);
}


//
// Multi, single (bfloat)
// CHECK-LABEL: @test_svdot_single_za32_vg1x2_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN]], i64 8)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x2.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z33test_svdot_single_za32_vg1x2_bf16j14svbfloat16x2_tu14__SVBfloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x2.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[ZM:%.*]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_single_za32_vg1x2_bf16(uint32_t slice_base, svbfloat16x2_t zn, svbfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_single_za32,,_bf16,,_vg1x2)(slice_base, zn, zm);
}

// CHECK-LABEL: @test_svdot_single_za32_vg1x4_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 16)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 24)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x4.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]], <vscale x 8 x bfloat> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z33test_svdot_single_za32_vg1x4_bf16j14svbfloat16x4_tu14__SVBfloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 16)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 24)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.single.za32.vg1x4.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]], <vscale x 8 x bfloat> [[ZM:%.*]])
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_single_za32_vg1x4_bf16(uint32_t slice_base, svbfloat16x4_t zn, svbfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_single_za32,,_bf16,,_vg1x4)(slice_base, zn, zm);
}


//
// Multi, indexed (bfloat)
// CHECK-LABEL: @test_svdot_lane_za32_vg1x2_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN]], i64 8)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x2.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[ZM:%.*]], i32 3)
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z31test_svdot_lane_za32_vg1x2_bf16j14svbfloat16x2_tu14__SVBfloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv16bf16(<vscale x 16 x bfloat> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x2.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[ZM:%.*]], i32 3)
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_lane_za32_vg1x2_bf16(uint32_t slice_base, svbfloat16x2_t zn, svbfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_lane_za32,,_bf16,,_vg1x2)(slice_base, zn, zm, 3);
}

// CHECK-LABEL: @test_svdot_lane_za32_vg1x4_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN:%.*]], i64 0)
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 8)
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 16)
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 24)
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x4.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]], <vscale x 8 x bfloat> [[ZM:%.*]], i32 3)
// CHECK-NEXT:    ret void
//
// CPP-CHECK-LABEL: @_Z31test_svdot_lane_za32_vg1x4_bf16j14svbfloat16x4_tu14__SVBfloat16_t(
// CPP-CHECK-NEXT:  entry:
// CPP-CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN:%.*]], i64 0)
// CPP-CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 8)
// CPP-CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 16)
// CPP-CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x bfloat> @llvm.vector.extract.nxv8bf16.nxv32bf16(<vscale x 32 x bfloat> [[ZN]], i64 24)
// CPP-CHECK-NEXT:    tail call void @llvm.aarch64.sme.fdot.lane.za32.vg1x4.nxv8bf16(i32 [[SLICE_BASE:%.*]], <vscale x 8 x bfloat> [[TMP0]], <vscale x 8 x bfloat> [[TMP1]], <vscale x 8 x bfloat> [[TMP2]], <vscale x 8 x bfloat> [[TMP3]], <vscale x 8 x bfloat> [[ZM:%.*]], i32 3)
// CPP-CHECK-NEXT:    ret void
//
void test_svdot_lane_za32_vg1x4_bf16(uint32_t slice_base, svbfloat16x4_t zn, svbfloat16_t zm) __arm_streaming __arm_inout("za") {
  SVE_ACLE_FUNC(svdot_lane_za32,,_bf16,,_vg1x4)(slice_base, zn, zm, 3);
}
