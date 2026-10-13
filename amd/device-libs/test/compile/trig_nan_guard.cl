// Trig reduction guards the quadrant-index cast against NaN input:
//   isnan(fn) ? 0 : ((int)fn & 0x3)   (and the (short) variant for half)
// v_cvt_i*_f* already returns 0 for NaN, so the isnan guard must fold away:
// the result is the same convert (+ mask) as the unguarded cast, with no
// compare/select. See llvm/llvm-project#201435.
// __builtin_isnan is what the device-libs BUILTIN_ISNAN_F* macros expand to.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// float -> int: trigredsmallF, trigredH, trigpiredF
// CHECK-LABEL: {{^}}test_isnan_guard_fptosi_f32:
// CHECK: v_cvt_i32_f32
// CHECK-NOT: v_cmp
// CHECK-NOT: v_cndmask
int
test_isnan_guard_fptosi_f32(float x)
{
    return __builtin_isnan(x) ? 0 : ((int)x & 0x3);
}

// double -> int: trigredsmallD, trigpiredD
// CHECK-LABEL: {{^}}test_isnan_guard_fptosi_f64:
// CHECK: v_cvt_i32_f64
// CHECK-NOT: v_cmp
// CHECK-NOT: v_cndmask
int
test_isnan_guard_fptosi_f64(double x)
{
    return __builtin_isnan(x) ? 0 : ((int)x & 0x3);
}

// half -> short: trigpiredH
// CHECK-LABEL: {{^}}test_isnan_guard_fptosi_f16:
// CHECK: v_cvt_i16_f16
// CHECK-NOT: v_cmp
// CHECK-NOT: v_cndmask
short
test_isnan_guard_fptosi_f16(half x)
{
    return __builtin_isnan(x) ? (short)0 : ((short)x & (short)0x3);
}
