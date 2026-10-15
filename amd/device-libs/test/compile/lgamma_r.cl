// Verify lgamma_r function constant folds to correct values.
// Run with filecheck from test cmake

__attribute__((always_inline))
static float test_lgamma_r(float val, volatile global int* sign_out) {
   int tmp;
   float result = lgamma_r(val, &tmp);
   *sign_out = tmp;
   return result;
}

// CHECK-LABEL: {{^}}constant_fold_lgamma_r_f32:
// CONSTANTFOLD-LABEL: @constant_fold_lgamma_r_f32(
kernel void constant_fold_lgamma_r_f32(volatile global float* out,
                                       volatile global int* sign_out) {
    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf
    out[0] = test_lgamma_r(0.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf
    out[0] = test_lgamma_r(-0.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float -qnan,
    out[0] = test_lgamma_r(__builtin_nanf(""), sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float -nan(0x200000),
    out[0] = test_lgamma_r(__builtin_nansf(""), sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float +inf,
    out[0] = test_lgamma_r(__builtin_inff(), sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf,
    out[0] = test_lgamma_r(-__builtin_inff(), sign_out);

    // Large arguments hit the Stirling/table path, which uses target
    // intrinsics (e.g. amdgcn.rcp) that do not constant fold. The result is a
    // runtime value and its computation may be interleaved with the stores, so
    // do not use -NEXT around these.
    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float %{{.*}}
    out[0] = test_lgamma_r(0x1.0p+23f, sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf,
    out[0] = test_lgamma_r(-0x1.0p+23f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float 0.000000e+00,
    out[0] = test_lgamma_r(1.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float 0.000000e+00,
    out[0] = test_lgamma_r(2.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float f0x3F317218,
    out[0] = test_lgamma_r(3.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float f0x3F128682,
    out[0] = test_lgamma_r(0.5f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float f0x42B00F34,
    out[0] = test_lgamma_r(0x1.0p-127f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float %{{.*}}
    out[0] = test_lgamma_r(nextafter(0x1.0p+23f, __builtin_inff()), sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float %{{.*}}
    out[0] = test_lgamma_r(nextafter(0x1.0p+23f, -__builtin_inff()), sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float %{{.*}}
    out[0] = test_lgamma_r(nextafter(-0x1.0p+23f, __builtin_inff()), sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf,
    out[0] = test_lgamma_r(nextafter(-0x1.0p+23f, -__builtin_inff()), sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf,
    out[0] = test_lgamma_r(-1.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf,
    out[0] = test_lgamma_r(-2.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 0,
    // CONSTANTFOLD: store volatile float +inf,
    out[0] = test_lgamma_r(-3.0f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float %{{.*}}
    out[0] = test_lgamma_r(-3.5f, sign_out);

    // CONSTANTFOLD: store volatile i32 1,
    // CONSTANTFOLD: store volatile float %{{.*}}
    out[0] = test_lgamma_r(as_float(0xcaffffff), sign_out);
}
