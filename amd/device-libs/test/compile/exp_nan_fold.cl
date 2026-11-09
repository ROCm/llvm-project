// Verify that ocml functions with an exp-style exponent constant fold to NaN
// for NaN input. They compute (int)rint(x * log2(e)) for ldexp, and fptosi of
// NaN is poison, so the conversion is guarded against NaN.
// Run with filecheck from test cmake

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float __ocml_sinh_f32(float);
float __ocml_cosh_f32(float);
float __ocml_expm1_f32(float);
double __ocml_sinh_f64(double);
double __ocml_cosh_f64(double);
double __ocml_tanh_f64(double);
double __ocml_expm1_f64(double);
double __ocml_exp_f64(double);
double __ocml_exp2_f64(double);
double __ocml_exp10_f64(double);

// CONSTANTFOLD-LABEL: @constant_fold_exp_nan_f32(
kernel void constant_fold_exp_nan_f32(volatile global float* out) {
    // CONSTANTFOLD: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinh_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinh_f32(-__builtin_nansf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cosh_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cosh_f32(-__builtin_nansf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_expm1_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_expm1_f32(__builtin_nansf(""));
}

// CONSTANTFOLD-LABEL: @constant_fold_exp_nan_f64(
kernel void constant_fold_exp_nan_f64(volatile global double* out) {
    // CONSTANTFOLD: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinh_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinh_f64(-__builtin_nans(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cosh_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cosh_f64(-__builtin_nans(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tanh_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_expm1_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_exp_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_exp2_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_exp10_f64(__builtin_nan(""));
}
