// Verify that ocml functions constant fold to NaN for NaN and infinite input.
// They convert a NaN or infinite value to int (an exponent for ldexp, a
// quadrant index for trig reduction), which must not produce poison.
// Run with filecheck from test cmake

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

float
__ocml_cos_f32(float);
float
__ocml_cosh_f32(float);
float
__ocml_cospi_f32(float);
float
__ocml_expm1_f32(float);
float
__ocml_sin_f32(float);
float
__ocml_sinh_f32(float);
float
__ocml_sinpi_f32(float);
float
__ocml_tan_f32(float);
float
__ocml_tanpi_f32(float);
double
__ocml_cos_f64(double);
double
__ocml_cosh_f64(double);
double
__ocml_cospi_f64(double);
double
__ocml_exp_f64(double);
double
__ocml_exp10_f64(double);
double
__ocml_exp2_f64(double);
double
__ocml_expm1_f64(double);
double
__ocml_sin_f64(double);
double
__ocml_sinh_f64(double);
double
__ocml_sinpi_f64(double);
double
__ocml_tan_f64(double);
double
__ocml_tanh_f64(double);
double
__ocml_tanpi_f64(double);
half __ocml_cos_f16(half);
half __ocml_cospi_f16(half);
half __ocml_sin_f16(half);
half __ocml_sinpi_f16(half);
half __ocml_tan_f16(half);
half __ocml_tanpi_f16(half);

// CONSTANTFOLD-LABEL: @constant_fold_nan_f32(
kernel void
constant_fold_nan_f32(volatile global float *out)
{
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
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sin_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sin_f32(__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cos_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cos_f32(__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tan_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tan_f32(__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinpi_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinpi_f32(__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cospi_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cospi_f32(__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tanpi_f32(__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile float {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tanpi_f32(__builtin_inff());
}

// CONSTANTFOLD-LABEL: @constant_fold_nan_f64(
kernel void
constant_fold_nan_f64(volatile global double *out)
{
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
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sin_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sin_f64(__builtin_inf());
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cos_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cos_f64(__builtin_inf());
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tan_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tan_f64(__builtin_inf());
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinpi_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinpi_f64(__builtin_inf());
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cospi_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cospi_f64(__builtin_inf());
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tanpi_f64(__builtin_nan(""));
    // CONSTANTFOLD-NEXT: store volatile double {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tanpi_f64(__builtin_inf());
}

// CONSTANTFOLD-LABEL: @constant_fold_nan_f16(
kernel void
constant_fold_nan_f16(volatile global half *out)
{
    // CONSTANTFOLD: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sin_f16((half)__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sin_f16((half)__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cos_f16((half)__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cos_f16((half)__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tan_f16((half)__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tan_f16((half)__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinpi_f16((half)__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_sinpi_f16((half)__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cospi_f16((half)__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_cospi_f16((half)__builtin_inff());
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tanpi_f16((half)__builtin_nanf(""));
    // CONSTANTFOLD-NEXT: store volatile half {{[-+](qnan|nan\(0x[0-9a-f]+\))}},
    out[0] = __ocml_tanpi_f16((half)__builtin_inff());
}
