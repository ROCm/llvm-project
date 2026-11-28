
// Verify that ocml's log(float) is lowered through the extended-precision
// polynomial path (via MATH_PRIVATE(lnep)) rather than the hardware
// v_log_f32 primitive. The native path is available separately via
// __ocml_native_log_f32 (see test/compile/native_log.cl).

// CHECK-LABEL: {{^}}test_log_f32:

// The polynomial path starts with frexp to split mantissa/exponent.
// CHECK: v_frexp_mant_f32

// The polynomial evaluation is a sequence of fmas.
// CHECK: v_fma_f32

// The primary log hardware instruction must NOT appear in the generated
// ISA — that would mean ocml_log has regressed back to the fast path
// that matches __ocml_native_log_f32.
// CHECK-NOT: v_log_f32

float test_log_f32(float arg) {
    return log(arg);
}
