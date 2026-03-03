/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

__ocml_sincos_f64_result
MATH_MANGLE(sincos_stret)(double x)
{
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F64(x) ? QNAN_F64 : x;

    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred2)(r.hi, r.lo);

    long flip = r.i > 1 ? SIGNBIT_DP64 : 0;
    bool odd = (r.i & 1) != 0;

    double s = odd ? sc.c : sc.s;
    s = AS_DOUBLE(AS_LONG(s) ^ flip ^ (AS_LONG(x) & SIGNBIT_DP64));
    sc.s = -sc.s;

    double c = odd ? sc.s : sc.c;
    c = AS_DOUBLE(AS_LONG(c) ^ flip);

    __ocml_sincos_f64_result result = {s, c};
    return result;
}

double
MATH_MANGLE(sincos)(double x, __private double *cp)
{
    __ocml_sincos_f64_result result = MATH_MANGLE(sincos_stret)(x);
    *cp = result.__cos;
    return result.__sin;
}
