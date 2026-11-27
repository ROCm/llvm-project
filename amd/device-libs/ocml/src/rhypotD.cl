/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(rhypot)(double x, double y)
{
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double t = BUILTIN_MAX_F64(a, b);
    int e = BUILTIN_FREXP_EXP_F64(t);
    a = BUILTIN_FLDEXP_F64(a, -e);
    b = BUILTIN_FLDEXP_F64(b, -e);
#if defined EXTRA_ACCURACY
    double b2 = b * b;
    double d2 = MATH_MAD(a, a, b2);
    double z = BUILTIN_AMDGPU_RSQRT_F64(d2);

    // Residual 1 - (d2 + b2l)*z^2 including the error of d2*z and the
    // rounding error of b*b
    double p = d2 * z;
    double u = MATH_MAD(-p, z, 1.0);
    u = MATH_MAD(-MATH_MAD(d2, z, -p), z, u);
    u = MATH_MAD(-MATH_MAD(b, b, -b2), z*z, u);
#else
    double d2 = MATH_MAD(a, a, b*b);
    double z = BUILTIN_AMDGPU_RSQRT_F64(d2);
    double u = MATH_MAD(-d2*z, z, 1.0);
#endif
    z = MATH_MAD(z*u, MATH_MAD(u, 0.375, 0.5), z);
    double ret = BUILTIN_FLDEXP_F64(z, -e);

    if (!FINITE_ONLY_OPT()) {
        ret = t == 0.0 ? PINF_F64 : ret;

        ret = BUILTIN_ISUNORDERED_F64(x, y) ? QNAN_F64 : ret;

        ret = (BUILTIN_ISINF_F64(x) | BUILTIN_ISINF_F64(y)) ? 0.0 : ret;
    }

    return ret;
}

