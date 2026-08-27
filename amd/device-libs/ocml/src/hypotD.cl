/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#undef EXTRA_ACCURACY

CONSTATTR double
MATH_MANGLE(hypot)(double x, double y)
{
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double t = BUILTIN_MAX_F64(a, b);
    int e = BUILTIN_FREXP_EXP_F64(t);
    a = BUILTIN_FLDEXP_F64(a, -e);
    b = BUILTIN_FLDEXP_F64(b, -e);

#if defined EXTRA_ACCURACY
    double u = BUILTIN_MAX_F64(a, b);
    double v = BUILTIN_MIN_F64(a, b);

    double u2 = u * u;
    double v2 = v * v;
    double s2h = u2 + v2;
    double s2l = (BUILTIN_FMA_F64(u, u, -u2) + BUILTIN_FMA_F64(v, v, -v2)) + (v2 - (s2h - u2));

    double sh = MATH_FAST_SQRT(s2h);
    double l = BUILTIN_FMA_F64(-sh, sh, s2h) + s2l;
    double sl = l * (0.5 * BUILTIN_AMDGPU_RCP_F64(sh));
    double r = sh + sl;
    r = (s2h == 0.0) ? 0.0 : r;

    double ret = BUILTIN_FLDEXP_F64(r, e);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISUNORDERED_F64(x, y) ? QNAN_F64 : ret;
        ret = (BUILTIN_ISINF_F64(x) | BUILTIN_ISINF_F64(y)) ? PINF_F64 : ret;
    }
#else
    double ret = BUILTIN_FLDEXP_F64(MATH_FAST_SQRT(MATH_MAD(a, a, b * b)), e);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F64(t) ? PINF_F64 : ret;
    }
#endif

    return ret;
}
