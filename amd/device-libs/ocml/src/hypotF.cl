/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#undef EXTRA_ACCURACY

CONSTATTR float
MATH_MANGLE(hypot)(float x, float y)
{
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float t = BUILTIN_MAX_F32(a, b);
    int e = BUILTIN_FREXP_EXP_F32(t);
    a = BUILTIN_FLDEXP_F32(a, -e);
    b = BUILTIN_FLDEXP_F32(b, -e);

#if defined EXTRA_ACCURACY
    float u = BUILTIN_MAX_F32(a, b);
    float v = BUILTIN_MIN_F32(a, b);

    float u2 = u * u;
    float v2 = v * v;
    float s2h = u2 + v2;
    float s2l = (BUILTIN_FMA_F32(u, u, -u2) + BUILTIN_FMA_F32(v, v, -v2)) + (v2 - (s2h - u2));

    float sh = MATH_FAST_SQRT(s2h);
    float l = BUILTIN_FMA_F32(-sh, sh, s2h) + s2l;
    float sl = l * (0.5f * MATH_FAST_RCP(sh));
    float r = sh + sl;
    r = (s2h == 0.0f) ? 0.0f : r;

    float ret = BUILTIN_FLDEXP_F32(r, e);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISUNORDERED_F32(x, y) ? QNAN_F32 : ret;
        ret = (BUILTIN_ISINF_F32(x) | BUILTIN_ISINF_F32(y)) ? PINF_F32 : ret;
    }
#else
    float ret = BUILTIN_FLDEXP_F32(MATH_FAST_SQRT(MATH_MAD(a, a, b*b)), e);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F32(t) ? PINF_F32 : ret;
    }
#endif

    return ret;
}
