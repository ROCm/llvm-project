/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern CONSTATTR float MATH_PRIVATE(atanpired)(float);

CONSTATTR float
MATH_MANGLE(atan2pi)(float y, float x)
{
    const float pi = 0x1.921fb6p+1f;

    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    float v = BUILTIN_MIN_F32(ax, ay);
    float u = BUILTIN_MAX_F32(ax, ay);

#if defined MORE_ACCURACY
    // v / u with one Newton step on the mantissa quotient (MATH_DIV is 2.5 ulp)
    int eu, ev;
    float mu = BUILTIN_FREXP_F32(u, &eu);
    float mv = BUILTIN_FREXP_F32(v, &ev);
    float rc = BUILTIN_AMDGPU_RCP_F32(mu);
    float q = mv * rc;
    q = BUILTIN_FMA_F32(BUILTIN_FMA_F32(-q, mu, mv), rc, q);
    float vbyu = BUILTIN_FLDEXP_F32(q, ev - eu);
    if (!FINITE_ONLY_OPT())
        vbyu = BUILTIN_ISINF_F32(u) ? 0.0f : vbyu;
#else
    float vbyu = MATH_DIV(v, u);
#endif

    float a = MATH_PRIVATE(atanpired)(vbyu);

    float at = 0.5f - a;
    a = ay > ax ? at : a;
    at = 1.0f - a;
    a = x < 0.0f ? at : a;

    at = AS_INT(x) < 0 ? 1.0f : 0.0f;
    a = y == 0.0f ? at : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        at = x < 0.0f ? 0.75f : 0.25f;
        a = (BUILTIN_ISINF_F32(x) & BUILTIN_ISINF_F32(y)) ? at : a;

        // x or y is NaN
        a = BUILTIN_ISUNORDERED_F32(x, y) ? QNAN_F32 : a;
    }

    return BUILTIN_COPYSIGN_F32(a, y);
}
