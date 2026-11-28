/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern CONSTATTR float MATH_PRIVATE(atanred)(float);
extern CONSTATTR float MATH_PRIVATE(atan2ep)(float2 y, float2 x);

CONSTATTR float
MATH_MANGLE(atan2)(float y, float x)
{
#if defined EXTRA_ACCURACY
    return MATH_PRIVATE(atan2ep)((float2)(0.0f, y), (float2)(0.0f, x));
#else
    const float pi = 0x1.921fb6p+1f;
    const float piby2 = 0x1.921fb6p+0f;
    const float piby4 = 0x1.921fb6p-1f;
    const float threepiby4 = 0x1.2d97c8p+1f;

    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    float v = BUILTIN_MIN_F32(ax, ay);
    float u = BUILTIN_MAX_F32(ax, ay);

#if defined MORE_ACCURACY
    // v / u with one Newton step on the mantissa quotient
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

    float a = MATH_PRIVATE(atanred)(vbyu);

#if defined MORE_ACCURACY
    const float piby2_l = -0x1.777a5cp-25f;
    const float pi_l = -0x1.777a5cp-24f;
    float t = (piby2 - a) + piby2_l;
    a = ay > ax ? t : a;
    t = (pi - a) + pi_l;
    a = x < 0.0f ? t : a;
#else
    float t = piby2 - a;
    a = ay > ax ? t : a;
    t = pi - a;
    a = x < 0.0f ? t : a;
#endif

    t = AS_INT(x) < 0 ? pi : 0.0f;
    a = y == 0.0f ? t : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        t = x < 0.0f ? threepiby4 : piby4;
        a = (BUILTIN_ISINF_F32(x) & BUILTIN_ISINF_F32(y)) ? t : a;

        // x or y is NaN
        a = BUILTIN_ISUNORDERED_F32(x, y) ? QNAN_F32 : a;
    }

    return BUILTIN_COPYSIGN_F32(a, y);
#endif
}

