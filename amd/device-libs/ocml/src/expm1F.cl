/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 x);

CONSTATTR float
MATH_MANGLE(expm1)(float x)
{
#if defined EXTRA_ACCURACY
    float2 e = sub(MATH_PRIVATE(epexpep)(con(x, 0.0f)), 1.0f);
    float z = e.hi;
#else
    // fn >= -24 keeps s - 1 exact; then |t| <= 1.4 for x >= -18
    float fn = BUILTIN_MAX_F32(BUILTIN_RINT_F32(x * 0x1.715476p+0f), -24.0f);
#if defined MORE_ACCURACY
    // t + tl = x - fn*ln2
    float th = BUILTIN_FMA_F32(-fn, 0x1.62e430p-1f, x);
    float t = BUILTIN_FMA_F32(-fn, -0x1.05c610p-29f, th);
    float tl = BUILTIN_FMA_F32(-fn, -0x1.05c610p-29f, th - t);
#else
    float t = BUILTIN_FMA_F32(-fn, -0x1.05c610p-29f, BUILTIN_FMA_F32(-fn, 0x1.62e430p-1f, x));
#endif
    float p = PE5(t, 0x1.a26762p-13f, 0x1.6d2e00p-10f, 0x1.110ff2p-7f, 0x1.555502p-5f, 0x1.555556p-3f,
                    0x1.000000p-1f);
    int e = fn == 128.0f ? 127 : (int)fn;
    float s = BUILTIN_FLDEXP_F32(1.0f, e);
#if defined MORE_ACCURACY
    // ph + pl = expm1(t + tl)
    float u = t * p;
    float ph = BUILTIN_FMA_F32(t, u, t);
    float pl = BUILTIN_FMA_F32(t, u, t - ph);
    pl = BUILTIN_FMA_F32(tl, ph, pl + tl);

    // s - 1 may be inexact for large e; a gets its exact residual
    float sm1 = s - 1.0f;
    float c = (s - sm1) - 1.0f;
    float a = BUILTIN_FMA_F32(s, ph, sm1);
    float z = a + (BUILTIN_FMA_F32(s, ph, sm1 - a) + BUILTIN_FMA_F32(s, pl, c));
#else
    p = BUILTIN_FMA_F32(t, t*p, t);
    float z = BUILTIN_FMA_F32(s, p, s - 1.0f);
#endif
    z = fn == 128.0 ? 2.0f*z : z;
#endif
    
    if (!FINITE_ONLY_OPT()) {
        z = x > 0x1.62e42ep+6f ? PINF_F32 : z;
    }

    z = x < -18.0f ? -1.0f : z;

    return z;
}

