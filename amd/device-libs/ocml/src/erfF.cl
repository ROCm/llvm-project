/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if defined EXTRA_ACCURACY
#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float MATH_PRIVATE(expep)(float2);
#endif

CONSTATTR float
MATH_MANGLE(erf)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 1.0f) {
        float t = ax*ax;
#if defined EXTRA_ACCURACY
        // one more term, with c0 = 2/sqrt(pi) - 1 correctly rounded
        float p = PE6(t, 0x1.491b54p-14f, -0x1.a3d938p-11f, 0x1.54019p-8f, -0x1.b7f88ep-6f, 0x1.ce2cf2p-4f,
                        -0x1.81273ep-2f, 0x1.06eba8p-3f);
#else
        float p = PE5(t, -0x1.268bc2p-11f, 0x1.420828p-8f, -0x1.b5937p-6f, 0x1.ce077cp-4f, -0x1.81266p-2f,
                        0x1.06eba0p-3f);
#endif
        ret = BUILTIN_FMA_F32(ax, p, ax);
    } else {
        float p = PE6(ax, 0x1.1d3156p-16f, -0x1.8d129p-12f, 0x1.f9a6d2p-9f, -0x1.8c3164p-6f, 0x1.b4e9c8p-4f,
                        0x1.4515fap-1f, 0x1.078e50p-3f);
#if defined EXTRA_ACCURACY
        // ax + ax*p as hi + lo (ax - ph is exact near 1, where it matters)
        float ph = BUILTIN_FMA_F32(ax, p, ax);
        float pl = BUILTIN_FMA_F32(ax, p, ax - ph);
        ret = 1.0f - MATH_PRIVATE(expep)(con(-ph, -pl));
#else
        p = BUILTIN_FMA_F32(ax, p, ax);
        ret = 1.0f - MATH_MANGLE(exp)(-p);
#endif
    }

    ret = BUILTIN_COPYSIGN_F32(ret, x);
    return ret;
}

