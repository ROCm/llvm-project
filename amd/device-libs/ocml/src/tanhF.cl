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

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 x);
#endif

CONSTATTR float
MATH_MANGLE(tanh)(float x)
{
    float y = BUILTIN_ABS_F32(x);

#if defined EXTRA_ACCURACY
    float2 e = MATH_PRIVATE(epexpep)(con(2.0f * y, 0.0f));
    float2 t = fdiv(fsub(e, 1.0f), fadd(e, 1.0f));
    float z = t.hi;

    z = y > 10.0f ? 1.0f : z;
    z = y < 0x1.0p-13f ? y : z;
#else
    float z;
    if (y < 0.625f) {
        float y2 = y*y;
        float p = PE4(y2, -0x1.758e7ap-8f, 0x1.521192p-6f, -0x1.b8389cp-5f, 0x1.110704p-3f, -0x1.555532p-2f);
        z = MATH_MAD(y2, y*p, y);
    } else {
        float t = MATH_MANGLE(exp)(2.0f * y);
#if defined MORE_ACCURACY
        // h + hl = (t + 1)/2, z = 1 - 1/(h + hl) with one final rounding
        float h = BUILTIN_FMA_F32(0.5f, t, 0.5f);
        float hl = BUILTIN_FMA_F32(0.5f, t, -h) + 0.5f;
        float w = BUILTIN_AMDGPU_RCP_F32(h);
        float e = BUILTIN_FMA_F32(-hl, w, BUILTIN_FMA_F32(-h, w, 1.0f));
        float s = 1.0f - w;
        float sl = (1.0f - s) - w;
        z = s + BUILTIN_FMA_F32(-e, w, sl);
        z = y > 10.0f ? 1.0f : z;
#else
        z = MATH_MAD(-2.0f, MATH_FAST_RCP(t + 1.0f), 1.0f);
#endif
    }
#endif

    return BUILTIN_COPYSIGN_F32(z, x);
}

